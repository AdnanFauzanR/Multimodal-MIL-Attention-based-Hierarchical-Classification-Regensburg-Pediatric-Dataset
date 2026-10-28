"""Evaluasi pasien, subgroup bag, dan intervensi modalitas tanpa melatih ulang model."""
from pathlib import Path
from statistics import NormalDist
import html
import json

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_curve, precision_recall_curve

from multimodal_mil_model import CLASS_NAMES, labels_to_class


MODES = ("multimodal", "image_only_intervention", "clinical_only_intervention")
GROUPS = ("all", "single_image", "multiple_images")
PROB = ["prob_no", "prob_uncomplicated", "prob_complicated"]


def ratio(a, b):
    return float(a / b) if b else None


def wilson(successes, total):
    if not total:
        return None
    z = NormalDist().inv_cdf(.975)
    p = successes / total
    center = (p + z*z/(2*total)) / (1 + z*z/total)
    radius = z*np.sqrt(p*(1-p)/total + z*z/(4*total*total)) / (1 + z*z/total)
    return [max(0., float(center-radius)), min(1., float(center+radius))]


def auc(y, scores):
    y, scores = np.asarray(y, dtype=bool), np.asarray(scores, dtype=float)
    positive, negative = int(y.sum()), int((~y).sum())
    if not positive or not negative:
        return None
    return float((rankdata(scores)[y].sum()-positive*(positive+1)/2)/(positive*negative))


def binary_report(y, score, predicted=None, threshold=.5):
    y, score = np.asarray(y, dtype=bool), np.asarray(score, dtype=float)
    pred = score >= threshold if predicted is None else np.asarray(predicted, dtype=bool)
    tp, tn = int((y & pred).sum()), int((~y & ~pred).sum())
    fp, fn = int((~y & pred).sum()), int((y & ~pred).sum())
    sensitivity, specificity = ratio(tp,tp+fn), ratio(tn,tn+fp)
    denominator = np.sqrt(float((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
    probability = np.clip(score, np.finfo(float).eps, 1-np.finfo(float).eps)
    positive_f1, negative_f1 = ratio(2*tp,2*tp+fp+fn), ratio(2*tn,2*tn+fp+fn)
    present_f1 = ([positive_f1 or 0.] if tp+fn else []) + ([negative_f1 or 0.] if tn+fp else [])
    return {
        "status": "ok" if len(y) else "empty", "patients": len(y),
        "support_positive": tp+fn, "support_negative": tn+fp,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": ratio(tp+tn,len(y)), "precision": ratio(tp,tp+fp),
        "ppv": ratio(tp,tp+fp), "npv": ratio(tn,tn+fn),
        "sensitivity": sensitivity, "recall": sensitivity, "specificity": specificity,
        "f1": positive_f1, "negative_class_precision": ratio(tn,tn+fn),
        "negative_class_recall": specificity, "negative_class_f1": negative_f1,
        "macro_f1": float(np.mean(present_f1)) if present_f1 else None,
        "weighted_f1": ((positive_f1 or 0.)*(tp+fn)+(negative_f1 or 0.)*(tn+fp))/len(y) if len(y) else None,
        "fpr": ratio(fp,fp+tn), "fnr": ratio(fn,fn+tp),
        "balanced_accuracy": (sensitivity+specificity)/2 if sensitivity is not None and specificity is not None else None,
        "mcc": ratio(tp*tn-fp*fn,denominator),
        "auroc": auc(y,score),
        "auprc_average_precision": float(average_precision_score(y,score)) if y.any() and (~y).any() else None,
        "brier": float(np.square(score-y).mean()) if len(y) else None,
        "log_loss": float(-(y*np.log(probability)+(~y)*np.log1p(-probability)).mean()) if len(y) else None,
        "confusion_matrix": [[tn,fp],[fn,tp]], "confusion_matrix_order": ["negative","positive"],
        "wilson_95ci": {"accuracy": wilson(tp+tn,len(y)), "sensitivity": wilson(tp,tp+fn),
                        "specificity": wilson(tn,tn+fp), "ppv": wilson(tp,tp+fp), "npv": wilson(tn,tn+fn)},
    }


def macro_scores(y, pred):
    cm = np.bincount(3*y+pred,minlength=9).reshape(3,3)
    tp, actual, predicted = cm.diagonal(), cm.sum(1), cm.sum(0)
    present = actual > 0
    f1 = np.divide(2*tp,actual+predicted,out=np.zeros(3,dtype=float),where=actual+predicted>0)
    return cm, float(f1[present].mean()) if present.any() else None


def summarize(frame, threshold=.5):
    y = frame.y_true.to_numpy(dtype=int)
    pred = frame.y_pred.to_numpy(dtype=int)
    p, q = frame.p_diagnosis.to_numpy(), frame.q_complication.to_numpy()
    probability = frame[PROB].to_numpy(dtype=float)
    n = len(frame)
    cm, macro_f1 = macro_scores(y,pred)
    per_class = []
    for i,name in enumerate(CLASS_NAMES):
        r = binary_report(y==i,probability[:,i],pred==i,threshold)
        per_class.append({"class":name,"class_index":i,
            "support":r["support_positive"],"predicted_support":int((pred==i).sum()),
            **{k:r[k] for k in ["tp","tn","fp","fn","precision","ppv","npv","sensitivity","specificity","f1","fpr","fnr","balanced_accuracy","mcc","auroc","auprc_average_precision"]},
            "recall":r['sensitivity'], "accuracy_within_class":r['sensitivity'],
            "one_vs_rest_accuracy":r['accuracy'], "recall_wilson_95ci":r['wilson_95ci']['sensitivity'],
            "wilson_95ci":r['wilson_95ci']})
    present = [r for r in per_class if r['support']]
    def mean(key):
        # Tidak memprediksi kelas yang hadir -> precision/F1 nol untuk macro.
        if key == 'specificity':
            valid = [r[key] for r in present if r[key] is not None]
            return float(np.mean(valid)) if valid else None
        return float(np.mean([r[key] if r[key] is not None else 0. for r in present])) if present else None
    total_pred, total_true = cm.sum(0), cm.sum(1)
    denominator = np.sqrt(float((n*n-(total_pred**2).sum())*(n*n-(total_true**2).sum())))
    p_clip = np.clip(probability,np.finfo(float).eps,1.)
    predicted_argmax = probability.argmax(1) if n else np.array([],dtype=int)
    confidence = probability.max(1) if n else np.array([])
    calibration = []
    ece = 0.
    for lo in np.arange(10)/10:
        selected = (confidence>=lo) & ((confidence<lo+.1) if lo<.9 else (confidence<=1))
        if selected.any():
            accuracy = float((predicted_argmax[selected]==y[selected]).mean())
            avg = float(confidence[selected].mean())
            ece += selected.sum()/n*abs(accuracy-avg)
            calibration.append({'bin_low':float(lo),'patients':int(selected.sum()),'confidence':avg,'accuracy':accuracy})
    overall = {"patients":n,"accuracy":ratio(int(np.trace(cm)),n),"f1":macro_f1,
        "macro_f1":macro_f1,"precision":mean('precision'),"sensitivity":mean('sensitivity'),
        "specificity":mean('specificity'),"balanced_accuracy":mean('sensitivity'),
        "weighted_f1":sum((r['f1'] or 0)*r['support'] for r in per_class)/n if n else None,
        "mcc":ratio(int(np.trace(cm))*n-int(total_pred@total_true),denominator),
        "cohen_kappa":ratio(int(np.trace(cm))*n-int(total_pred@total_true),n*n-int(total_pred@total_true)),
        "auroc_macro_ovr":mean('auroc') if len(present)==3 else None,
        "auprc_macro_average_precision":mean('auprc_average_precision') if len(present)==3 else None,
        "nll_three_classes":float(-np.log(p_clip[np.arange(n),y]).mean()) if n else None,
        "brier_three_classes":float(np.square(probability-np.eye(3)[y]).sum(1).mean()) if n else None,
        "ece_top_label_argmax_10bins":float(ece) if n else None,
        "argmax_accuracy":ratio(int((predicted_argmax==y).sum()),n),
        "accuracy_wilson_95ci":wilson(int(np.trace(cm)),n)}
    app = y>0
    diagnosis = binary_report(app,p,threshold=threshold)
    severity = binary_report(y[app]==2,q[app],threshold=threshold)
    gated = binary_report(y[app]==2,(p*q)[app],pred[app]==2,threshold)
    end_to_end = binary_report(y==2,p*q,pred==2,threshold)
    diagnosis.update(positive_class='appendicitis',negative_class='no appendicitis')
    severity.update(positive_class='complicated',negative_class='uncomplicated')
    gated.update(positive_class='complicated',negative_class='uncomplicated')
    end_to_end.update(positive_class='complicated',negative_class='no or uncomplicated')
    gated['decision_rule'] = end_to_end['decision_rule'] = f'p >= {threshold} AND q >= {threshold}'
    gated['ranking_score'] = end_to_end['ranking_score'] = 'p*q; decision is not threshold(p*q)'
    return {"status":"ok" if n else "empty","overall":overall,"per_class":per_class,
        "confusion_matrix":cm.tolist(),"diagnosis":diagnosis,
        "severity_direct_true_appendicitis":severity,
        "severity_after_gating_true_appendicitis":gated,
        "complicated_end_to_end_all_patients":end_to_end,"calibration_top_label":calibration}


def bootstrap_ci(frame, resamples, seed):
    if resamples<0:
        raise ValueError('bootstrap_resamples harus >=0')
    if not len(frame) or not resamples:
        return {'method':'patient_stratified_percentile','resamples':resamples,'metrics':{}}
    rng = np.random.default_rng(seed)
    y = frame.y_true.to_numpy(dtype=int)
    pred = frame.y_pred.to_numpy(dtype=int)
    probabilities = frame[PROB].to_numpy()
    p,q = frame.p_diagnosis.to_numpy(),frame.q_complication.to_numpy()
    strata = [np.flatnonzero(y==i) for i in np.unique(y)]
    samples = {k:[] for k in ['accuracy','macro_f1','auroc_macro_ovr','diagnosis_auroc',
                              'severity_direct_auroc','complicated_end_to_end_auroc','complicated_recall']}
    for _ in range(resamples):
        idx = np.concatenate([rng.choice(s,len(s),replace=True) for s in strata])
        target,decisions = y[idx],pred[idx]
        _,f1 = macro_scores(target,decisions)
        app=target>0
        values = {'accuracy':float((target==decisions).mean()),'macro_f1':f1,
            'auroc_macro_ovr':float(np.mean([auc(target==i,probabilities[idx,i]) for i in range(3)])) if len(strata)==3 else None,
            'diagnosis_auroc':auc(app,p[idx]),'severity_direct_auroc':auc(target[app]==2,q[idx][app]),
            'complicated_end_to_end_auroc':auc(target==2,p[idx]*q[idx]),
            'complicated_recall':ratio(int(((target==2)&(decisions==2)).sum()),int((target==2).sum()))}
        for key,value in values.items():
            if value is not None:samples[key].append(value)
    return {'method':'patient_stratified_percentile','resamples':resamples,'seed':seed,'metrics':{
        key:{'lower':float(np.percentile(v,2.5)) if v else None,
             'upper':float(np.percentile(v,97.5)) if v else None,'valid_resamples':len(v)} for key,v in samples.items()}}


@torch.no_grad()
def collect_modalities(model, loader, device, threshold):
    if model.cfg.head!='dual_head':
        raise ValueError('Laporan ini memerlukan hierarchical dual_head')
    model.eval()
    rows={mode:[] for mode in MODES}
    for batch in loader:
        b={k:v.to(device) for k,v in batch.items()}
        with torch.autocast(device_type=torch.device(device).type,enabled=False):
            model._check_inputs(b['images'],b['image_mask'],b['tabular'])
            h=model.encode_images(b['images'].float(),b['image_mask'])
            clinical,_,_=model.tabular(b['tabular'].float())
            zd,zc,_,_=model.aggregator(h,b['image_mask'])
            y=labels_to_class(b['y_appendicitis'],b['y_complicated'])
            for mode in MODES:
                e=torch.zeros_like(clinical) if mode=='image_only_intervention' else clinical
                d=torch.zeros_like(zd) if mode=='clinical_only_intervention' else zd
                c=torch.zeros_like(zc) if mode=='clinical_only_intervention' else zc
                output={'logit_diagnosis':model.head_diagnosis(model.fusion_diagnosis(torch.cat([d,e],1))).squeeze(-1),
                        'logit_complication':model.head_complication(model.fusion_complication(torch.cat([c,e],1))).squeeze(-1)}
                probability=model.predict_proba(output).cpu().numpy()
                p=output['logit_diagnosis'].sigmoid().cpu().numpy()
                q=output['logit_complication'].sigmoid().cpu().numpy()
                pred=model.predict_classes(output,threshold).cpu().numpy()
                for i,pid in enumerate(b['patient_id'].cpu().tolist()):
                    rows[mode].append({'patient_id':pid,'n_images':int(b['image_mask'][i].sum()),
                        'y_true':int(y[i]),'y_appendicitis':int(b['y_appendicitis'][i]),
                        'y_complicated':int(b['y_complicated'][i]),'p_diagnosis':float(p[i]),
                        'q_complication':float(q[i]),**dict(zip(PROB,map(float,probability[i]))),
                        'y_pred':int(pred[i]),'y_pred_argmax':int(probability[i].argmax())})
    frames={mode:pd.DataFrame(records) for mode,records in rows.items()}
    for frame in frames.values():
        if len(frame)!=len(loader.dataset) or frame.patient_id.duplicated().any():
            raise ValueError('Jumlah/ID pasien evaluasi tidak valid')
        if not np.isfinite(frame[PROB+['p_diagnosis','q_complication']].to_numpy()).all():
            raise ValueError('Probability inferensi tidak finite')
        if not np.allclose(frame[PROB].sum(axis=1),1.,atol=1e-6):
            raise ValueError('Probability kelas tidak berjumlah satu')
    return frames


def save_plots(frames, directory):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig=Figure(figsize=(14,12));FigureCanvasAgg(fig)
    axes=fig.subplots(3,3)
    for i,mode in enumerate(MODES):
        for j,group in enumerate(GROUPS):
            f=frames[mode]
            if group!='all':f=f[f.n_images.eq(1) if group=='single_image' else f.n_images.gt(1)]
            cm,_=macro_scores(f.y_true.to_numpy(dtype=int),f.y_pred.to_numpy(dtype=int))
            ax=axes[i,j];ax.imshow(cm,cmap='Blues');ax.set_title(f'{mode}\n{group} (n={len(f)})',fontsize=9)
            for a in range(3):
                for b in range(3):ax.text(b,a,str(cm[a,b]),ha='center',va='center')
            ax.set(xticks=range(3),yticks=range(3),xlabel='Predicted class',ylabel='True class')
    fig.tight_layout();fig.savefig(directory/'confusion_matrices.png',dpi=140)
    fig=Figure(figsize=(15,8));FigureCanvasAgg(fig);axes=fig.subplots(2,3)
    for j,(name,target_column,score_column,conditional) in enumerate([
        ('Diagnosis','y_appendicitis','p_diagnosis',False),
        ('Severity direct','y_complicated','q_complication',True),
        ('Complicated end-to-end','y_true','prob_complicated',False)]):
        for mode,frame in frames.items():
            f=frame[frame.y_appendicitis.eq(1)] if conditional else frame
            target=f[target_column].to_numpy()
            if name=='Complicated end-to-end':target=target==2
            if np.unique(target).size!=2:continue
            fpr,tpr,_=roc_curve(target,f[score_column]);precision,recall,_=precision_recall_curve(target,f[score_column])
            axes[0,j].plot(fpr,tpr,label=mode);axes[1,j].plot(recall,precision,label=mode)
        axes[0,j].plot([0,1],[0,1],'--',color='gray')
        axes[0,j].set(title=name,xlabel='False positive rate',ylabel='True positive rate')
        axes[1,j].set(xlabel='Recall',ylabel='Precision')
        if axes[0,j].get_legend_handles_labels()[0]:axes[0,j].legend(fontsize=6)
    fig.tight_layout();fig.savefig(directory/'binary_roc_pr.png',dpi=140)
    f=frames['multimodal'];fig=Figure(figsize=(12,4));FigureCanvasAgg(fig);axes=fig.subplots(1,3)
    for i,name in enumerate(CLASS_NAMES):
        target=f.y_true.eq(i)
        if target.nunique()==2:
            x,y,_=roc_curve(target,f[PROB[i]]);axes[0].plot(x,y,label=name)
            precision,recall,_=precision_recall_curve(target,f[PROB[i]]);axes[1].plot(recall,precision,label=name)
    for mode,frame in frames.items():
        bins=summarize(frame)['calibration_top_label']
        if bins:axes[2].plot([b['confidence'] for b in bins],[b['accuracy'] for b in bins],marker='o',label=mode)
    axes[0].set(title='Multimodal: class ROC',xlabel='FPR',ylabel='TPR')
    axes[1].set(title='Multimodal: class precision-recall',xlabel='Recall',ylabel='Precision')
    axes[2].plot([0,1],[0,1],'--',color='gray');axes[2].set(title='Top-label calibration (argmax)',xlabel='Confidence',ylabel='Accuracy')
    for ax in axes:
        if ax.get_legend_handles_labels()[0]:ax.legend(fontsize=6)
    fig.tight_layout();fig.savefig(directory/'class_roc_pr_calibration.png',dpi=140)


def evaluate_test_suite(model, loader, device, directory, checkpoint_path, threshold=.5,
                        bootstrap_resamples=1000, seed=1, expected_predictions=None):
    from experiment_runner import write_json,file_sha256
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    frames=collect_modalities(model,loader,device,threshold)
    if expected_predictions is not None:
        actual=frames['multimodal'].sort_values('patient_id').reset_index(drop=True)
        expected=expected_predictions.sort_values('patient_id').reset_index(drop=True)
        pd.testing.assert_series_equal(actual.patient_id,expected.patient_id,check_dtype=False)
        np.testing.assert_allclose(actual[PROB],expected[PROB],rtol=1e-5,atol=1e-6)
        np.testing.assert_array_equal(actual.y_pred,expected.y_pred)
    report={'checkpoint_sha256':file_sha256(checkpoint_path),'threshold':threshold,'seed':seed,
        'class_order':list(CLASS_NAMES),'unit':'patient','groups':{},
        'definitions':{
            'intervention':'Zero clinical embedding for image_only; zero both pooled image embeddings for clinical_only. Same trained multimodal weights; no retraining or baseline claim.',
            'grouping':'Original number of images per patient; single=1, multiple>1. No image-level sample counting.',
            'macro_average':'Only true-label-present classes contribute. Absent-class AUROC and undefined per-class ratios are null; macro precision uses zero for a present class never predicted, specificity skips undefined terms.',
            'class_accuracy':'accuracy_within_class is recall TP/(TP+FN); one_vs_rest_accuracy is (TP+TN)/N.',
            'severity':'Direct q evaluated only on true appendicitis. Gated severity includes diagnosis failures. End-to-end complicated is class 2 vs all other patients.',
            'ci':'Wilson 95% for binomial rates; patient bootstrap stratified by true three-class labels for selected aggregate metrics.',
            'auprc':'Average precision, not trapezoidal PR area. Brier multiclass sums squared errors over three classes.',
            'calibration':'Top-label ECE uses argmax probability/accuracy, not the hierarchical threshold decision.',
        }}
    overview=[];class_rows=[];binary_rows=[];ci_rows=[]
    for mode,frame in frames.items():
        frame.to_csv(directory/f'{mode}_predictions.csv',index=False)
        for group in GROUPS:
            subset=frame if group=='all' else frame[frame.n_images.eq(1) if group=='single_image' else frame.n_images.gt(1)]
            result=summarize(subset,threshold)
            result['bootstrap_95ci']=bootstrap_ci(subset,bootstrap_resamples,seed)
            report['groups'][f'{mode}/{group}']=result
            write_json(directory/f'{mode}_{group}_metrics.json',result)
            overview.append({'mode':mode,'group':group,**result['overall'],
                'diagnosis_auroc':result['diagnosis']['auroc'],
                'severity_direct_auroc':result['severity_direct_true_appendicitis']['auroc'],
                'complicated_recall':result['complicated_end_to_end_all_patients']['sensitivity']})
            for row in result['per_class']:class_rows.append({'mode':mode,'group':group,**row})
            for task in ['diagnosis','severity_direct_true_appendicitis','severity_after_gating_true_appendicitis','complicated_end_to_end_all_patients']:
                binary_rows.append({'mode':mode,'group':group,'task':task,**result[task]})
            for name,interval in result['bootstrap_95ci']['metrics'].items():
                ci_rows.append({'mode':mode,'group':group,'metric':name,**interval})
    overview=pd.DataFrame(overview);classes=pd.DataFrame(class_rows);binary=pd.DataFrame(binary_rows)
    overview.to_csv(directory/'overview.csv',index=False)
    classes.to_csv(directory/'per_class.csv',index=False)
    binary.to_csv(directory/'binary_tasks.csv',index=False)
    pd.DataFrame(ci_rows,columns=['mode','group','metric','lower','upper','valid_resamples']).to_csv(directory/'bootstrap_95ci.csv',index=False)
    write_json(directory/'report.json',report)
    save_plots(frames,directory)
    fragments=['<h1>Patient-level test report</h1>',
        '<p>Image-only and clinical-only are embedding interventions on the same multimodal checkpoint, not retrained unimodal baselines.</p>',
        '<p>Missing/undefined metrics are blank/null. Within-class accuracy is recall; one-vs-rest accuracy also counts true negatives.</p>',
        '<h2>Overview</h2>',overview.to_html(index=False,float_format=lambda x:f'{x:.4f}'),
        '<h2>Per-class metrics</h2>',classes.to_html(index=False,float_format=lambda x:f'{x:.4f}'),
        '<h2>Binary tasks</h2>',binary.to_html(index=False,float_format=lambda x:f'{x:.4f}'),
        '<h2>Bootstrap 95% confidence intervals</h2>',pd.DataFrame(ci_rows).to_html(index=False),
        '<h2>Definitions</h2><pre>'+html.escape(json.dumps(report['definitions'],indent=2))+'</pre>']
    for filename in ['confusion_matrices.png','binary_roc_pr.png','class_roc_pr_calibration.png']:
        fragments.append(f'<p><img src="{filename}" style="max-width:100%"></p>')
    (directory/'report.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font-family:Arial;margin:24px}table{border-collapse:collapse;font-size:12px}td,th{padding:5px;border:1px solid #ddd}pre{white-space:pre-wrap}</style>'+''.join(fragments),encoding='utf-8')
    return report


def print_test_report(directory):
    directory=Path(directory)
    print('\nMetrik dalam skala 0-1. Unit evaluasi: pasien.')
    print('Image-only/clinical-only adalah intervensi embedding, bukan baseline yang dilatih terpisah.')
    print('Accuracy within class = recall; accuracy OvR menghitung TN juga. NaN/null = tidak terdefinisi.')
    for filename in ['overview.csv','per_class.csv','binary_tasks.csv','bootstrap_95ci.csv']:
        frame=pd.read_csv(directory/filename)
        print('\n'+filename+'\n'+frame.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    print('\nLaporan HTML:',(directory/'report.html').resolve())
