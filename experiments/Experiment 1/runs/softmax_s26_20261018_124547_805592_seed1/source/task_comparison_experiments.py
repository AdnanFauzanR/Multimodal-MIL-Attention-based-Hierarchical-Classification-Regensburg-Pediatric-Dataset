"""Experiment 1: matched softmax, diagnosis and paper-defined severity controls."""
from contextlib import redirect_stdout
from dataclasses import asdict
from pathlib import Path
import json
import shutil
import time
import traceback

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import auc as integrate_auc, average_precision_score, precision_recall_curve, roc_curve
from torch.utils.data import DataLoader

from multimodal_mil_model import ModelConfig, MultimodalMILModel, build_optimizer, count_parameters, labels_to_class
from binary_training import BinaryMILModel
from training import fit, compute_metrics, run_epoch, set_seed
from experiment_runner import Tee, collate_with_ids, file_sha256, fingerprint, write_json, release_gpu
from experiment_evaluation import binary_report, summarize, bootstrap_ci, GROUPS

VARIANTS = ('softmax', 'diagnosis', 'severity')


class DualFusionSoftmaxModel(MultimodalMILModel):
    """Keep both attention/fusion streams; replace hierarchical heads with CE head.

    Stream names are inherited for state-dict continuity, but neither stream has
    an independently supervised diagnosis/severity loss in this flat model.
    """
    def __init__(self, cfg):
        if cfg.head != 'softmax' or not cfg.dual_attention:
            raise ValueError('Softmax control requires head=softmax and dual attention')
        super().__init__(ModelConfig(**{**cfg.to_dict(), 'head': 'dual_head'}))
        dim = self.head_diagnosis.in_features
        del self.head_diagnosis, self.head_complication
        self.head_softmax = nn.Linear(2*dim, 3)
        self.cfg = cfg

    def forward(self, images, image_mask, tabular, return_attention=False):
        self._check_inputs(images, image_mask, tabular)
        clinical, sparse, masks = self.tabular(tabular, return_masks=return_attention)
        d, c, ad, ac = self.aggregator(self.encode_images(images, image_mask), image_mask)
        fd = self.fusion_diagnosis(torch.cat([d, clinical], 1))
        fc = self.fusion_complication(torch.cat([c, clinical], 1))
        output = {'logits': self.head_softmax(torch.cat([fd, fc], 1)), 'tab_sparsity_loss': sparse}
        if return_attention:
            output.update(attention_stream1=ad, attention_stream2=ac, tabular_masks=masks)
        return output


class PaperSeverityMILModel(BinaryMILModel):
    """Complicated vs uncomplicated OR no appendicitis, matching paper Table 6."""
    def __init__(self, cfg):
        super().__init__(cfg, 'severity')

    def targets(self, batch):
        return ((batch['y_appendicitis'] == 1) & (batch['y_complicated'] == 1)).long()


def make_model(cfg, variant):
    if variant == 'softmax': return DualFusionSoftmaxModel(cfg)
    if variant == 'diagnosis': return BinaryMILModel(cfg, 'diagnosis')
    if variant == 'severity': return PaperSeverityMILModel(cfg)
    raise ValueError('Unknown comparison variant')


def binary_macro_metrics(y, pred):
    result = compute_metrics(y, pred, num_classes=2)
    result['per_class'].index = ['negative', 'positive']
    return result


def aupr_details(y, probability):
    if len(np.unique(y)) != 2:
        return {'auprc_ap': None, 'auprc_trapezoid': None}
    precision, recall, _ = precision_recall_curve(y, probability)
    return {'auprc_ap': float(average_precision_score(y, probability)),
            'auprc_trapezoid': float(integrate_auc(recall, precision))}


def binary_full_report(y, probability, predicted=None):
    return {**binary_report(y, probability, predicted=predicted), **aupr_details(y, probability)}


def comparison_epoch(model, loader, device, optimizer=None, scaler=None, use_amp=False):
    training = optimizer is not None
    if isinstance(model, DualFusionSoftmaxModel):
        return run_epoch(model, loader, device, optimizer, scaler, use_amp=bool(use_amp and training))
    model.train(training)
    sums, targets, scores, count = {}, [], [], 0
    with torch.set_grad_enabled(training):
        for batch in loader:
            b = {k:v.to(device) for k,v in batch.items()}
            y = model.targets(b)
            with torch.autocast(device_type=torch.device(device).type, enabled=bool(use_amp and training)):
                output = model(b['images'], b['image_mask'], b['tabular'])
                loss, parts = model.compute_loss(output, y)
            if not torch.isfinite(loss): raise FloatingPointError('Non-finite comparison loss')
            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            for key,value in parts.items(): sums[key] = sums.get(key, 0.) + value*len(y)
            count += len(y)
            targets.extend(y.detach().cpu().tolist())
            scores.extend(output['logit'].detach().float().sigmoid().cpu().tolist())
    if not count: raise ValueError('Empty loader')
    summary = {('loss' if k=='loss_total' else k):v/count for k,v in sums.items()}
    report = binary_full_report(targets, scores)
    summary.update(auroc=report['auroc'], **aupr_details(targets, scores))
    return summary, np.array(targets), (np.array(scores)>=.5).astype(int)


@torch.no_grad()
def predict_comparison(model, loader, device, variant):
    model.eval(); rows=[]
    for batch in loader:
        b={k:v.to(device) for k,v in batch.items()}
        with torch.autocast(device_type=torch.device(device).type,enabled=False):
            output=model(b['images'].float(),b['image_mask'],b['tabular'].float())
            y=labels_to_class(b['y_appendicitis'],b['y_complicated'])
            if variant=='softmax':
                probs=model.predict_proba(output).cpu().numpy()
                positive=probs[:,1:].sum(1)
                q=np.divide(probs[:,2],positive,out=np.zeros_like(positive),where=positive>0)
            else:positive=output['logit'].float().sigmoid().cpu().numpy()
        for i,pid in enumerate(b['patient_id'].tolist()):
            row={'patient_id':pid,'n_images':int(b['image_mask'][i].sum()),'y_true':int(y[i]),
                 'y_appendicitis':int(b['y_appendicitis'][i]),'y_complicated':int(b['y_complicated'][i])}
            if variant=='softmax':
                row.update(prob_no=float(probs[i,0]),prob_uncomplicated=float(probs[i,1]),prob_complicated=float(probs[i,2]),
                           p_diagnosis=float(positive[i]),q_complication=float(q[i]),
                           y_pred=int(probs[i].argmax()),y_pred_argmax=int(probs[i].argmax()))
            else:
                row.update(target_binary=int(row['y_appendicitis'] if variant=='diagnosis' else row['y_true']==2),
                           probability_positive=float(positive[i]),y_pred_binary=int(positive[i]>=.5))
            rows.append(row)
    frame=pd.DataFrame(rows)
    if len(frame)!=len(loader.dataset) or frame.patient_id.duplicated().any():
        raise ValueError('Patient predictions not aligned')
    probability_cols=['prob_no','prob_uncomplicated','prob_complicated'] if variant=='softmax' else ['probability_positive']
    values=frame[probability_cols].to_numpy()
    if not np.isfinite(values).all() or (values<0).any() or (values>1).any():
        raise ValueError('Invalid probabilities')
    if variant=='softmax' and not np.allclose(values.sum(1),1.,atol=1e-6):
        raise ValueError('Softmax probabilities do not sum to one')
    return frame


def metrics_for_frame(frame, variant):
    if variant!='softmax':
        result=binary_full_report(frame.target_binary,frame.probability_positive)
        result['definition']='appendicitis_vs_no' if variant=='diagnosis' else 'complicated_vs_no_or_uncomplicated'
        return result
    result=summarize(frame)
    for key in ('severity_after_gating_true_appendicitis','complicated_end_to_end_all_patients'):
        result[key]['decision_rule']='argmax(P_no,P_uncomplicated,P_complicated)==2'
    result['prediction_rule']='argmax_three_classes'
    result['diagnosis_marginal']=binary_full_report(frame.y_appendicitis,frame.p_diagnosis)
    result['severity_marginal_all']=binary_full_report(frame.y_true.eq(2),frame.prob_complicated)
    app=frame[frame.y_appendicitis.eq(1)]
    result['severity_conditional_direct']=binary_full_report(app.y_complicated,app.q_complication)
    return result


def patient_ids(dataset):
    return list(map(int,dataset.patient_ids))


def save_learning(history, directory):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig=Figure(figsize=(12,4));FigureCanvasAgg(fig)
    for ax,key in zip(fig.subplots(1,3),('loss','f1','auroc')):
        for split in ('train','val'): ax.plot(history.index,history[f'{split}_{key}'],label=split)
        ax.axvline(history.attrs['best_epoch'],color='gray',ls='--')
        ax.set(title=key,xlabel='Epoch');ax.legend()
    fig.tight_layout();fig.savefig(directory/'training_curves.png',dpi=150)


def train_comparison(directory, variant, cfg, opt_cfg, datasets, settings, provenance):
    directory=Path(directory)
    if variant not in VARIANTS or settings['monitor']!='val_f1' or settings['early_stopping_rule']!='best_monitor':
        raise ValueError('Expected fixed task variant and best validation macro-F1')
    spec={'variant':variant,'model':cfg.to_dict(),'optimizer':asdict(opt_cfg),'training':settings,'provenance':provenance,
          'target_definition':('three_class_softmax' if variant=='softmax' else
              'appendicitis_vs_no' if variant=='diagnosis' else 'complicated_vs_no_or_uncomplicated'),
          'monitor_definition':'macro_f1_three_classes' if variant=='softmax' else 'macro_f1_two_classes',
          'architecture':'dual_attention_dual_fusion_joint_softmax' if variant=='softmax' else 'one_task_attention_fusion_sigmoid',
          'patient_ids':{s:patient_ids(datasets[s]) for s in ('train','val','test')}}
    signature=fingerprint(spec)
    if directory.exists():
        if not (directory/'result.json').exists():raise ValueError('Incomplete run: start a new run; weights are not resumed')
        previous=json.loads((directory/'config.json').read_text())
        if previous['fingerprint']!=signature:raise ValueError('Configuration/code changed: use a new run directory')
        result=json.loads((directory/'result.json').read_text())
        if file_sha256(directory/'best_model.pt')!=result['checkpoint_sha256']:raise ValueError('Checkpoint changed')
        return result
    directory.mkdir(parents=True)
    write_json(directory/'config.json',{**spec,'fingerprint':signature})
    write_json(directory/'status.json',{'status':'training'})
    if provenance.get('project_root'):
        root=Path(provenance['project_root']);source=directory/'source';source.mkdir()
        for name,digest in provenance['source_sha256'].items():
            path=root/name
            if file_sha256(path)!=digest:raise ValueError(f'Source changed since configuration: {name}')
            shutil.copy2(path,source/path.name)
        if provenance.get('notebook_path'):shutil.copy2(provenance['notebook_path'],source/Path(provenance['notebook_path']).name)
        shutil.copy2(Path(provenance['reference_dir'])/'split_manifest.csv',directory/'split_manifest.csv')
    model=optimizer=None;start=time.perf_counter();device=torch.device(settings['device'])
    try:
        set_seed(settings['seed']);model=make_model(cfg,variant).to(device)
        optimizer=build_optimizer(model,opt_cfg)
        write_json(directory/'parameters.json',count_parameters(model))
        set_seed(settings['seed'])
        common=dict(batch_size=settings['batch_size'],num_workers=settings['num_workers'],collate_fn=collate_with_ids)
        train_loader=DataLoader(datasets['train'],shuffle=True,drop_last=True,
            generator=torch.Generator().manual_seed(settings['seed']),**common)
        val_loader=DataLoader(datasets['val'],shuffle=False,**common)
        if not len(train_loader):raise ValueError('Empty training loader')
        metric_fn=compute_metrics if variant=='softmax' else binary_macro_metrics
        with (directory/'training.log').open('w',encoding='utf-8') as log,redirect_stdout(Tee(__import__('sys').stdout,log)):
            print(json.dumps(spec,indent=2))
            history=fit(model,train_loader,val_loader,optimizer,device,settings['epochs'],monitor='val_f1',
                use_amp=settings['use_amp'],checkpoint_path=directory/'best_model.pt',restore_best=True,
                early_stopping_patience=settings['patience'],early_stopping_min_delta=settings['min_delta'],
                early_stopping_rule='best_monitor',checkpoint_tie_break='val_loss',
                epoch_runner=comparison_epoch,metric_fn=metric_fn,verbose=settings.get('verbose',True))
        history.to_csv(directory/'history.csv');write_json(directory/'history_metadata.json',history.attrs)
        if settings.get('save_plots',True):save_learning(history,directory)
        metrics={}
        for split,key in [('train','train_eval'),('val','val')]:
            frame=predict_comparison(model,DataLoader(datasets[key],shuffle=False,**common),device,variant)
            frame.to_csv(directory/f'{split}_predictions.csv',index=False)
            metrics[split]=metrics_for_frame(frame,variant);write_json(directory/f'{split}_metrics.json',metrics[split])
        f1=metrics['val']['overall']['macro_f1'] if variant=='softmax' else metrics['val']['macro_f1']
        if not np.isclose(f1,history.attrs['best_value'],atol=1e-6,rtol=0):
            raise ValueError(f'FP32 checkpoint F1 {f1} differs from selected value {history.attrs["best_value"]}')
        result={'status':'completed','variant':variant,'best_epoch':history.attrs['best_epoch'],
                'checkpoint_sha256':file_sha256(directory/'best_model.pt'),'seconds_total':time.perf_counter()-start,
                'train':metrics['train'],'val':metrics['val']}
        write_json(directory/'result.json',result);write_json(directory/'status.json',{'status':'completed'})
        return result
    except BaseException as error:
        (directory/'error.log').write_text(traceback.format_exc(),encoding='utf-8')
        write_json(directory/'status.json',{'status':'interrupted' if isinstance(error,KeyboardInterrupt) else 'failed','error':str(error)})
        raise
    finally:
        model=optimizer=None;release_gpu()


def bootstrap_binary(frame, seed, repeats):
    rng=np.random.default_rng(seed);groups=[np.flatnonzero(frame.target_binary.to_numpy()==v) for v in (0,1)]
    groups=[g for g in groups if len(g)];samples={key:[] for key in ('accuracy','macro_f1','auroc','auprc_ap','auprc_trapezoid')}
    for _ in range(repeats if groups else 0):
        f=frame.iloc[np.concatenate([rng.choice(g,len(g),replace=True) for g in groups])]
        m=binary_full_report(f.target_binary,f.probability_positive)
        for key in samples:
            if m[key] is not None:samples[key].append(m[key])
    return {key:{'lower':float(np.percentile(v,2.5)) if v else None,'upper':float(np.percentile(v,97.5)) if v else None}
            for key,v in samples.items()}


def test_comparison(directory, datasets):
    directory=Path(directory);spec=json.loads((directory/'config.json').read_text())
    result=json.loads((directory/'result.json').read_text());variant=spec['variant'];settings=spec['training']
    if result['status']!='completed' or file_sha256(directory/'best_model.pt')!=result['checkpoint_sha256']:
        raise ValueError('Incomplete or altered checkpoint')
    if patient_ids(datasets['test'])!=spec['patient_ids']['test']:raise ValueError('Test patients changed')
    model=None
    try:
        model=make_model(ModelConfig(**{**spec['model'],'pretrained':False}),variant)
        model.load_state_dict(torch.load(directory/'best_model.pt',map_location='cpu',weights_only=True)['model_state'])
        device=torch.device(settings['device']);model.to(device)
        loader=DataLoader(datasets['test'],batch_size=settings['batch_size'],shuffle=False,
                          num_workers=settings['num_workers'],collate_fn=collate_with_ids)
        frame=predict_comparison(model,loader,device,variant)
        output=directory/'testing';output.mkdir(exist_ok=True)
        frame.to_csv(output/'test_predictions.csv',index=False)
        groups={};rows=[]
        for group in GROUPS:
            f=frame if group=='all' else frame[frame.n_images.eq(1) if group=='single_image' else frame.n_images.gt(1)]
            m=metrics_for_frame(f,variant)
            m['bootstrap_95ci']=(bootstrap_ci(f,settings['test_bootstrap_resamples'],settings['seed']) if variant=='softmax'
                                 else bootstrap_binary(f,settings['seed'],settings['test_bootstrap_resamples']))
            groups[group]=m;write_json(output/f'{variant}_{group}_metrics.json',m)
            rows.append({'group':group,**(m['overall'] if variant=='softmax' else {k:v for k,v in m.items() if not isinstance(v,(dict,list))})})
        if variant!='softmax':
            f=frame[frame.y_appendicitis.eq(1)].copy()
            if variant=='severity':groups['true_appendicitis_only']=metrics_for_frame(f,variant)
        report={'variant':variant,'checkpoint_sha256':result['checkpoint_sha256'],
                'target_definition':spec['target_definition'],'groups':groups}
        write_json(output/'report.json',report);pd.DataFrame(rows).to_csv(output/'overview.csv',index=False)
        save_test_curves(frame,variant,output)
        return report
    finally:
        model=None;release_gpu()


def save_test_curves(frame,variant,output):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig=Figure(figsize=(11,4));FigureCanvasAgg(fig);axes=fig.subplots(1,2)
    tasks=[('diagnosis',frame.y_appendicitis,frame.p_diagnosis),('severity_all',frame.y_true.eq(2),frame.prob_complicated)] if variant=='softmax' else [(variant,frame.target_binary,frame.probability_positive)]
    for label,y,score in tasks:
        if len(np.unique(y))<2:continue
        x,t,_=roc_curve(y,score);p,r,_=precision_recall_curve(y,score)
        axes[0].plot(x,t,label=label);axes[1].plot(r,p,label=label)
    for ax in axes:
        if ax.get_legend_handles_labels()[0]:ax.legend()
    axes[0].set(title='ROC',xlabel='FPR',ylabel='Sensitivity');axes[1].set(title='Precision-recall',xlabel='Recall',ylabel='Precision')
    fig.tight_layout();fig.savefig(output/'roc_pr.png',dpi=150)


def local_row(label,task,frame,score,prediction=None,rule='',note=''):
    y=frame.y_appendicitis if task=='diagnosis' else frame.y_true.eq(2)
    report=binary_full_report(y,score,predicted=prediction)
    keys=('accuracy','precision','sensitivity','specificity','f1','macro_f1','auroc','auprc_ap','auprc_trapezoid')
    return {'model':label,'task':task,'source_type':'local_same_test_patients','status':'evaluated',
            'patients':len(frame),'positive_cases':int(y.sum()),'initializations':1,
            **{k:report[k] for k in keys},'decision_rule':rule,'note':note}


def export_paper_comparisons(reference_dir, runs_root, current_run, baseline_csv, protocol):
    """Latest completed matching runs, chosen by timestamp, never by test score.

    Missing runs stay pending. Literature rows are as reported, not reproductions.
    """
    import html
    reference_dir,runs_root,current_run=map(Path,(reference_dir,runs_root,current_run))
    selection=json.loads((reference_dir/'best_model.json').read_text())
    reference_trial=reference_dir/'trials'/selection['winner']['scenario_id']
    ref=pd.read_csv(reference_trial/'testing/multimodal_predictions.csv')
    baselines=pd.read_csv(baseline_csv)
    selected={};provenance=[]
    for variant in VARIANTS:
        matches=[]
        for folder in sorted(runs_root.glob(f'{variant}_s26_*')):
            if not (folder/'result.json').exists():continue
            cfg=json.loads((folder/'config.json').read_text())
            if cfg['provenance'].get('comparison_protocol')==protocol:matches.append(folder)
        if matches:
            latest=current_run if current_run in matches else matches[-1]
            tested=latest/'testing/test_predictions.csv'
            selected[variant]=pd.read_csv(tested) if tested.exists() else None
            provenance.append({'variant':variant,'directory':str(latest.resolve()),
                'checkpoint_sha256':file_sha256(latest/'best_model.pt'),'test_available':tested.exists()})
    for frame in selected.values():
        if frame is None:continue
        a,b=ref.sort_values('patient_id'),frame.sort_values('patient_id')
        np.testing.assert_array_equal(a[['patient_id','y_true','n_images']],b[['patient_id','y_true','n_images']])
    local={};tables={}
    for task in ('diagnosis','severity'):
        target=(ref.y_pred.gt(0) if task=='diagnosis' else ref.y_pred.eq(2))
        score=ref.p_diagnosis if task=='diagnosis' else ref.prob_complicated
        rule='p>=0.5' if task=='diagnosis' else 'p>=0.5 AND q>=0.5'
        rows=[local_row('proposed_hierarchical',task,ref,score,target,rule),
              local_row('proposed_task_without_retraining',task,ref,score,rule='marginal_probability>=0.5',
                        note='Same checkpoint and ranking scores as hierarchical; diagnosis decisions are identical. Severity uses pq, not q, on all patients.')]
        for variant,label in [(task,'proposed_task_retrained'),('softmax','proposed_softmax')]:
            frame=selected.get(variant)
            if frame is None:
                rows.append({'model':label,'task':task,'source_type':'local_same_test_patients','status':'pending_training_or_test'})
            else:
                score=(frame.p_diagnosis if task=='diagnosis' else frame.prob_complicated) if variant=='softmax' else frame.probability_positive
                rows.append(local_row(label,task,frame,score,rule='marginal_probability>=0.5' if variant=='softmax' else 'sigmoid>=0.5',
                    note='Derived binary marginal; primary softmax 3-class prediction remains argmax.' if variant=='softmax' else 'Fresh single-task model; no parent weights loaded.'))
        local[task]=pd.DataFrame(rows)
        published=baselines[(baselines.paper.eq('Marcinkevics2024'))&baselines.task.eq(task)]
        tables[f'marcinkevics_{task}']=pd.concat([published,local[task]],ignore_index=True)
    published=baselines[baselines.paper.eq('Barthwal2026')]
    tables['multimodal_paper_diagnosis']=pd.concat([published,local['diagnosis']],ignore_index=True)
    app=ref[ref.y_appendicitis.eq(1)]
    conditional=[local_row('proposed_hierarchical_gated','severity',app,app.prob_complicated,app.y_pred.eq(2),'p>=0.5 AND q>=0.5'),
                 local_row('proposed_q_without_retraining','severity',app,app.q_complication,rule='q>=0.5',note='Conditional severity, excluded from paper all-patient comparison.')]
    for variant,label in [('severity','severity_retrained_all_patients'),('softmax','proposed_softmax_conditional')]:
        frame=selected.get(variant)
        if frame is None:continue
        f=frame[frame.y_appendicitis.eq(1)]
        score=f.q_complication if variant=='softmax' else f.probability_positive
        conditional.append(local_row(label,'severity',f,score,rule='conditional_q>=0.5' if variant=='softmax' else 'unconditional_complicated_score>=0.5',
            note='All rows restricted to true appendicitis; not the paper severity denominator.'))
    tables['conditional_severity_not_paper_target']=pd.DataFrame(conditional)
    if selected.get('softmax') is not None:
        f=selected['softmax'];r=metrics_for_frame(f,'softmax');parent=summarize(ref)
        tables['three_class_softmax_vs_hierarchical']=pd.DataFrame([
            {'model':name,**m['overall']} for name,m in [('hierarchical',parent),('softmax_dual_fusion',r)]])
        tables['softmax_argmax_collapsed_binary']=pd.DataFrame([
            local_row('softmax_argmax_collapsed','diagnosis',f,f.p_diagnosis,f.y_pred.gt(0),'argmax_class>0'),
            local_row('softmax_argmax_collapsed','severity',f,f.prob_complicated,f.y_pred.eq(2),'argmax_class==2')])
    output=current_run/'comparisons';output.mkdir(exist_ok=True)
    paragraphs=['<h1>Experiment 1: published references and matched local controls</h1>',
        '<p>Literature rows have different cohorts/splits and ten initializations for Marcinkevics, versus one local seed. They are not matched replications. No ranking or significance claim is made.</p>',
        '<p>Scores are fractions (0–1). F1 is positive-class binary F1; macro-F1 is a separate column and is the checkpoint monitor. AP and trapezoidal PR area are both reported; the paper AUPR implementation is not assumed to be identical.</p>',
        '<p>Paper severity = complicated versus uncomplicated OR no appendicitis (all patients). Direct q on true appendicitis is reported separately. Original hierarchy and direct marginalization share ranking scores; identical AUROC/AP is expected, and diagnosis decisions are identical.</p>',
        '<p>Softmax binary tables use marginal probabilities at threshold 0.5. Original argmax-derived binary decisions are exported separately. Retained runs are the most recent completed matching runs, not the highest test scores. Untested/missing models remain pending.</p>']
    for name,frame in tables.items():
        frame.to_csv(output/f'{name}.csv',index=False)
        paragraphs.extend([f'<h2>{html.escape(name)}</h2>',frame.to_html(index=False,na_rep='not available',float_format=lambda v:f'{v:.4f}')])
    (output/'comparison.html').write_text('<!doctype html><meta charset="utf-8">'+''.join(paragraphs),encoding='utf-8')
    write_json(output/'sources.json',{'reference_dir':str(reference_dir.resolve()),'protocol':protocol,
        'baseline_csv_sha256':file_sha256(baseline_csv),'runs':provenance,'test_used_for_selection':False})
    return tables
