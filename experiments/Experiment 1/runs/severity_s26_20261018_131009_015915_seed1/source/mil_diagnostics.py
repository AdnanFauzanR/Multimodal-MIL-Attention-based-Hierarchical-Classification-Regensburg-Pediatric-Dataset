"""Deterministic checkpoint analysis: view embeddings, attention, and uniform pooling.

No optimizer steps. Test set should remain held out. Uniform pooling is an
inference intervention on a trained checkpoint, not a separately trained baseline.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from multimodal_mil_model import labels_to_class
from training import binary_metrics, compute_metrics


def transform_saved_tabular(df, reference):
    """Apply the recorded imputer/encoders/scaler without fitting anything."""
    imp=reference['imputation'];num,cat=imp['knn_num'],imp['knn_cat']
    matrix=np.hstack([df[num].to_numpy(dtype=float),imp['encoder'].transform(df[cat])])
    filled=imp['scaler'].inverse_transform(imp['imputer'].transform(imp['scaler'].transform(matrix)))
    out=df.copy()
    numeric=pd.DataFrame(filled[:,:len(num)],index=df.index,columns=num)
    out[num]=df[num].where(df[num].notna(),numeric)
    codes=np.rint(filled[:,len(num):])
    for j,values in enumerate(imp['encoder'].categories_):codes[:,j]=np.clip(codes[:,j],0,len(values)-1)
    categorical=pd.DataFrame(imp['encoder'].inverse_transform(codes),index=df.index,columns=cat)
    out[cat]=df[cat].where(df[cat].notna(),categorical)
    out=out.fillna(imp['high_fill'])
    numeric=pd.DataFrame(reference['numeric_scaler'].transform(out[reference['num_feats']]),
                         index=df.index,columns=reference['num_feats'])
    categorical=pd.DataFrame(reference['category_encoder'].transform(out[reference['cat_feats']]),
                             index=df.index,columns=reference['cat_feats'])
    encoded=pd.concat([numeric,categorical],axis=1).loc[:,reference['feature_cols']].astype(np.float32)
    if not np.isfinite(encoded.to_numpy()).all():raise ValueError('Preprocessing tersimpan menghasilkan NaN/Inf')
    return out,encoded


def _normalized_entropy(alpha):
    if len(alpha) < 2:return float('nan')
    return float(-(alpha*alpha.clamp_min(1e-30).log()).sum()/np.log(len(alpha)))


@torch.no_grad()
def analyze_mil_checkpoint(model, loader, device, output_dir):
    if model.cfg.head != 'dual_head' or model.cfg.aggregation != 'attention':
        raise ValueError('Analisis ini memerlukan attention + dual_head')
    device=torch.device(device)
    model.to(device).eval()
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    bags,instances,predictions,features,feature_ids,image_ids=[],[],[],[],[],[]
    paths_by_patient={pid:paths for pid,paths in zip(loader.dataset.patient_ids,loader.dataset.bags)}
    for batch in loader:
        batch={k:v.to(device) for k,v in batch.items()}
        with torch.autocast(device_type=device.type,enabled=False):
            h=model.encode_images(batch['images'].float(),batch['image_mask'])
            e,_,_=model.tabular(batch['tabular'].float())
            zd,zc,ad,ac=model.aggregator(h,batch['image_mask'])
            mean=(h*batch['image_mask'][...,None]).sum(1)/batch['image_mask'].sum(1,keepdim=True)
            def outputs(d,c):
                return {'logit_diagnosis':model.head_diagnosis(model.fusion_diagnosis(torch.cat([d,e],1))).squeeze(-1),
                        'logit_complication':model.head_complication(model.fusion_complication(torch.cat([c,e],1))).squeeze(-1)}
            normal,uniform=outputs(zd,zc),outputs(mean,mean)
            p,q=normal['logit_diagnosis'].sigmoid(),normal['logit_complication'].sigmoid()
            pu,qu=uniform['logit_diagnosis'].sigmoid(),uniform['logit_complication'].sigmoid()
            cp,cu=model.predict_classes(normal),model.predict_classes(uniform)
            y=labels_to_class(batch['y_appendicitis'],batch['y_complicated'])
        for i,pid_value in enumerate(batch['patient_id']):
            pid=int(pid_value);n=int(batch['image_mask'][i].sum());v=h[i,:n].float()
            if n>1:
                unit=F.normalize(v,dim=1)
                pairwise=(unit@unit.T)[torch.triu(torch.ones(n,n,dtype=torch.bool,device=device),diagonal=1)]
                cosine_mean=float(pairwise.mean());cosine_min=float(pairwise.min())
            else:cosine_mean=cosine_min=float('nan')
            dispersion=float((v-v.mean(0)).norm(dim=1).mean()/v.norm(dim=1).mean().clamp_min(1e-12))
            da,ca=ad[i,:n].float(),ac[i,:n].float()
            bags.append({'patient_id':pid,'n_images':n,'mean_pairwise_cosine':cosine_mean,
                         'min_pairwise_cosine':cosine_min,'relative_embedding_dispersion':dispersion,
                         'attention_D_entropy_normalized':_normalized_entropy(da),
                         'attention_C_entropy_normalized':_normalized_entropy(ca),
                         'attention_D_max':float(da.max()),'attention_C_max':float(ca.max()),
                         'dual_attention_total_variation':float((da-ca).abs().sum()/2)})
            predictions.append({'patient_id':pid,'y_true_class':int(y[i]),'y_appendicitis':int(batch['y_appendicitis'][i]),
                'y_complicated':int(batch['y_complicated'][i]),'p_attention':float(p[i]),'p_uniform':float(pu[i]),
                'q_attention':float(q[i]),'q_uniform':float(qu[i]),'abs_delta_p':float((p[i]-pu[i]).abs()),
                'abs_delta_q':float((q[i]-qu[i]).abs()),'class_attention':int(cp[i]),'class_uniform':int(cu[i]),
                'diagnosis_changed':bool((p[i]>=.5)!=(pu[i]>=.5)),
                'subtype_changed':bool((q[i]>=.5)!=(qu[i]>=.5)),'three_class_changed':bool(cp[i]!=cu[i])})
            for j in range(n):
                image_id=paths_by_patient[pid][j].stem
                instances.append({'patient_id':pid,'image_id':image_id,'instance_index':j,
                    'attention_D':float(da[j]),'attention_C':float(ca[j]),'embedding_norm':float(v[j].norm())})
                features.append(v[j].cpu().numpy());feature_ids.append(pid);image_ids.append(image_id)
    if not predictions:raise ValueError('Loader kosong')
    bags,instances,predictions=map(pd.DataFrame,(bags,instances,predictions))
    if predictions.patient_id.duplicated().any():raise ValueError('Pasien duplikat dalam loader')
    multi=bags.n_images>1
    summary={'patients':len(bags),'multiview_patients':int(multi.sum()),
             'mean_abs_delta_p':float(predictions.abs_delta_p.mean()),
             'mean_abs_delta_q':float(predictions.abs_delta_q.mean()),
             'three_class_changed':int(predictions.three_class_changed.sum()),
             'diagnosis_changed':int(predictions.diagnosis_changed.sum()),
             'mean_pairwise_cosine_multiview':float(bags.loc[multi,'mean_pairwise_cosine'].mean()),
             'mean_D_entropy_multiview':float(bags.loc[multi,'attention_D_entropy_normalized'].mean()),
             'mean_C_entropy_multiview':float(bags.loc[multi,'attention_C_entropy_normalized'].mean())}
    for tag in ['attention','uniform']:
        metrics=compute_metrics(predictions.y_true_class.to_numpy(),predictions['class_'+tag].to_numpy())
        summary[tag+'_accuracy']=metrics['accuracy'];summary[tag+'_macro_f1']=metrics['f1']
        app=predictions.y_appendicitis.to_numpy()==1
        d=binary_metrics(predictions.y_appendicitis.to_numpy(),predictions['p_'+tag].to_numpy()>=.5)
        c=binary_metrics(predictions.loc[app,'y_complicated'].to_numpy(),predictions.loc[app,'q_'+tag].to_numpy()>=.5)
        summary[tag+'_diagnosis_accuracy']=d['accuracy'];summary[tag+'_conditional_severity_f1']=c['f1']
    bags.to_csv(output_dir/'bag_embedding_summary.csv',index=False)
    instances.to_csv(output_dir/'instance_attention.csv',index=False)
    predictions.to_csv(output_dir/'attention_vs_uniform_predictions.csv',index=False)
    np.savez_compressed(output_dir/'instance_embeddings.npz',features=np.stack(features),
                        patient_ids=np.array(feature_ids),image_ids=np.array(image_ids))
    import json
    (output_dir/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return {'summary':summary,'bags':bags,'instances':instances,'predictions':predictions}
