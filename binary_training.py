"""Single-task binary MIL experiments using the same encoders and fit/early stopping.

Diagnosis: all patients, positive=appendicitis. Severity: true appendicitis patients
only, positive=complicated. One active attention, fusion and head; no gating by
the other task. Validation and reporting use FP32 consistently.
"""
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import confusion_matrix, roc_auc_score

from multimodal_mil_model import ModelConfig, MultimodalMILModel, collate_bags, set_backbone_mode
from training import binary_metrics, fit


CLASS_NAMES = {
    'diagnosis': ('no appendicitis', 'appendicitis'),
    'severity': ('uncomplicated_appendicitis', 'complicated_appendicitis'),
}


class BinaryMILModel(nn.Module):
    def __init__(self, cfg: ModelConfig, task: str):
        super().__init__()
        if task not in CLASS_NAMES:
            raise ValueError('task harus diagnosis atau severity')
        if cfg.head != 'dual_head':
            raise ValueError('BinaryMILModel menggunakan komponen dari konfigurasi dual_head')
        self.cfg, self.binary_task = cfg, task
        # Construct identically to the parent experiment before keeping one branch.
        base = MultimodalMILModel(cfg)
        self.backbone, self.image_proj, self.tabular = base.backbone, base.image_proj, base.tabular
        if task == 'diagnosis':
            self.pool = base.aggregator.first
            self.fusion, self.head = base.fusion_diagnosis, base.head_diagnosis
        else:
            self.pool = base.aggregator.second if base.aggregator.second is not None else base.aggregator.first
            self.fusion, self.head = base.fusion_complication, base.head_complication

    def train(self, mode=True):
        super().train(mode)
        set_backbone_mode(self.backbone, self.cfg, mode)
        return self

    def forward(self, images, image_mask, tabular, return_attention=False):
        MultimodalMILModel._check_inputs(images, image_mask, tabular)
        feats = self.image_proj(self.backbone(images[image_mask]))
        h = feats.new_zeros(*image_mask.shape, feats.size(-1))
        h[image_mask] = feats
        z, attention = self.pool(h, image_mask)
        e, sparsity, masks = self.tabular(tabular, return_masks=return_attention)
        out = {'logit':self.head(self.fusion(torch.cat([z,e],1))).squeeze(-1),
               'tab_sparsity_loss':sparsity}
        if return_attention:out.update(attention=attention, tabular_masks=masks)
        return out

    def targets(self, batch):
        target = batch['y_appendicitis'] if self.binary_task == 'diagnosis' else batch['y_complicated']
        if self.binary_task == 'severity' and not (batch['y_appendicitis'] == 1).all():
            raise ValueError('Severity hanya menerima pasien true appendicitis')
        if not ((target == 0) | (target == 1)).all():
            raise ValueError('Target binary harus 0/1')
        return target

    def compute_loss(self, out, target):
        bce = F.binary_cross_entropy_with_logits(out['logit'].float(), target.float())
        total = bce
        parts = {'loss_bce':bce.item()}
        sparse = out.get('tab_sparsity_loss')
        if sparse is not None and self.cfg.tabnet_sparsity_weight > 0:
            total = total + self.cfg.tabnet_sparsity_weight * sparse
            parts['loss_tab_sparsity'] = sparse.item()
        parts['loss_total'] = total.item()
        return total, parts


def collate_binary_bags(items):
    batch = collate_bags(items)
    batch['patient_id'] = torch.tensor([int(item['patient_id']) for item in items])
    return batch


def _auc(y, probability):
    return float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else float('nan')


def run_binary_epoch(model, loader, device, optimizer=None, scaler=None, use_amp=False):
    training = optimizer is not None
    model.train(training)
    sums, ys, ps, count = {}, [], [], 0
    with torch.set_grad_enabled(training):
        for batch in loader:
            batch = {k:v.to(device, non_blocking=True) for k,v in batch.items()}
            target = model.targets(batch)
            # AMP for training only; validation/checkpoint/report all use FP32.
            with torch.autocast(device_type=device.type, enabled=use_amp and training):
                out = model(batch['images'],batch['image_mask'],batch['tabular'])
                loss, parts = model.compute_loss(out,target)
            if not torch.isfinite(loss):raise FloatingPointError('Binary loss tidak finite')
            if training:
                optimizer.zero_grad(set_to_none=True)
                if scaler is None:
                    loss.backward();optimizer.step()
                else:
                    scaler.scale(loss).backward();scaler.step(optimizer);scaler.update()
            bs = len(target);count += bs
            for k,v in parts.items():sums[k] = sums.get(k,0.) + v*bs
            ys.extend(target.detach().cpu().tolist())
            ps.extend(out['logit'].detach().float().sigmoid().cpu().tolist())
    if not count:raise ValueError('DataLoader binary kosong')
    summary = {('loss' if k == 'loss_total' else k):v/count for k,v in sums.items()}
    y,probability = np.array(ys),np.array(ps)
    summary['auroc'] = _auc(y,probability)
    return summary,y,(probability >= .5).astype(int)


def fit_binary(model, train_loader, val_loader, optimizer, device, epochs, **kwargs):
    history = fit(model,train_loader,val_loader,optimizer,device,epochs,
                  epoch_runner=run_binary_epoch,metric_fn=binary_metrics,**kwargs)
    history.attrs.update(binary_task=model.binary_task, threshold=.5,
                         positive_class=CLASS_NAMES[model.binary_task][1],validation_precision='fp32')
    checkpoint_path = kwargs.get('checkpoint_path','best_model.pt')
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path,map_location='cpu',weights_only=True)
        checkpoint.update(binary_task=model.binary_task,class_names=CLASS_NAMES[model.binary_task],threshold=.5)
        torch.save(checkpoint,checkpoint_path)
    return history


@torch.no_grad()
def predict_binary_loader(model, loader, device):
    device = torch.device(device)
    model.to(device).eval()
    records = []
    for batch in loader:
        batch = {k:v.to(device) for k,v in batch.items()}
        target = model.targets(batch)
        out = model(batch['images'],batch['image_mask'],batch['tabular'])
        p = out['logit'].float().sigmoid().cpu().numpy()
        records.extend({'patient_id':int(pid),'y_true':int(y),'probability_positive':float(prob),
                        'y_pred':int(prob >= .5)}
                       for pid,y,prob in zip(batch['patient_id'].cpu(),target.cpu(),p))
    if not records:raise ValueError('DataLoader binary kosong')
    return pd.DataFrame(records)


def detailed_binary_report(model, loader, device):
    predictions = predict_binary_loader(model,loader,device)
    y,p = predictions.y_true.to_numpy(),predictions.y_pred.to_numpy()
    metrics = binary_metrics(y,p)
    metrics['auroc'] = _auc(y,predictions.probability_positive.to_numpy())
    names = CLASS_NAMES[model.binary_task]
    cm = pd.DataFrame(confusion_matrix(y,p,labels=[0,1]),
                      index=[f'asli: {c}' for c in names],columns=[f'prediksi: {c}' for c in names])
    print(f'Task: {model.binary_task}; kelas positif: {names[1]}; threshold=0.5; FP32')
    print(cm.to_string())
    print({k:round(v,4) for k,v in metrics.items()})
    return {'metrics':metrics,'confusion_matrix':cm,'predictions':predictions}
