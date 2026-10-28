"""Independent image MIL and clinical MLP controls for Experiment 1.

Reuse the reference cohort/preprocessing and hyperparameters, never its learned
weights. Existing multimodal models, training and evaluation remain unchanged.
"""
from contextlib import redirect_stdout
from dataclasses import asdict
from pathlib import Path
import html
import json
import time

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from multimodal_mil_model import (
    ModelConfig, OptimizerConfig, MultimodalMILModel, BagAggregator, MLPEncoder,
    build_backbone, configure_backbone_training, set_backbone_mode, _mlp,
    build_optimizer, count_parameters,
)
from training import fit, set_seed, run_epoch
from experiment_runner import (
    Tee, collate_with_ids, evaluate_and_save, file_sha256, fingerprint,
    write_json, release_gpu, save_history_plots,
)
from experiment_evaluation import summarize, bootstrap_ci, GROUPS


class UnimodalHierarchicalModel(nn.Module):
    """Only the selected encoder exists. Task MLPs/head widths match S26."""
    predict_proba = MultimodalMILModel.predict_proba
    _class_probs_from = staticmethod(MultimodalMILModel._class_probs_from)
    predict_classes = MultimodalMILModel.predict_classes
    compute_loss = MultimodalMILModel.compute_loss
    encode_images = MultimodalMILModel.encode_images

    def __init__(self, cfg, modality):
        super().__init__()
        if modality not in ('image', 'clinical') or cfg.head != 'dual_head':
            raise ValueError('Use image/clinical and hierarchical dual_head')
        self.cfg, self.modality = cfg, modality
        if modality == 'image':
            self.backbone, dim = build_backbone(cfg.backbone, cfg.pretrained)
            configure_backbone_training(self.backbone, cfg)
            self.image_proj = nn.Identity() if cfg.image_embed_dim is None else nn.Sequential(
                nn.Linear(dim, cfg.image_embed_dim), nn.ReLU(), nn.Dropout(cfg.image_dropout))
            self.aggregator = BagAggregator(cfg, dim if cfg.image_embed_dim is None else cfg.image_embed_dim)
            input_dim = self.aggregator.out_dim
        else:
            if cfg.tabular_encoder != 'mlp':
                raise ValueError('Clinical control must use MLP')
            self.tabular = MLPEncoder(cfg.num_tabular_features, cfg.tabular_embed_dim,
                                      cfg.mlp_hidden_dims, cfg.mlp_dropout)
            input_dim = cfg.tabular_embed_dim
        # No concatenation with a dummy/zero embedding: only the active input.
        self.fusion_diagnosis, dim = _mlp(input_dim, cfg.fusion_hidden_dims, cfg.fusion_dropout)
        self.fusion_complication, _ = _mlp(input_dim, cfg.fusion_hidden_dims, cfg.fusion_dropout)
        self.head_diagnosis, self.head_complication = nn.Linear(dim, 1), nn.Linear(dim, 1)

    def train(self, mode=True):
        super().train(mode)
        if self.modality == 'image':
            set_backbone_mode(self.backbone, self.cfg, mode)
        return self

    def forward(self, images, image_mask, tabular, return_attention=False):
        if self.modality == 'image':
            MultimodalMILModel._check_inputs(images, image_mask, tabular)
            d, c, ad, ac = self.aggregator(self.encode_images(images, image_mask), image_mask)
        else:
            d, _, _ = self.tabular(tabular)
            c = d
            ad = ac = None
        out = {'logit_diagnosis': self.head_diagnosis(self.fusion_diagnosis(d)).squeeze(-1),
               'logit_complication': self.head_complication(self.fusion_complication(c)).squeeze(-1),
               'tab_sparsity_loss': None}
        if return_attention:
            out.update(attention_diagnosis=ad, attention_complication=ac)
        return out


class ClinicalDataset(Dataset):
    """Keep patient/bag metadata; never decode or augment images."""
    def __init__(self, patient_dataset):
        self.source = patient_dataset

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        s = self.source
        return {'images': torch.zeros(len(s.bags[index]), 3, 1, 1),
                'tabular': s.tabular[index], 'patient_id': s.patient_ids[index],
                'y_appendicitis': int(s.y_app[index]), 'y_complicated': int(s.y_comp[index])}


def independent_epoch(model, loader, device, optimizer=None, scaler=None, use_amp=False):
    """AMP only for training; checkpoint selection and final evaluation use FP32."""
    return run_epoch(model, loader, device, optimizer, scaler,
                     use_amp=bool(use_amp and optimizer is not None))


def load_reference_data(root, reference_dir):
    """Read locked patient order and saved transformers, without any fit/split."""
    from mil_diagnostics import transform_saved_tabular
    from image_preprocessing import ImagePreprocessConfig, ImageTransform, PatientBagDataset, load_stats
    root, reference = Path(root), Path(reference_dir)
    config = json.loads((reference / 'config.json').read_text(encoding='utf-8'))
    artifacts = json.loads((reference / 'shared_artifacts.json').read_text(encoding='utf-8'))
    for name, record in artifacts.items():
        if isinstance(record, dict) and 'sha256' in record:
            if file_sha256(reference / name) != record['sha256']:
                raise ValueError(f'Reference artifact changed: {name}')
    saved = joblib.load(reference / 'tabular_preprocessing.joblib')
    protocol = json.loads((reference / 'split_protocol.json').read_text(encoding='utf-8'))
    if file_sha256(reference / 'cohort.csv') != protocol['cohort_sha256']:
        raise ValueError('Reference cohort changed')
    split = pd.read_csv(reference / 'split_manifest.csv')
    cohort = pd.read_csv(reference / 'cohort.csv').set_index('US_Number', drop=False)
    if split.patient_id.duplicated().any() or set(split.patient_id) != set(cohort.index):
        raise ValueError('Invalid reference patient manifest')
    if not np.array_equal(cohort.loc[split.patient_id, 'Label'].to_numpy(), split.label.to_numpy()):
        raise ValueError('Patient labels differ from the locked manifest')
    # Hash source images too: a reused path alone cannot prove identical data.
    image_root = (root / 'US_Pictures_clean').resolve()
    integrity = pd.read_csv(reference / 'image_integrity.csv')
    for row in integrity.itertuples():
        path = (image_root / row.path).resolve()
        if not path.is_relative_to(image_root) or file_sha256(path) != row.sha256:
            raise ValueError(f'Reference image changed or invalid path: {row.path}')
    train_ids = split.loc[split.split.eq('train')].sort_values('order').patient_id.tolist()
    if train_ids != saved['train_patient_ids']:
        raise ValueError('Saved preprocessing was fitted on different training patients')
    cfg = ImagePreprocessConfig(**config['preprocessing']['image'])
    stats = load_stats(reference / 'image_stats.json')
    datasets = {}
    for name in ('train', 'val', 'test'):
        ids = split.loc[split.split.eq(name)].sort_values('order').patient_id.tolist()
        frame = cohort.loc[ids].copy()
        _, features = transform_saved_tabular(frame, saved)
        labels = pd.DataFrame({'y_appendicitis': frame.Label.ne('no appendicitis').astype(int),
            'y_complicated': frame.Label.map({'uncomplicated_appendicitis': 0,
                                               'complicated_appendicitis': 1}).fillna(-1).astype(int)}, index=frame.index)
        manifest = pd.read_csv(reference / f'{name}_image_manifest.csv')
        if set(manifest.US_Number) != set(ids):
            raise ValueError('Image manifest has different patients')
        expected_images = integrity[integrity.split.eq(name)]
        actual_pairs = sorted(zip(manifest.US_Number, manifest.path))
        expected_pairs = sorted(zip(expected_images.US_Number, expected_images.path))
        if actual_pairs != expected_pairs:
            raise ValueError('Image bags differ from the locked reference')
        for key, training in [(name, name == 'train')] + ([('train_eval', False)] if name == 'train' else []):
            datasets[key] = PatientBagDataset(features, labels, ids, manifest,
                root / 'US_Pictures_clean', ImageTransform(cfg, stats, train=training))
    return datasets, saved['feature_cols']


def train_independent(directory, modality, model_config, optimizer_config, datasets, settings, provenance):
    """New model/optimizer/scaler/RNG per call; matching completed runs are read-only."""
    directory = Path(directory)
    spec = {'modality': modality, 'model': model_config.to_dict(),
            'optimizer': asdict(optimizer_config), 'training': settings, 'provenance': provenance}
    signature = fingerprint(spec)
    if directory.exists():
        if not (directory / 'result.json').exists():
            raise ValueError('Incomplete prior trial: use a new run directory; no automatic continuation')
        old = json.loads((directory / 'config.json').read_text())
        if old['fingerprint'] != signature:
            raise ValueError('Configuration changed: use a new run directory')
        result = json.loads((directory / 'result.json').read_text())
        if file_sha256(directory / 'best_model.pt') != result['checkpoint_sha256']:
            raise ValueError('Checkpoint changed')
        return result
    directory.mkdir(parents=True)
    write_json(directory / 'config.json', {**spec, 'fingerprint': signature})
    write_json(directory / 'status.json', {'status': 'training'})
    model = optimizer = None
    start = time.perf_counter()
    device = torch.device(settings['device'])
    try:
        set_seed(settings['seed'])
        model = UnimodalHierarchicalModel(model_config, modality).to(device)
        optimizer = build_optimizer(model, optimizer_config)
        write_json(directory / 'parameters.json', count_parameters(model))
        # A different architecture consumes a different amount of RNG at construction.
        # Reset again so sampling/augmentation begin independently from the same seed.
        set_seed(settings['seed'])
        data = {k: ClinicalDataset(v) if modality == 'clinical' else v for k, v in datasets.items()}
        common = dict(batch_size=settings['batch_size'], num_workers=settings['num_workers'], collate_fn=collate_with_ids)
        train_loader = DataLoader(data['train'], shuffle=True, drop_last=True,
                                 generator=torch.Generator().manual_seed(settings['seed']), **common)
        val_loader = DataLoader(data['val'], shuffle=False, **common)
        if not len(train_loader):
            raise ValueError('Empty training loader')
        checkpoint = directory / 'best_model.pt'
        with (directory / 'training.log').open('w', encoding='utf-8') as log, redirect_stdout(Tee(__import__('sys').stdout, log)):
            history = fit(model, train_loader, val_loader, optimizer, device,
                epochs=settings['epochs'], monitor=settings['monitor'], checkpoint_path=checkpoint,
                restore_best=True, use_amp=settings['use_amp'], verbose=settings.get('verbose', True),
                early_stopping_patience=settings['patience'], early_stopping_min_delta=settings['min_delta'],
                early_stopping_rule=settings['early_stopping_rule'], checkpoint_tie_break='val_loss',
                epoch_runner=independent_epoch)
        history.to_csv(directory / 'history.csv')
        write_json(directory / 'history_metadata.json', history.attrs)
        if settings.get('save_plots', True):
            save_history_plots(history, directory)
        metrics = {}
        for split, dataset in [('train', data['train_eval']), ('val', data['val'])]:
            loader = DataLoader(dataset, shuffle=False, **common)
            metrics[split], _ = evaluate_and_save(model, loader, device, directory, split, checkpoint, settings['threshold'])
        if not np.isclose(metrics['val']['overall']['f1'], history.attrs['best_value'], atol=1e-6, rtol=0):
            raise ValueError(f'Restored checkpoint macro-F1={metrics["val"]["overall"]["f1"]:.8f} '
                             f'differs from selected value={history.attrs["best_value"]:.8f}. '
                             'Validation during training and restored evaluation must both use FP32.')
        result = {'modality': modality, 'best_epoch': history.attrs['best_epoch'],
                  'checkpoint_sha256': file_sha256(checkpoint), 'seconds_total': time.perf_counter()-start,
                  'train': metrics['train'], 'val': metrics['val'], 'status': 'completed'}
        write_json(directory / 'result.json', result)
        write_json(directory / 'status.json', {'status': 'completed'})
        return result
    except BaseException as error:
        import traceback
        (directory / 'error.log').write_text(traceback.format_exc(), encoding='utf-8')
        write_json(directory / 'status.json', {'status': 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', 'error': str(error)})
        raise
    finally:
        model = optimizer = None
        release_gpu()


def test_independent(directory, test_dataset, bootstrap_resamples=1000):
    """Evaluate the val-selected unimodal checkpoint, including patient subgroups."""
    directory = Path(directory)
    spec = json.loads((directory / 'config.json').read_text())
    result = json.loads((directory / 'result.json').read_text())
    checkpoint = directory / 'best_model.pt'
    if result['status'] != 'completed' or file_sha256(checkpoint) != result['checkpoint_sha256']:
        raise ValueError('Incomplete or altered checkpoint')
    settings, modality = spec['training'], spec['modality']
    model = None
    try:
        # Reconstruct offline; trained weights are loaded only for evaluation.
        model = UnimodalHierarchicalModel(ModelConfig(**{**spec['model'], 'pretrained': False}), modality)
        model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True)['model_state'])
        device = torch.device(settings['device'])
        model.to(device).eval()
        dataset = ClinicalDataset(test_dataset) if modality == 'clinical' else test_dataset
        loader = DataLoader(dataset, batch_size=settings['batch_size'], shuffle=False,
                            num_workers=settings['num_workers'], collate_fn=collate_with_ids)
        output = directory / 'testing'
        output.mkdir(exist_ok=True)
        _, frame = evaluate_and_save(model, loader, device, output, 'test', checkpoint, settings['threshold'])
        overview, classes, binary, intervals = [], [], [], []
        report = {'modality': modality, 'training': 'independently_trained_unimodal',
                  'checkpoint_sha256': result['checkpoint_sha256'], 'groups': {}}
        for group in GROUPS:
            subset = frame if group == 'all' else frame[frame.n_images.eq(1) if group == 'single_image' else frame.n_images.gt(1)]
            metrics = summarize(subset, settings['threshold'])
            metrics['bootstrap_95ci'] = bootstrap_ci(subset, bootstrap_resamples, settings['seed'])
            report['groups'][group] = metrics
            write_json(output / f'{modality}_{group}_metrics.json', metrics)
            overview.append({'mode': modality, 'group': group, **metrics['overall']})
            classes.extend({'mode': modality, 'group': group, **r} for r in metrics['per_class'])
            for task in ('diagnosis', 'severity_direct_true_appendicitis', 'severity_after_gating_true_appendicitis', 'complicated_end_to_end_all_patients'):
                binary.append({'mode': modality, 'group': group, 'task': task, **metrics[task]})
            intervals.extend({'mode': modality, 'group': group, 'metric': key, **value}
                             for key, value in metrics['bootstrap_95ci']['metrics'].items())
        tables = {'overview': pd.DataFrame(overview), 'per_class': pd.DataFrame(classes),
                  'binary_tasks': pd.DataFrame(binary), 'bootstrap_95ci': pd.DataFrame(intervals)}
        fragments = ['<h1>Independent ' + html.escape(modality) + ' model</h1>',
                     '<p>Patient-level evaluation of a separately trained model. Class accuracy = recall; OvR accuracy includes TN. Single/multiple groups use the original image count, including for the clinical model.</p>']
        for name, table in tables.items():
            table.to_csv(output / f'{name}.csv', index=False)
            fragments += [f'<h2>{name}</h2>', table.to_html(index=False)]
        _save_test_plots(frame, report, output)
        fragments.append('<img src="test_curves.png" style="max-width:100%">')
        (output / 'report.html').write_text('<!doctype html><meta charset="utf-8">' + ''.join(fragments), encoding='utf-8')
        write_json(output / 'report.json', report)
        return report
    finally:
        model = None
        release_gpu()


def _save_test_plots(frame, report, directory):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from sklearn.metrics import roc_curve, precision_recall_curve
    fig = Figure(figsize=(13, 8)); FigureCanvasAgg(fig)
    axes = fig.subplots(2, 3)
    for i, group in enumerate(GROUPS):
        cm = report['groups'][group]['confusion_matrix']
        axes[0, i].imshow(cm, cmap='Blues')
        axes[0, i].set(title=group, xlabel='Predicted', ylabel='True', xticks=range(3), yticks=range(3))
        for a in range(3):
            for b in range(3): axes[0, i].text(b, a, str(cm[a][b]), ha='center', va='center')
    for label, data, target, score in [
        ('diagnosis', frame, 'y_appendicitis', 'p_diagnosis'),
        ('severity', frame[frame.y_appendicitis.eq(1)], 'y_complicated', 'q_complication'),
        ('complicated', frame.assign(complicated=frame.y_true.eq(2)), 'complicated', 'prob_complicated')]:
        if data[target].nunique() != 2: continue
        x, y, _ = roc_curve(data[target], data[score]); axes[1, 0].plot(x, y, label=label)
        p, r, _ = precision_recall_curve(data[target], data[score]); axes[1, 1].plot(r, p, label=label)
    bins = report['groups']['all']['calibration_top_label']
    axes[1, 2].plot([0, 1], [0, 1], '--', color='gray')
    if bins: axes[1, 2].plot([b['confidence'] for b in bins], [b['accuracy'] for b in bins], 'o-')
    for ax, title, x, y in [(axes[1, 0], 'ROC', 'FPR', 'TPR'), (axes[1, 1], 'Precision-recall', 'Recall', 'Precision'),
                           (axes[1, 2], 'Calibration (argmax)', 'Confidence', 'Accuracy')]:
        ax.set(title=title, xlabel=x, ylabel=y)
        if ax.get_legend_handles_labels()[0]: ax.legend()
    fig.tight_layout(); fig.savefig(directory / 'test_curves.png', dpi=140)
