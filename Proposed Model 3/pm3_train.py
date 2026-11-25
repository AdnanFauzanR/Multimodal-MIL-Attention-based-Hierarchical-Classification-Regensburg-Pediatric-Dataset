"""Training, balanced hierarchical loss, 5-fold CV, and nested threshold calibration for Proposed Model 3."""
from __future__ import annotations

import gc
import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.nn import functional as F
from torch.utils.data import DataLoader

from pm3_data import (
    BAG_MODES,
    CLASS_NAMES,
    CONCEPTS_PM3,
    FEATURES,
    ID_COL,
    ClinicalPipeline,
    PM3PatientBagDataset,
    collate_patients,
    encode_pm3_concepts,
)
from pm3_model import CLASS_IDS, ModelConfig, PM3Model


@dataclass
class TrainConfig:
    epochs: int = 60
    batch_size: int = 16
    amp_fp16: bool = True
    num_workers: int = 0
    seed: int = 1
    bootstrap_samples: int = 2000
    patience: int = 15
    warmup_freeze_epochs: int = 5
    scheduler_t_max: int = 40
    scheduler_eta_min: float = 1e-6
    backbone_lr: float = 2e-5
    attention_lr: float = 2e-4
    head_lr: float = 4e-4
    comp_weight_mode: str = "sqrt"
    label_smoothing: float = 0.03
    unrecorded_smoothing: float = 0.05
    bag_mode: str = "all"
    severity_scope: str = "all"
    w_diag: float = 1.0
    w_comp: float = 1.0
    w_hier_nll: float = 0.5
    w_concept: float = 0.35
    w_aux_img: float = 0.5

    def __post_init__(self):
        if self.epochs < 1 or self.patience < 1 or self.batch_size < 1:
            raise ValueError("epochs, patience, dan batch_size harus positif.")
        if self.warmup_freeze_epochs < 0:
            raise ValueError("warmup_freeze_epochs tidak boleh negatif.")
        if self.comp_weight_mode not in ("sqrt", "full"):
            raise ValueError("comp_weight_mode harus 'sqrt' atau 'full'.")
        if self.bag_mode not in BAG_MODES:
            raise ValueError(f"bag_mode harus dalam {BAG_MODES}.")
        if self.severity_scope not in ("all", "appendicitis_only"):
            raise ValueError("severity_scope harus 'all' atau 'appendicitis_only'.")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_json(path: str | Path, value):
    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, np.ndarray):
            return clean(x.tolist())
        if isinstance(x, (float, np.floating)):
            return float(x) if np.isfinite(x) else None
        if isinstance(x, (int, np.integer)):
            return int(x)
        return x

    Path(path).write_text(json.dumps(clean(value), indent=2, allow_nan=False), encoding="utf-8")


def training_weights(
    labels: np.ndarray,
    concepts_binary: np.ndarray,
    concept_mask: np.ndarray,
    comp_weight_mode: str = "sqrt",
    severity_scope: str = "all",
) -> dict:
    """Compute balanced training weights from train fold only."""
    labels = np.asarray(labels, dtype=int)
    counts = np.bincount(labels, minlength=3)
    if (counts == 0).any():
        raise ValueError("Semua 3 kelas harus tersedia pada train fold.")
    n_total = float(len(labels))
    n_neg = float(counts[0])
    n_pos = float(counts[1] + counts[2])

    # Equal total gradient mass between No-App and App while preserving unit mean weight
    diag_sample_w = [n_total / (2.0 * n_neg), n_total / (2.0 * n_pos)]

    ratio_comp = float(counts[1] / counts[2])
    comp_pos_w = math_sqrt(ratio_comp) if comp_weight_mode == "sqrt" else ratio_comp

    # 3-class sqrt-balanced weights normalized to mean 1.0 for coupled NLL / Softmax
    raw_w3 = np.sqrt(n_total / (3.0 * counts.astype(float)))
    w3 = (raw_w3 / raw_w3.mean()).tolist()

    # Binary severity (all vs appendicitis_only)
    if severity_scope == "all":
        sev_neg = float(counts[0] + counts[1])
        sev_pos = float(counts[2])
    else:
        sev_neg = float(counts[1])
        sev_pos = float(counts[2])
    sev_pos_w = float(np.clip(sev_neg / max(sev_pos, 1.0), 1.0, 6.0))
    if comp_weight_mode == "sqrt":
        sev_pos_w = float(np.sqrt(sev_pos_w))

    # Concept pos_weight clipped to [0.5, 4.0] to avoid instability on rare concepts
    pos = (concepts_binary * concept_mask).sum(0)
    neg = ((1.0 - concepts_binary) * concept_mask).sum(0)
    c_w = np.divide(neg, pos, out=np.ones_like(pos), where=(pos > 0) & (neg > 0))
    c_w = np.clip(c_w, 0.5, 4.0)

    return {
        "class_counts": counts.tolist(),
        "diag_sample_weight": diag_sample_w,
        "comp_pos_weight": float(comp_pos_w),
        "class_weight_3": w3,
        "diag_binary_pos_weight": float(n_neg / n_pos),
        "sev_binary_pos_weight": float(sev_pos_w),
        "concept_pos_weight": c_w.tolist(),
    }


def math_sqrt(x: float) -> float:
    return float(np.sqrt(x))


def _binary_target_for_item(label: int, task_mode: str) -> float:
    if task_mode == "binary_diag":
        return float(label > 0)
    return float(label == 2)


def loss_terms(out: dict, item: dict, weights: dict, model_cfg: ModelConfig, train_cfg: TrainConfig) -> dict:
    """Compute per-patient numerator loss terms for any task_mode / modality."""
    label = int(item["label"])
    device = (
        out["logit_diag"].device
        if "logit_diag" in out
        else out["logits"].device
        if "logits" in out
        else out["logit"].device
    )
    zero = torch.tensor(0.0, device=device, dtype=torch.float32)

    terms = {
        "diag": zero.clone(),
        "comp": zero.clone(),
        "hier_nll": zero.clone(),
        "softmax": zero.clone(),
        "binary": zero.clone(),
        "concept": zero.clone(),
        "aux_img": zero.clone(),
        "is_app": 1.0 if label > 0 else 0.0,
    }

    if model_cfg.task_mode == "hierarchical":
        y_diag = torch.tensor(float(label > 0), device=device)
        sw_diag = weights["diag_sample_weight"][1 if label > 0 else 0]
        logit_d = out["logit_diag"].float()
        terms["diag"] = sw_diag * F.binary_cross_entropy_with_logits(logit_d, y_diag, reduction="none")

        logit_c = out["logit_comp"].float()
        if label > 0:
            y_comp = torch.tensor(float(label == 2), device=device)
            pw_c = torch.tensor([weights["comp_pos_weight"]], device=device, dtype=torch.float32)
            terms["comp"] = F.binary_cross_entropy_with_logits(
                logit_c.view(-1), y_comp.view(-1), pos_weight=pw_c, reduction="sum"
            )

        # Coupled 3-class Hierarchical NLL
        log_p0 = F.logsigmoid(-logit_d)
        log_p1 = F.logsigmoid(logit_d) + F.logsigmoid(-logit_c)
        log_p2 = F.logsigmoid(logit_d) + F.logsigmoid(logit_c)
        log_probs = torch.stack((log_p0, log_p1, log_p2))
        eps = train_cfg.label_smoothing
        nll = -(1.0 - eps) * log_probs[label] - eps * log_probs.mean()
        terms["hier_nll"] = weights["class_weight_3"][label] * nll

        if model_cfg.modality == "multimodal" and "aux_img_logit_diag" in out:
            aux_d = sw_diag * F.binary_cross_entropy_with_logits(
                out["aux_img_logit_diag"].float(), y_diag, reduction="none"
            )
            aux_c = zero
            if label > 0:
                y_comp = torch.tensor(float(label == 2), device=device)
                pw_c = torch.tensor([weights["comp_pos_weight"]], device=device, dtype=torch.float32)
                aux_c = F.binary_cross_entropy_with_logits(
                    out["aux_img_logit_comp"].float().view(-1), y_comp.view(-1), pos_weight=pw_c, reduction="sum"
                )
            terms["aux_img"] = (aux_d, aux_c)
    elif model_cfg.task_mode == "softmax":
        target = torch.tensor([label], dtype=torch.long, device=device)
        cw = torch.tensor(weights["class_weight_3"], dtype=torch.float32, device=device)
        terms["softmax"] = F.cross_entropy(
            out["logits"].float().unsqueeze(0),
            target,
            weight=cw,
            label_smoothing=train_cfg.label_smoothing,
            reduction="none",
        )[0]
        if model_cfg.modality == "multimodal" and "aux_img_logits" in out:
            terms["aux_img"] = F.cross_entropy(
                out["aux_img_logits"].float().unsqueeze(0),
                target,
                weight=cw,
                label_smoothing=train_cfg.label_smoothing,
                reduction="none",
            )[0]
    else:
        y_bin = torch.tensor(_binary_target_for_item(label, model_cfg.task_mode), device=device)
        pw_val = (
            weights["diag_binary_pos_weight"]
            if model_cfg.task_mode == "binary_diag"
            else weights["sev_binary_pos_weight"]
        )
        pw = torch.tensor([pw_val], device=device, dtype=torch.float32)
        terms["binary"] = F.binary_cross_entropy_with_logits(
            out["logit"].float().view(-1), y_bin.view(-1), pos_weight=pw, reduction="sum"
        )
        if model_cfg.modality == "multimodal" and "aux_img_logit" in out:
            terms["aux_img"] = F.binary_cross_entropy_with_logits(
                out["aux_img_logit"].float().view(-1), y_bin.view(-1), pos_weight=pw, reduction="sum"
            )

    if model_cfg.modality in ("multimodal", "image_only") and model_cfg.use_concepts and "concept_logits" in out:
        tgt_c = item["concepts_target"].to(device, dtype=torch.float32)
        mask_c = item["concept_mask"].to(device, dtype=torch.float32)
        pw_concepts = torch.tensor(weights["concept_pos_weight"], device=device, dtype=torch.float32)
        terms["concept"] = (
            F.binary_cross_entropy_with_logits(
                out["concept_logits"].float(), tgt_c, pos_weight=pw_concepts, reduction="none"
            )
            * mask_c
        ).sum()

    return terms


def combine_batch_loss(
    terms: dict,
    n_batch: int,
    n_app: int,
    n_concepts: float,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
) -> torch.Tensor:
    """Normalize per-patient numerators using batch denominators (1:1 diag vs comp balance)."""
    n_app_safe = max(n_app, 1)
    n_con_safe = max(n_concepts, 1.0)
    loss = terms["concept"] * (train_cfg.w_concept / n_con_safe)

    if model_cfg.task_mode == "hierarchical":
        loss = (
            loss
            + train_cfg.w_diag * (terms["diag"] / n_batch)
            + train_cfg.w_comp * (terms["comp"] / n_app_safe)
            + train_cfg.w_hier_nll * (terms["hier_nll"] / n_batch)
        )
        if isinstance(terms["aux_img"], tuple):
            aux_d, aux_c = terms["aux_img"]
            loss = loss + train_cfg.w_aux_img * (0.5 * aux_d / n_batch + 0.5 * aux_c / n_app_safe)
    elif model_cfg.task_mode == "softmax":
        loss = loss + (terms["softmax"] / n_batch) + train_cfg.w_aux_img * (terms["aux_img"] / n_batch)
    else:
        loss = loss + (terms["binary"] / n_batch) + train_cfg.w_aux_img * (terms["aux_img"] / n_batch)
    return loss


def forward_patient(model: PM3Model, item: dict, device: torch.device, amp: bool) -> dict:
    imgs = item["images"].to(device) if item["images"] is not None else None
    masks = item["pixel_masks"].to(device) if item["pixel_masks"] is not None else None
    tab = item["tabular"].to(device) if item["tabular"] is not None else None
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
        return model(imgs, masks, tab)


def clip_gradients(model: PM3Model):
    if hasattr(model, "clinical_mlp"):
        clin = list(model.clinical_mlp.parameters())
        if hasattr(model, "clinical_skip_a") and model.clinical_skip_a is not None:
            clin += list(model.clinical_skip_a.parameters())
        if hasattr(model, "clinical_skip_b") and model.clinical_skip_b is not None:
            clin += list(model.clinical_skip_b.parameters())
        clin = [p for p in clin if p.requires_grad and p.grad is not None]
        if clin:
            torch.nn.utils.clip_grad_norm_(clin, 0.5, error_if_nonfinite=True)

    backbone = (
        [p for p in model.encoder.parameters() if p.requires_grad and p.grad is not None]
        if hasattr(model, "encoder")
        else []
    )
    if backbone:
        torch.nn.utils.clip_grad_norm_(backbone, 1.0, error_if_nonfinite=True)

    other = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
    if other:
        torch.nn.utils.clip_grad_norm_(other, 1.5, error_if_nonfinite=True)


def compute_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    predictions: np.ndarray,
    task_mode: str,
) -> dict:
    labels = np.asarray(labels, dtype=int)
    probs = np.asarray(probs, dtype=float)
    pred = np.asarray(predictions, dtype=int)

    if task_mode in ("binary_diag", "binary_sev"):
        pos_prob = probs[:, 1] if probs.ndim == 2 else probs
        cm = confusion_matrix(labels, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        sens = float(tp / max(tp + fn, 1))
        spec = float(tn / max(tn + fp, 1))
        prec = float(tp / max(tp + fp, 1))
        f1_pos = float(f1_score(labels, pred, pos_label=1, zero_division=0))
        macro_f1 = float(f1_score(labels, pred, labels=[0, 1], average="macro", zero_division=0))
        auc = float(roc_auc_score(labels, pos_prob)) if len(np.unique(labels)) == 2 else float("nan")
        aupr = float(average_precision_score(labels, pos_prob)) if len(np.unique(labels)) == 2 else float("nan")
        return {
            "accuracy": float((labels == pred).mean()),
            "precision": prec,
            "sensitivity": sens,
            "specificity": spec,
            "f1": f1_pos,
            "macro_f1": macro_f1,
            "auroc": auc,
            "aupr": aupr,
            "primary_auroc": auc,
            "confusion_matrix": cm.tolist(),
        }

    # 3-class metrics
    auc = (
        float(roc_auc_score(labels, probs, labels=CLASS_IDS, multi_class="ovr", average="macro"))
        if len(np.unique(labels)) == 3
        else float("nan")
    )
    cm = confusion_matrix(labels, pred, labels=CLASS_IDS)
    tp = np.diag(cm).astype(float)
    fn, fp = cm.sum(1) - tp, cm.sum(0) - tp
    tn = cm.sum() - tp - fn - fp

    def ratio(a, b):
        return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b > 0)

    precision, recalls = ratio(tp, tp + fp), ratio(tp, tp + fn)
    specificity, f1 = ratio(tn, tn + fp), ratio(2 * tp, 2 * tp + fp + fn)
    aupr = [
        float(average_precision_score((labels == k).astype(int), probs[:, k]))
        if len(np.unique(labels == k)) == 2
        else float("nan")
        for k in CLASS_IDS
    ]
    metrics = {
        "accuracy": float((labels == pred).mean()),
        "macro_precision": float(precision.mean()),
        "macro_sensitivity": float(recalls.mean()),
        "macro_specificity": float(specificity.mean()),
        "macro_f1": float(f1.mean()),
        "complication_recall": float(recalls[2]),
        "multiclass_auroc": auc,
        "primary_auroc": auc,
        "macro_aupr": float(np.nanmean(aupr)),
        "recall_per_class": recalls.tolist(),
        "confusion_matrix": cm.tolist(),
    }
    for k in range(3):
        binary = (labels == k).astype(int)
        class_auc = float(roc_auc_score(binary, probs[:, k])) if len(np.unique(binary)) == 2 else float("nan")
        metrics.update(
            {
                f"precision_class_{k}": float(precision[k]),
                f"sensitivity_class_{k}": float(recalls[k]),
                f"specificity_class_{k}": float(specificity[k]),
                f"f1_class_{k}": float(f1[k]),
                f"auroc_class_{k}": class_auc,
                f"aupr_class_{k}": aupr[k],
            }
        )
    return metrics


def summarize_loss_dict(
    sums: dict,
    n_patients: int,
    n_app: int,
    n_concepts: float,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
) -> dict:
    """Compute epoch-level raw and lambda-weighted loss components consistently."""
    n_p = max(n_patients, 1)
    n_a = max(n_app, 1)
    n_c = max(n_concepts, 1.0)

    raw_concept = sums["concept"] / n_c
    w_concept = train_cfg.w_concept * raw_concept

    if model_cfg.task_mode == "hierarchical":
        raw_diag = sums["diag"] / n_p
        raw_comp = sums["comp"] / n_a
        raw_hier_nll = sums["hier_nll"] / n_p
        raw_aux_img = 0.5 * (sums["aux_img_d"] / n_p) + 0.5 * (sums["aux_img_c"] / n_a)
        raw = {
            "diag": raw_diag,
            "comp": raw_comp,
            "hier_nll": raw_hier_nll,
            "concept": raw_concept,
            "aux_img": raw_aux_img,
        }
        weighted = {
            "diag": train_cfg.w_diag * raw_diag,
            "comp": train_cfg.w_comp * raw_comp,
            "hier_nll": train_cfg.w_hier_nll * raw_hier_nll,
            "concept": w_concept,
            "aux_img": train_cfg.w_aux_img * raw_aux_img if model_cfg.modality == "multimodal" else 0.0,
        }
    elif model_cfg.task_mode == "softmax":
        raw_sm = sums["softmax"] / n_p
        raw_aux_img = sums["aux_img"] / n_p
        raw = {"softmax": raw_sm, "concept": raw_concept, "aux_img": raw_aux_img}
        weighted = {
            "softmax": raw_sm,
            "concept": w_concept,
            "aux_img": train_cfg.w_aux_img * raw_aux_img if model_cfg.modality == "multimodal" else 0.0,
        }
    else:
        raw_bin = sums["binary"] / n_p
        raw_aux_img = sums["aux_img"] / n_p
        raw = {"binary": raw_bin, "concept": raw_concept, "aux_img": raw_aux_img}
        weighted = {
            "binary": raw_bin,
            "concept": w_concept,
            "aux_img": train_cfg.w_aux_img * raw_aux_img if model_cfg.modality == "multimodal" else 0.0,
        }

    out = {"loss_total": float(sums["total"] / n_p)}
    for k, v in raw.items():
        out[f"loss_{k}"] = float(v)
    for k, v in weighted.items():
        out[f"loss_weighted_{k}"] = float(v)
    return out


def error_probability_summary(table: pd.DataFrame, task_mode: str) -> List[dict]:
    """Mean predicted probabilities among misclassified validation patients per true class."""
    class_ids = (0, 1) if task_mode in ("binary_diag", "binary_sev") else CLASS_IDS
    prob_cols = [f"p_{k}" for k in class_ids]
    rows = []
    for lbl in class_ids:
        actual = table.loc[table["label"].eq(lbl)]
        wrong = actual.loc[actual["pred"].ne(lbl)]
        row = {
            "label": int(lbl),
            "n_total": int(len(actual)),
            "n_wrong": int(len(wrong)),
        }
        for col in prob_cols:
            row[f"{col}_mean"] = float(wrong[col].mean()) if len(wrong) else float("nan")
        if task_mode == "hierarchical":
            row["p_diag_mean"] = float(wrong["p_diag"].mean()) if len(wrong) else float("nan")
            row["q_comp_mean"] = float(wrong["q_comp"].mean()) if len(wrong) else float("nan")
        rows.append(row)
    return rows


def format_epoch_report(
    variant_label: str,
    fold: int,
    epoch: int,
    epochs: int,
    train: dict,
    val: dict,
    best_epoch: int,
    model_cfg: ModelConfig,
    used_lrs: Dict[str, float],
    backbone_frozen: bool,
    no_improve: int,
    patience: int,
    seconds: float,
) -> str:
    """Format rich per-epoch training & validation diagnostics matching PM2 style."""
    if model_cfg.task_mode == "hierarchical":
        comps = ("diag", "comp", "hier_nll", "concept", "aux_img")
    elif model_cfg.task_mode == "softmax":
        comps = ("softmax", "concept", "aux_img")
    else:
        comps = ("binary", "concept", "aux_img")

    b_state = "FROZEN(warmup)" if backbone_frozen else "TRAINABLE"
    lines = [
        f"\n{variant_label} | fold {fold} | epoch {epoch}/{epochs} | best epoch {best_epoch} | backbone={b_state} | {seconds:.1f}s",
        f"TRAIN | Loss {train['loss_total']:.6f} | Acc {train['accuracy']:.4f} | F1 macro {train['macro_f1']:.4f} | AUROC {train['primary_auroc']:.4f}",
        "  loss raw      : " + " | ".join(f"{k}={train.get('loss_' + k, 0.0):.6f}" for k in comps),
        "  loss weighted : " + " | ".join(f"{k}={train.get('loss_weighted_' + k, 0.0):.6f}" for k in comps),
    ]

    if model_cfg.task_mode in ("binary_diag", "binary_sev"):
        lines.append(
            f"VAL   | Loss {val['loss_total']:.6f} | Acc {val['accuracy']:.4f} | "
            f"Prec {val['precision']:.4f} | Sens {val['sensitivity']:.4f} | Spec {val['specificity']:.4f} | "
            f"F1 pos {val['f1']:.4f} | F1 macro {val['macro_f1']:.4f} | "
            f"AUROC {val['auroc']:.4f} | AUPR {val['aupr']:.4f} | "
            f"Concept AUC {val['concept_mean_auroc']:.4f} | Concept F1 {val['concept_mean_f1']:.4f}"
        )
        names = ["0: Negative", "1: Positive"]
    else:
        lines.append(
            f"VAL   | Loss {val['loss_total']:.6f} | Acc {val['accuracy']:.4f} | "
            f"Prec macro {val['macro_precision']:.4f} | Sens macro {val['macro_sensitivity']:.4f} | "
            f"Spec macro {val['macro_specificity']:.4f} | F1 macro {val['macro_f1']:.4f} | "
            f"AUROC OvR {val['multiclass_auroc']:.4f} | AUPR OvR {val['macro_aupr']:.4f} | "
            f"Concept AUC {val['concept_mean_auroc']:.4f} | Concept F1 {val['concept_mean_f1']:.4f}"
        )
        names = [f"{k}: {name}" for k, name in zip(CLASS_IDS, CLASS_NAMES)]

    lines.extend([
        "  loss raw      : " + " | ".join(f"{k}={val.get('loss_' + k, 0.0):.6f}" for k in comps),
        "  loss weighted : " + " | ".join(f"{k}={val.get('loss_weighted_' + k, 0.0):.6f}" for k in comps),
    ])

    cm = np.asarray(val["confusion_matrix"], dtype=int)
    lines.append("Confusion matrix VAL (baris=label asli, kolom=prediksi):")
    lines.append(pd.DataFrame(cm, index=names, columns=names).to_string())

    err_rows = val.get("error_prob_by_class", [])
    if err_rows:
        edf = pd.DataFrame(err_rows)
        class_map = dict(enumerate(names))
        edf["label"] = edf["label"].map(class_map)
        edf = edf.rename(columns={"label": "kelas_asli", "n_wrong": "n_salah"})
        lines.append("Kesalahan VAL per kelas asli (rata-rata probabilitas kasus salah):")
        lines.append(edf.to_string(index=False, float_format=lambda x: f"{x:.4f}", na_rep="-"))

    diag_keys = [
        k for k in sorted(val.keys())
        if k.startswith(("entropy_norm_", "attn_spread_", "inst_", "bag_", "gate_")) and np.isfinite(val[k])
    ]
    if diag_keys:
        lines.append("MIL & FiLM: " + " | ".join(f"{k}={val[k]:.4f}" for k in diag_keys))

    lines.append(
        "LR used: "
        + " | ".join(f"{name}={value:.6g}" for name, value in used_lrs.items())
        + f" | patience {no_improve}/{patience}"
    )
    return "\n".join(lines)


def train_epoch(
    model: PM3Model,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    weights: dict,
    device: torch.device,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
) -> dict:
    model.train()
    amp = train_cfg.amp_fp16 and device.type == "cuda"
    zero = torch.tensor(0.0, device=device, dtype=torch.float32)
    t_sums = {
        "total": zero.clone(), "diag": zero.clone(), "comp": zero.clone(), "hier_nll": zero.clone(),
        "softmax": zero.clone(), "binary": zero.clone(), "concept": zero.clone(),
        "aux_img_d": zero.clone(), "aux_img_c": zero.clone(), "aux_img": zero.clone(),
    }
    n_patients, n_app_total, n_concepts_total = 0, 0, 0.0
    labels, probs_gpu, preds_gpu = [], [], []

    for items in loader:
        n_batch = len(items)
        n_app = sum(1 for it in items if int(it["label"]) > 0)
        n_con = sum(float(it["concept_mask"].sum()) for it in items)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            if model_cfg.modality != "tabular_only":
                lengths = [len(it["images"]) for it in items]
                all_imgs = torch.cat([it["images"] for it in items], dim=0).to(device, non_blocking=True)
                all_msks = torch.cat([it["pixel_masks"] for it in items], dim=0).to(device, non_blocking=True)
                all_h = model.encode_images_chunked(all_imgs, all_msks, pad_to_microbatch=True)
                h_splits = torch.split(all_h, lengths, dim=0)
            else:
                h_splits = [None] * n_batch

            batch_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
            for item, h_i in zip(items, h_splits):
                tab_i = item["tabular"].to(device, non_blocking=True) if item["tabular"] is not None else None
                out = model.forward_from_embeddings(h_i, tab_i)
                terms = loss_terms(out, item, weights, model_cfg, train_cfg)
                loss_i = combine_batch_loss(terms, n_batch, n_app, n_con, model_cfg, train_cfg)
                batch_loss = batch_loss + loss_i

                t_sums["diag"] = t_sums["diag"] + terms["diag"].detach()
                t_sums["comp"] = t_sums["comp"] + terms["comp"].detach()
                t_sums["hier_nll"] = t_sums["hier_nll"] + terms["hier_nll"].detach()
                t_sums["softmax"] = t_sums["softmax"] + terms["softmax"].detach()
                t_sums["binary"] = t_sums["binary"] + terms["binary"].detach()
                t_sums["concept"] = t_sums["concept"] + terms["concept"].detach()
                if isinstance(terms["aux_img"], tuple):
                    t_sums["aux_img_d"] = t_sums["aux_img_d"] + terms["aux_img"][0].detach()
                    t_sums["aux_img_c"] = t_sums["aux_img_c"] + terms["aux_img"][1].detach()
                else:
                    t_sums["aux_img"] = t_sums["aux_img"] + terms["aux_img"].detach()

                y_true = (
                    int(_binary_target_for_item(item["label"], model_cfg.task_mode))
                    if model_cfg.task_mode in ("binary_diag", "binary_sev")
                    else int(item["label"])
                )
                labels.append(y_true)
                probs_gpu.append(out["probs"].detach())
                preds_gpu.append(out["pred"].detach())

        if not torch.isfinite(batch_loss):
            raise FloatingPointError("Loss non-finite terdeteksi pada batch training.")
        t_sums["total"] = t_sums["total"] + batch_loss.detach() * n_batch
        scaler.scale(batch_loss).backward()

        scaler.unscale_(optimizer)
        finite = all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        if finite:
            clip_gradients(model)
        scaler.step(optimizer)
        scaler.update()

        n_patients += n_batch
        n_app_total += n_app
        n_concepts_total += n_con

    sums = {k: float(v) for k, v in t_sums.items()}
    probs_arr = torch.stack(probs_gpu).float().cpu().numpy()
    preds_arr = torch.stack(preds_gpu).cpu().numpy()
    m = compute_metrics(np.asarray(labels), probs_arr, preds_arr, model_cfg.task_mode)
    loss_summary = summarize_loss_dict(sums, n_patients, n_app_total, n_concepts_total, model_cfg, train_cfg)
    return {
        **loss_summary,
        "accuracy": m["accuracy"],
        "macro_f1": m["macro_f1"],
        "primary_auroc": m["primary_auroc"],
    }


@torch.inference_mode()
def evaluate(
    model: PM3Model,
    loader: DataLoader,
    weights: dict,
    device: torch.device,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    collect_attention: bool = False,
):
    model.eval()
    amp = train_cfg.amp_fp16 and device.type == "cuda"
    zero = torch.tensor(0.0, device=device, dtype=torch.float32)
    t_sums = {
        "total": zero.clone(), "diag": zero.clone(), "comp": zero.clone(), "hier_nll": zero.clone(),
        "softmax": zero.clone(), "binary": zero.clone(), "concept": zero.clone(),
        "aux_img_d": zero.clone(), "aux_img_c": zero.clone(), "aux_img": zero.clone(),
    }
    n_patients, n_app_total, n_concepts_total = 0, 0, 0.0
    rows, attn_rows = [], []

    for items in loader:
        n_batch = len(items)
        n_app = sum(1 for it in items if int(it["label"]) > 0)
        n_con = sum(float(it["concept_mask"].sum()) for it in items)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            if model_cfg.modality != "tabular_only":
                lengths = [len(it["images"]) for it in items]
                all_imgs = torch.cat([it["images"] for it in items], dim=0).to(device, non_blocking=True)
                all_msks = torch.cat([it["pixel_masks"] for it in items], dim=0).to(device, non_blocking=True)
                all_h = model.encode_images_chunked(all_imgs, all_msks, allow_checkpoint=False, pad_to_microbatch=True)
                h_splits = torch.split(all_h, lengths, dim=0)
            else:
                h_splits = [None] * n_batch

            for item, h_i in zip(items, h_splits):
                tab_i = item["tabular"].to(device, non_blocking=True) if item["tabular"] is not None else None
                out = model.forward_from_embeddings(h_i, tab_i)
                terms = loss_terms(out, item, weights, model_cfg, train_cfg)
                loss_i = combine_batch_loss(terms, n_batch, n_app, n_con, model_cfg, train_cfg)

                t_sums["total"] = t_sums["total"] + loss_i * n_batch
                t_sums["diag"] = t_sums["diag"] + terms["diag"]
                t_sums["comp"] = t_sums["comp"] + terms["comp"]
                t_sums["hier_nll"] = t_sums["hier_nll"] + terms["hier_nll"]
                t_sums["softmax"] = t_sums["softmax"] + terms["softmax"]
                t_sums["binary"] = t_sums["binary"] + terms["binary"]
                t_sums["concept"] = t_sums["concept"] + terms["concept"]
                if isinstance(terms["aux_img"], tuple):
                    t_sums["aux_img_d"] = t_sums["aux_img_d"] + terms["aux_img"][0]
                    t_sums["aux_img_c"] = t_sums["aux_img_c"] + terms["aux_img"][1]
                else:
                    t_sums["aux_img"] = t_sums["aux_img"] + terms["aux_img"]

                n_patients += 1
                n_app_total += int(item["label"] > 0)
                n_concepts_total += float(item["concept_mask"].sum())

                y_eval = (
                    int(_binary_target_for_item(item["label"], model_cfg.task_mode))
                    if model_cfg.task_mode in ("binary_diag", "binary_sev")
                    else int(item["label"])
                )
                row = {
                    "patient_id": int(item["patient_id"]),
                    "raw_label": int(item["label"]),
                    "label": y_eval,
                    "pred": int(out["pred"]),
                    "n_images": len(item["image_ids"]),
                }
                if model_cfg.task_mode == "hierarchical":
                    row["p_diag"] = float(out["p"])
                    row["q_comp"] = float(out["q"])
                    for k, p in enumerate(out["probs"]):
                        row[f"p_{k}"] = float(p)
                elif model_cfg.task_mode == "softmax":
                    for k, p in enumerate(out["probs"]):
                        row[f"p_{k}"] = float(p)
                else:
                    row["prob"] = float(out["prob"])
                    row["p_0"] = float(out["probs"][0])
                    row["p_1"] = float(out["probs"][1])

                for k_stat, v_stat in out.get("film_stats", {}).items():
                    row[k_stat] = float(v_stat)

                if "concept_logits" in out:
                    c_probs = out["concept_logits"].sigmoid()
                    for k, cname in enumerate(CONCEPTS_PM3):
                        row[f"concept_p_{cname}"] = float(c_probs[k])
                        row[f"concept_y_{cname}"] = float(item["concepts_binary"][k])
                        row[f"concept_mask_{cname}"] = float(item["concept_mask"][k])

                for branch, a in out["attention"].items():
                    a_np = a.detach().float().cpu().numpy()
                    ent = float(-(a_np * np.log(np.clip(a_np, 1e-7, 1.0))).sum())
                    row[f"entropy_{branch}"] = ent
                    row[f"entropy_norm_{branch}"] = ent / np.log(len(a_np)) if len(a_np) > 1 else np.nan
                    row[f"attn_spread_{branch}"] = float(a_np.max() - a_np.min()) if len(a_np) > 1 else np.nan
                    if collect_attention:
                        for iid, w in zip(item["image_ids"], a_np):
                            attn_rows.append({"patient_id": int(item["patient_id"]), "image_id": iid, "branch": branch, "weight": float(w)})
                rows.append(row)

    sums = {k: float(v) for k, v in t_sums.items()}
    table = pd.DataFrame(rows)
    if model_cfg.task_mode in ("binary_diag", "binary_sev"):
        probs_arr = table[["p_0", "p_1"]].to_numpy()
    else:
        probs_arr = table[["p_0", "p_1", "p_2"]].to_numpy()

    metrics = compute_metrics(table["label"].to_numpy(), probs_arr, table["pred"].to_numpy(), model_cfg.task_mode)
    metrics["error_prob_by_class"] = error_probability_summary(table, model_cfg.task_mode)
    metrics.update(summarize_loss_dict(sums, n_patients, n_app_total, n_concepts_total, model_cfg, train_cfg))

    concept_rows = []
    if f"concept_p_{CONCEPTS_PM3[0]}" in table.columns:
        for cname in CONCEPTS_PM3:
            sub = table.loc[table[f"concept_mask_{cname}"].eq(1)]
            y_c = sub[f"concept_y_{cname}"].to_numpy()
            p_c = sub[f"concept_p_{cname}"].to_numpy()
            auc_c = float(roc_auc_score(y_c, p_c)) if len(np.unique(y_c)) == 2 else float("nan")
            f1_c = float(f1_score(y_c, p_c >= 0.5, zero_division=0)) if len(sub) else float("nan")
            concept_rows.append(
                {"concept": cname, "n_valid": len(sub), "n_positive": int(y_c.sum()), "auroc": auc_c, "f1": f1_c}
            )
    concept_table = pd.DataFrame(concept_rows)
    if len(concept_table):
        metrics["concept_mean_auroc"] = float(concept_table["auroc"].mean())
        metrics["concept_mean_f1"] = float(concept_table["f1"].mean())
    else:
        metrics["concept_mean_auroc"] = float("nan")
        metrics["concept_mean_f1"] = float("nan")

    for col in table.columns:
        if col.startswith(("entropy_", "attn_spread_", "inst_", "bag_", "gate_")):
            metrics[col] = float(table[col].mean())

    return metrics, table, concept_table, pd.DataFrame(attn_rows)


def calibrate_thresholds_nested_oof(oof: pd.DataFrame, task_mode: str) -> Tuple[pd.DataFrame, dict]:
    """Out-of-fold nested threshold calibration (never tunes thresholds on the evaluated fold)."""
    df = oof.copy()
    if task_mode == "softmax":
        df["pred_calibrated"] = df["pred"].astype(int)
        metrics = compute_metrics(
            df["label"].to_numpy(),
            df[["p_0", "p_1", "p_2"]].to_numpy(),
            df["pred_calibrated"].to_numpy(),
            task_mode,
        )
        return df, metrics

    folds = sorted(df["fold"].unique())
    if len(folds) < 2:
        df["pred_calibrated"] = df["pred"].astype(int)
        probs = df[["p_0", "p_1"]].to_numpy() if task_mode in ("binary_diag", "binary_sev") else df[["p_0", "p_1", "p_2"]].to_numpy()
        return df, compute_metrics(df["label"].to_numpy(), probs, df["pred_calibrated"].to_numpy(), task_mode)

    if task_mode == "hierarchical":
        df["tau_diag_used"] = 0.50
        df["tau_comp_used"] = 0.50
        df["pred_calibrated"] = df["pred"].astype(int)
        td_grid = np.round(np.arange(0.35, 0.66, 0.02), 2)
        tc_grid = np.round(np.arange(0.35, 0.76, 0.02), 2)
        for fk in folds:
            tr = df["fold"] != fk
            va = df["fold"] == fk
            p_tr, q_tr, y_tr = df.loc[tr, "p_diag"].to_numpy(), df.loc[tr, "q_comp"].to_numpy(), df.loc[tr, "label"].to_numpy()
            best_score, best_pair = (-1.0, -999.0), (0.50, 0.50)
            for td in td_grid:
                for tc in tc_grid:
                    pr = np.where(p_tr < td, 0, np.where(q_tr < tc, 1, 2))
                    f1 = float(f1_score(y_tr, pr, labels=CLASS_IDS, average="macro", zero_division=0))
                    dist_pen = -(abs(td - 0.50) + abs(tc - 0.50))
                    if (f1, dist_pen) > best_score:
                        best_score = (f1, dist_pen)
                        best_pair = (float(td), float(tc))
            td_star, tc_star = best_pair
            p_va, q_va = df.loc[va, "p_diag"].to_numpy(), df.loc[va, "q_comp"].to_numpy()
            df.loc[va, "tau_diag_used"] = td_star
            df.loc[va, "tau_comp_used"] = tc_star
            df.loc[va, "pred_calibrated"] = np.where(p_va < td_star, 0, np.where(q_va < tc_star, 1, 2))
        metrics = compute_metrics(
            df["label"].to_numpy(),
            df[["p_0", "p_1", "p_2"]].to_numpy(),
            df["pred_calibrated"].to_numpy(),
            task_mode,
        )
        return df, metrics

    # Binary tasks
    df["tau_used"] = 0.50
    df["pred_calibrated"] = df["pred"].astype(int)
    t_grid = np.round(np.arange(0.30, 0.71, 0.02), 2)
    for fk in folds:
        tr = df["fold"] != fk
        va = df["fold"] == fk
        p_tr, y_tr = df.loc[tr, "prob"].to_numpy(), df.loc[tr, "label"].to_numpy()
        best_score, best_t = (-1.0, -999.0), 0.50
        for t in t_grid:
            pr = (p_tr >= t).astype(int)
            f1 = float(f1_score(y_tr, pr, labels=[0, 1], average="macro", zero_division=0))
            dist_pen = -abs(t - 0.50)
            if (f1, dist_pen) > best_score:
                best_score, best_t = (f1, dist_pen), float(t)
        df.loc[va, "tau_used"] = best_t
        df.loc[va, "pred_calibrated"] = (df.loc[va, "prob"].to_numpy() >= best_t).astype(int)
    metrics = compute_metrics(
        df["label"].to_numpy(),
        df[["p_0", "p_1"]].to_numpy(),
        df["pred_calibrated"].to_numpy(),
        task_mode,
    )
    return df, metrics


def bootstrap_ci(table: pd.DataFrame, task_mode: str, n_boot: int = 2000, seed: int = 1, pred_col: str = "pred") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    labels = table["label"].to_numpy(int)
    pred = table[pred_col].to_numpy(int)
    if task_mode in ("binary_diag", "binary_sev"):
        probs = table[["p_0", "p_1"]].to_numpy()
        keys = ["f1", "macro_f1", "sensitivity", "specificity", "auroc"]
        classes = [0, 1]
    else:
        probs = table[["p_0", "p_1", "p_2"]].to_numpy()
        keys = ["macro_f1", "complication_recall", "multiclass_auroc", "accuracy"]
        classes = [0, 1, 2]

    point = compute_metrics(labels, probs, pred, task_mode)
    groups = [np.flatnonzero(labels == k) for k in classes]
    samples = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups if len(g) > 0])
        vals = compute_metrics(labels[idx], probs[idx], pred[idx], task_mode)
        for key in keys:
            samples[key].append(vals[key])
    return pd.DataFrame(
        [
            {
                "metric": k,
                "estimate": point[k],
                "ci95_low": float(np.nanpercentile(samples[k], 2.5)),
                "ci95_high": float(np.nanpercentile(samples[k], 97.5)),
                "n_bootstrap": n_boot,
            }
            for k in keys
        ]
    )


def prepare_fold(
    cohort: pd.DataFrame,
    fold: dict,
    bags: Optional[Dict[int, List[str]]],
    cache: Optional[Dict[str, Path]],
    directory: str | Path,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    pipeline_cls=ClinicalPipeline,
):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    train_df = cohort.loc[fold["train"]]
    val_df = cohort.loc[fold["val"]]

    clinical = pipeline_cls().fit(train_df)
    joblib.dump(clinical, directory / "clinical_pipeline.joblib")
    save_json(
        directory / "preprocessing.json",
        {"columns": getattr(clinical, "columns_", FEATURES), "n_features": len(getattr(clinical, "columns_", FEATURES))},
    )

    weights = training_weights(
        train_df["label"].to_numpy(int),
        *encode_pm3_concepts(train_df, unrecorded_smoothing=train_cfg.unrecorded_smoothing)[1:],
        comp_weight_mode=train_cfg.comp_weight_mode,
        severity_scope=train_cfg.severity_scope,
    )
    save_json(directory / "loss_weights.json", weights)

    include_images = model_cfg.modality != "tabular_only"
    datasets = {}
    for split in ("train", "val"):
        ids = list(fold[split])
        if model_cfg.task_mode == "binary_sev" and train_cfg.severity_scope == "appendicitis_only":
            sub = cohort.loc[ids]
            ids = sub.index[sub["label"] > 0].astype(int).tolist()
        ds = PM3PatientBagDataset(
            ids,
            cohort,
            clinical,
            bags,
            cache,
            train=(split == "train"),
            include_images=include_images,
            unrecorded_smoothing=train_cfg.unrecorded_smoothing,
        )
        datasets[split] = ds
    return datasets, weights


def fit_fold(
    datasets: dict,
    weights: dict,
    train_cfg: TrainConfig,
    model_cfg: ModelConfig,
    directory: str | Path,
    device: torch.device,
    fold_number: int,
    verbose: bool = True,
    variant_label: Optional[str] = None,
):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    seed = train_cfg.seed + fold_number
    set_seed(seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    label_str = variant_label or f"{model_cfg.modality}_{model_cfg.task_mode}_{model_cfg.aggregator}"

    model = PM3Model(model_cfg).to(device)
    backbone_frozen = train_cfg.warmup_freeze_epochs > 0
    if backbone_frozen:
        model.set_backbone_frozen(True)

    optimizer = torch.optim.AdamW(
        model.parameter_groups(
            backbone_lr=train_cfg.backbone_lr,
            attention_lr=train_cfg.attention_lr,
            head_lr=train_cfg.head_lr,
        )
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_cfg.scheduler_t_max, eta_min=train_cfg.scheduler_eta_min
    )
    amp = train_cfg.amp_fp16 and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    train_loader = DataLoader(
        datasets["train"],
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        collate_fn=collate_patients,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        datasets["val"],
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        collate_fn=collate_patients,
    )

    info = {
        "variant_label": label_str,
        "model": asdict(model_cfg),
        "train": asdict(train_cfg),
        "fold": fold_number,
        "seed": seed,
        "device": str(device),
        "parameters_total": sum(p.numel() for p in model.parameters()),
        "parameters_trainable_unfrozen": sum(
            p.numel() for n, p in model.named_parameters() if not n.startswith("encoder.frozen")
        ),
        "selection": "highest validation primary_auroc, then lowest validation loss_total",
    }
    save_json(directory / "config.json", info)

    log_path = directory / "training_log.txt"
    log_path.write_text(
        f"=== {label_str} | fold {fold_number} | params_total={info['parameters_total']} | "
        f"params_unfrozen={info['parameters_trainable_unfrozen']} ===\n",
        encoding="utf-8",
    )

    history, best, best_epoch = [], (-float("inf"), -float("inf")), 0
    no_improve, stopped_early = 0, False
    n_cls = 2 if model_cfg.task_mode in ("binary_diag", "binary_sev") else 3

    for epoch in range(1, train_cfg.epochs + 1):
        if train_cfg.warmup_freeze_epochs > 0 and epoch == train_cfg.warmup_freeze_epochs + 1:
            model.set_backbone_frozen(False)
            backbone_frozen = False

        t0 = time.perf_counter()
        used_lrs = {g["name"]: float(g["lr"]) for g in optimizer.param_groups}
        tr = train_epoch(model, train_loader, optimizer, scaler, weights, device, model_cfg, train_cfg)
        val, _, _, _ = evaluate(model, val_loader, weights, device, model_cfg, train_cfg, collect_attention=False)
        elapsed = time.perf_counter() - t0

        auroc_score = val["primary_auroc"] if np.isfinite(val["primary_auroc"]) else 0.0
        score = (auroc_score, -val["loss_total"])
        if score > best:
            best, best_epoch = score, epoch
            no_improve = 0
            torch.save(
                {"model_state": model.state_dict(), "model_config": asdict(model_cfg), "epoch": epoch},
                directory / "best_model.pt",
            )
        else:
            no_improve += 1

        cm = val["confusion_matrix"]
        row = {
            "epoch": epoch,
            "seconds": elapsed,
            "backbone_frozen": backbone_frozen,
            **{f"lr_{k}": v for k, v in used_lrs.items()},
            **{f"train_{k}": v for k, v in tr.items()},
            **{f"val_{k}": v for k, v in val.items() if np.isscalar(v)},
            **{f"val_cm_{i}_{j}": int(cm[i][j]) for i in range(n_cls) for j in range(n_cls)},
            "best_epoch": best_epoch,
            "no_improve": no_improve,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(directory / "history.csv", index=False)

        rep = format_epoch_report(
            label_str,
            fold_number,
            epoch,
            train_cfg.epochs,
            tr,
            val,
            best_epoch,
            model_cfg,
            used_lrs,
            backbone_frozen,
            no_improve,
            train_cfg.patience,
            elapsed,
        )
        with log_path.open("a", encoding="utf-8") as fp:
            fp.write(rep + "\n")
        if verbose:
            print(rep, flush=True)

        if epoch <= train_cfg.scheduler_t_max:
            scheduler.step()
        if no_improve >= train_cfg.patience and epoch > train_cfg.warmup_freeze_epochs:
            stopped_early = True
            stop_msg = (
                f"\n[EARLY STOPPING] {label_str} | fold {fold_number} berhenti di epoch {epoch} "
                f"(tidak ada perbaikan selama {train_cfg.patience} epoch). Memuat best epoch {best_epoch}."
            )
            with log_path.open("a", encoding="utf-8") as fp:
                fp.write(stop_msg + "\n")
            if verbose:
                print(stop_msg, flush=True)
            break

    state = torch.load(directory / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    model.set_backbone_frozen(False)
    metrics, preds, concepts, attn = evaluate(
        model, val_loader, weights, device, model_cfg, train_cfg, collect_attention=True
    )
    metrics["best_epoch"] = best_epoch
    metrics["epochs_completed"] = len(history)
    metrics["stopped_early"] = stopped_early

    preds["fold"] = fold_number
    preds.to_csv(directory / "val_predictions.csv", index=False)
    concepts.to_csv(directory / "concept_metrics.csv", index=False)
    attn.to_csv(directory / "attention_weights.csv", index=False)

    stat_cols = [
        c for c in preds.columns
        if c in ("patient_id", "raw_label", "label", "pred", "n_images")
        or c.startswith(("entropy_", "attn_spread_", "inst_", "bag_", "gate_"))
    ]
    preds[stat_cols].to_csv(directory / "film_stats.csv", index=False)

    ci_df = bootstrap_ci(preds, model_cfg.task_mode, train_cfg.bootstrap_samples, seed)
    ci_df.to_csv(directory / "ci95.csv", index=False)
    save_json(directory / "val_metrics.json", metrics)

    best_msg = (
        f"\n[BEST FOLD {fold_number} SUMMARY | {label_str}] best_epoch={best_epoch}/{len(history)} | "
        f"acc={metrics['accuracy']:.4f} | macro_f1={metrics['macro_f1']:.4f} | "
        f"auroc={metrics['primary_auroc']:.4f} | concept_auc={metrics['concept_mean_auroc']:.4f}"
    )
    with log_path.open("a", encoding="utf-8") as fp:
        fp.write(best_msg + "\n")
    if verbose:
        print(best_msg, flush=True)

    del state, model, optimizer, scheduler, scaler, train_loader, val_loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics, preds, concepts


def _format_oof_report(
    label_str: str,
    task_mode: str,
    oof_fixed: dict,
    oof_cal: dict,
    ci_df: pd.DataFrame,
    fold_df: pd.DataFrame,
) -> str:
    lines = [
        "=" * 78,
        f"OOF SUMMARY REPORT: {label_str} (task_mode={task_mode})",
        "=" * 78,
        f"FIXED (0.50/0.50)       : acc={oof_fixed['accuracy']:.4f} | macro_f1={oof_fixed['macro_f1']:.4f} | auroc={oof_fixed['primary_auroc']:.4f}",
        f"CALIBRATED (Nested OOF) : acc={oof_cal['accuracy']:.4f} | macro_f1={oof_cal['macro_f1']:.4f} | auroc={oof_cal['primary_auroc']:.4f}",
        "",
        "OOF Confusion Matrix (Nested Calibrated):",
        str(np.asarray(oof_cal["confusion_matrix"], dtype=int)),
        "",
        "OOF 95% Bootstrap CI (Nested Calibrated):",
        ci_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        "",
        "Per-Fold Summary (Best Epoch & Validation Metrics):",
    ]
    show_cols = [c for c in ("fold", "best_epoch", "epochs_completed", "accuracy", "macro_f1", "primary_auroc", "concept_mean_auroc", "loss_total") if c in fold_df.columns]
    lines.append(fold_df[show_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    lines.append("=" * 78)
    return "\n".join(lines)


def run_experiment(
    cohort: pd.DataFrame,
    folds: List[dict],
    bags: Optional[Dict[int, List[str]]],
    cache: Optional[Dict[str, Path]],
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    directory: str | Path,
    device: torch.device,
    pipeline_cls=ClinicalPipeline,
    verbose: bool = True,
    skip_if_done: bool = True,
    variant_label: Optional[str] = None,
) -> dict:
    """Run full K-fold CV with per-fold/experiment resume and complete saved artifacts."""
    directory = Path(directory)
    label_str = variant_label or directory.name

    cal_json = directory / "oof_metrics_calibrated.json"
    fix_json = directory / "oof_metrics_fixed.json"
    oof_csv = directory / "oof_predictions.csv"
    fold_csv = directory / "fold_metrics.csv"
    if skip_if_done and cal_json.exists() and fix_json.exists() and oof_csv.exists() and fold_csv.exists():
        if verbose:
            print(f"[SKIP] Run sudah selesai, memuat hasil dari: {directory}", flush=True)
        return {
            "fold_metrics": pd.read_csv(fold_csv),
            "oof_fixed": json.loads(fix_json.read_text(encoding="utf-8")),
            "oof_calibrated": json.loads(cal_json.read_text(encoding="utf-8")),
            "oof_predictions": pd.read_csv(oof_csv),
            "directory": directory,
        }

    directory.mkdir(parents=True, exist_ok=True)
    save_json(directory / "folds.json", folds)
    save_json(
        directory / "experiment_config.json",
        {"variant_label": label_str, "model": asdict(model_cfg), "train": asdict(train_cfg)},
    )

    fold_rows, oof_list, concept_list = [], [], []
    for fold in folds:
        fk = fold["fold"]
        fold_model_dir = directory / f"fold_{fk}" / "model"
        vm_path = fold_model_dir / "val_metrics.json"
        vp_path = fold_model_dir / "val_predictions.csv"
        cm_path = fold_model_dir / "concept_metrics.csv"

        if skip_if_done and vm_path.exists() and vp_path.exists():
            metrics = json.loads(vm_path.read_text(encoding="utf-8"))
            preds = pd.read_csv(vp_path)
            concepts = pd.read_csv(cm_path) if cm_path.exists() else pd.DataFrame()
            if verbose:
                print(
                    f"[RESUME] Fold {fk} ({label_str}) sudah selesai — memuat hasil tersimpan "
                    f"(val_auc={metrics.get('primary_auroc', 0):.4f}, val_f1={metrics.get('macro_f1', 0):.4f}).",
                    flush=True,
                )
        else:
            datasets, weights = prepare_fold(
                cohort,
                fold,
                bags,
                cache,
                directory / f"fold_{fk}" / "data",
                model_cfg,
                train_cfg,
                pipeline_cls=pipeline_cls,
            )
            metrics, preds, concepts = fit_fold(
                datasets,
                weights,
                train_cfg,
                model_cfg,
                fold_model_dir,
                device,
                fk,
                verbose=verbose,
                variant_label=label_str,
            )
            del datasets

        fold_rows.append({"fold": fk, **{k: v for k, v in metrics.items() if np.isscalar(v)}})
        oof_list.append(preds)
        if len(concepts):
            concept_list.append(concepts.assign(fold=fk))
        pd.DataFrame(fold_rows).to_csv(fold_csv, index=False)

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(fold_csv, index=False)

    num_cols = [c for c in fold_df.columns if c != "fold" and pd.api.types.is_numeric_dtype(fold_df[c])]
    fold_df[num_cols].agg(["mean", "std"]).to_csv(directory / "mean_std_summary.csv")

    oof = pd.concat(oof_list, ignore_index=True).sort_values("patient_id").reset_index(drop=True)
    oof, oof_cal = calibrate_thresholds_nested_oof(oof, model_cfg.task_mode)
    oof.to_csv(oof_csv, index=False)

    probs_cols = ["p_0", "p_1"] if model_cfg.task_mode in ("binary_diag", "binary_sev") else ["p_0", "p_1", "p_2"]
    oof_fixed = compute_metrics(
        oof["label"].to_numpy(), oof[probs_cols].to_numpy(), oof["pred"].to_numpy(), model_cfg.task_mode
    )
    save_json(fix_json, oof_fixed)
    save_json(cal_json, oof_cal)
    ci_df = bootstrap_ci(
        oof, model_cfg.task_mode, train_cfg.bootstrap_samples, train_cfg.seed, pred_col="pred_calibrated"
    )
    ci_df.to_csv(directory / "oof_ci95.csv", index=False)

    if concept_list:
        cdf = pd.concat(concept_list, ignore_index=True)
        cdf.groupby("concept")[["n_valid", "n_positive", "auroc", "f1"]].agg(
            {"n_valid": "sum", "n_positive": "sum", "auroc": "mean", "f1": "mean"}
        ).to_csv(directory / "concept_summary.csv")

    report_txt = _format_oof_report(label_str, model_cfg.task_mode, oof_fixed, oof_cal, ci_df, fold_df)
    (directory / "oof_summary_report.txt").write_text(report_txt + "\n", encoding="utf-8")
    if verbose:
        print("\n" + report_txt, flush=True)

    return {
        "fold_metrics": fold_df,
        "oof_fixed": oof_fixed,
        "oof_calibrated": oof_cal,
        "oof_predictions": oof,
        "directory": directory,
    }
