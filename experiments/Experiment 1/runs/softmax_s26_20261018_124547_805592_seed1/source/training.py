#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
training.py
===========

Loop pelatihan + validasi per epoch, metrik klasifikasi, dan checkpoint terbaik
untuk `MultimodalMILModel` (multimodal_mil_model.py).

DAFTAR ISI
----------
1. Yang dilaporkan tiap epoch
2. Cara metrik dihitung
3. Alur pemakaian
4. Catatan
5. Isi file

-------------------------------------------------------------------------------
1. YANG DILAPORKAN TIAP EPOCH
-------------------------------------------------------------------------------

   Untuk train DAN validasi:
       loss, accuracy, precision, sensitivity, specificity, F1-score

   Contoh keluaran:
       Epoch 03/20 | lr 1.0e-04 | 41.2 dtk
         train | loss 0.9120 | acc 0.6120 | prec 0.5800 | sens 0.5500 | spec 0.7800 | F1 0.5400
         val   | loss 0.8740 | acc 0.6500 | prec 0.6100 | sens 0.6000 | spec 0.8000 | F1 0.5900  *terbaik*

   "Error loss" pada permintaan = loss validasi (`val_loss`).
   Semua angka juga disimpan di DataFrame `history` (satu baris per epoch), termasuk
   rincian loss (mis. loss_diagnosis dan loss_complication pada dual head).

-------------------------------------------------------------------------------
2. CARA METRIK DIHITUNG
-------------------------------------------------------------------------------

   Tugas ini 3 kelas hierarkis: 0 = no appendicitis, 1 = uncomplicated, 2 = complicated.
   Prediksi kelas diambil dari model.predict_classes (dual head: keputusan hierarkis
   p >= 0.5 lalu q >= 0.5; softmax: argmax).

   Metrik utama (di tabel dan history) = RATA-RATA MACRO atas 3 kelas, tiap kelas
   dihitung satu-lawan-sisa dari confusion matrix:
       sensitivity (recall) = TP / (TP + FN)
       specificity          = TN / (TN + FP)
       precision            = TP / (TP + FP)
       F1                   = 2 * precision * sensitivity / (precision + sensitivity)
       accuracy             = (jumlah benar) / (jumlah pasien)     [bukan macro]
   Macro berarti tiap kelas berbobot sama, sehingga kelas kecil (complicated) tidak
   tertutup kelas besar. Kelas yang tidak muncul di data asli (support 0) tidak ikut
   dirata-rata. Pembagian 0/0 dianggap 0.

   Selain itu `detailed_report` memberi: confusion matrix, metrik per kelas, dan dua
   metrik biner per tingkat hierarki:
       Level 1: apendisitis (kelas 1 atau 2) vs bukan.
       Level 2: complicated vs uncomplicated, hanya pada pasien yang SEBENARNYA apendisitis
                (prediksi kelas 0 pada pasien itu dihitung sebagai "bukan complicated").

-------------------------------------------------------------------------------
3. ALUR PEMAKAIAN
-------------------------------------------------------------------------------

   from training import set_seed, fit, plot_history, detailed_report, load_best

   set_seed(42)
   model = MultimodalMILModel(cfg).to(device)
   optimizer = build_optimizer(model, OptimizerConfig(...))

   history = fit(model, train_loader, val_loader, optimizer, device=device,
                 epochs=30, monitor="val_f1", checkpoint_path="best_model.pt")
   plot_history(history)
   detailed_report(model, val_loader, device)        # model sudah berisi bobot terbaik

   # nanti, memuat bobot terbaik di sesi lain:
   load_best(model, "best_model.pt", device)

-------------------------------------------------------------------------------
4. CATATAN
-------------------------------------------------------------------------------

   * Metrik TRAIN dihitung dari prediksi saat training (model mode train: dropout aktif,
     citra diaugmentasi, BN memakai statistik batch), jadi biasanya lebih rendah dari
     jika dievaluasi ulang dalam mode eval. Gunakan untuk melihat tren, bukan angka akhir.
   * `monitor` menentukan epoch "terbaik": "val_loss" (semakin kecil semakin baik) atau
     salah satu dari val_accuracy / val_precision / val_sensitivity / val_specificity /
     val_f1 (semakin besar semakin baik). Bobot terbaik disimpan ke `checkpoint_path`
     dan, bila restore_best=True, dimuat kembali ke model setelah training selesai.
   * Mixed precision (AMP) otomatis aktif hanya di GPU CUDA.
   * Belum ada: scheduler learning rate, early stopping, pembobotan kelas. Test set
     sengaja tidak disentuh di sini.

-------------------------------------------------------------------------------
5. ISI FILE
-------------------------------------------------------------------------------
   set_seed -> compute_metrics / binary_metrics -> run_epoch -> fit -> plot_history ->
   detailed_report / load_best -> _smoke_test (jalankan: python training.py)
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from multimodal_mil_model import CLASS_NAMES, labels_to_class

METRIC_KEYS = ("accuracy", "precision", "sensitivity", "specificity", "f1")
# Metrik yang semakin besar semakin baik; val_loss kebalikannya
_MAXIMIZE = {f"val_{k}" for k in METRIC_KEYS}
_MAXIMIZE.update({"val_auroc", "val_diagnosis_auroc", "val_severity_auroc"})
PathLike = Union[str, Path]


# =============================================================================
# SEED
# =============================================================================

def set_seed(seed: int = 42) -> None:
    """Atur seed Python, NumPy, dan PyTorch (CPU + semua GPU) agar hasil dapat diulang.
    Catatan: operasi GPU tertentu tetap non-deterministik; hasil bisa sedikit berbeda."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =============================================================================
# METRIK
# =============================================================================

def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """num / den dengan 0/0 -> 0 (elemen-per-elemen)."""
    num, den = np.asarray(num, dtype=float), np.asarray(den, dtype=float)
    return np.divide(num, den, out=np.zeros_like(num), where=den != 0)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 3) -> dict:
    """Metrik klasifikasi multikelas (rata-rata macro, satu-lawan-sisa).

    Mengembalikan dict berisi: accuracy, precision, sensitivity, specificity, f1 (skalar),
    "confusion_matrix" (baris = kelas sebenarnya, kolom = prediksi), dan "per_class"
    (DataFrame metrik tiap kelas beserta support).
    """
    y_true, y_pred = np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    np.add.at(cm, (y_true, y_pred), 1)

    total = cm.sum()
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = total - tp - fp - fn

    precision = _safe_div(tp, tp + fp)
    sensitivity = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    f1 = _safe_div(2 * precision * sensitivity, precision + sensitivity)

    support = cm.sum(axis=1)
    present = support > 0                      # hanya kelas yang ada di data asli ikut macro
    macro = (lambda v: float(v[present].mean()) if present.any() else 0.0)

    return {
        "accuracy": float(tp.sum() / total) if total else 0.0,
        "precision": macro(precision),
        "sensitivity": macro(sensitivity),
        "specificity": macro(specificity),
        "f1": macro(f1),
        "confusion_matrix": cm,
        "per_class": pd.DataFrame({
            "precision": precision, "sensitivity": sensitivity, "specificity": specificity,
            "f1": f1, "support": support,
        }, index=list(CLASS_NAMES[:num_classes])),
    }


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Metrik untuk satu kelas positif (array bool/0-1)."""
    y_true, y_pred = np.asarray(y_true, dtype=bool), np.asarray(y_pred, dtype=bool)
    tp = float((y_true & y_pred).sum())
    fp = float((~y_true & y_pred).sum())
    fn = float((y_true & ~y_pred).sum())
    tn = float((~y_true & ~y_pred).sum())
    precision = float(_safe_div(tp, tp + fp))
    sensitivity = float(_safe_div(tp, tp + fn))
    return {
        "accuracy": float(_safe_div(tp + tn, tp + tn + fp + fn)),
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": float(_safe_div(tn, tn + fp)),
        "f1": float(_safe_div(2 * precision * sensitivity, precision + sensitivity)),
        "support_positive": int(tp + fn),
        "support_negative": int(tn + fp),
    }


def probability_metrics(y_true, probabilities, diagnosis_probability=None,
                        severity_probability=None):
    """AUROC seluruh cohort, bukan rata-rata AUROC per batch.

    auroc: macro OvR tiga kelas. Diagnosis: p pada seluruh pasien.
    Severity: q hanya pada pasien appendicitis asli. Tidak terdefinisi -> NaN;
    pemilihan checkpoint memakai AUROC harus menolak nilai tersebut.
    """
    y = np.asarray(y_true, dtype=int)
    proba = np.asarray(probabilities, dtype=float)
    if proba.shape != (len(y), 3) or not np.isfinite(proba).all():
        raise ValueError("Probability harus finite dengan shape (N, 3)")
    if not np.isin(y, [0, 1, 2]).all():
        raise ValueError("Label harus 0, 1, atau 2")
    if ((proba < 0) | (proba > 1)).any() or not np.allclose(proba.sum(1), 1, atol=1e-6):
        raise ValueError("Probability kelas harus berada di [0,1] dan berjumlah satu")
    p = proba[:, 1:].sum(1) if diagnosis_probability is None else np.asarray(diagnosis_probability)
    q = (np.divide(proba[:, 2], p, out=np.zeros_like(p), where=p > 0)
         if severity_probability is None else np.asarray(severity_probability))
    if p.shape != y.shape or q.shape != y.shape or not (np.isfinite(p).all() and np.isfinite(q).all()):
        raise ValueError("Probability diagnosis/severity tidak valid")

    def auc(target, score):
        return float(roc_auc_score(target, score)) if np.unique(target).size == 2 else float("nan")

    app = y > 0
    return {
        "auroc": float(roc_auc_score(y, proba, labels=[0, 1, 2], multi_class="ovr", average="macro"))
                 if np.unique(y).size == 3 else float("nan"),
        "diagnosis_auroc": auc(app, p),
        "severity_auroc": auc(y[app] == 2, q[app]),
    }


# =============================================================================
# SATU EPOCH
# =============================================================================

def run_epoch(model: nn.Module, loader: DataLoader, device: torch.device,
              optimizer: Optional[torch.optim.Optimizer] = None,
              scaler: Optional[torch.amp.GradScaler] = None,
              use_amp: bool = False) -> Tuple[dict, np.ndarray, np.ndarray]:
    """Jalankan satu epoch. optimizer diberikan -> training; None -> evaluasi (tanpa gradien).

    Mengembalikan (ringkasan, y_true_kelas, y_pred_kelas). `ringkasan` berisi loss rata-rata
    per pasien ("loss") dan rata-rata tiap komponen loss ("loss_diagnosis", dst.).
    """
    training = optimizer is not None
    model.train(training)

    sums: Dict[str, float] = {}
    n_patients = 0
    y_true: List[np.ndarray] = []
    y_pred: List[np.ndarray] = []
    probabilities, diagnosis_probabilities, severity_probabilities = [], [], []

    with torch.set_grad_enabled(training):
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, enabled=use_amp):
                out = model(batch["images"], batch["image_mask"], batch["tabular"])
                loss, parts = model.compute_loss(out, batch["y_appendicitis"], batch["y_complicated"])

            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            bs = batch["images"].size(0)
            n_patients += bs
            for k, v in parts.items():                    # rata-rata berbobot jumlah pasien
                sums[k] = sums.get(k, 0.0) + v * bs
            y_true.append(labels_to_class(batch["y_appendicitis"], batch["y_complicated"]).cpu().numpy())
            y_pred.append(model.predict_classes(out).cpu().numpy())
            probabilities.append(model.predict_proba(out).detach().cpu().numpy())
            if "logit_diagnosis" in out:
                diagnosis_probabilities.append(out["logit_diagnosis"].detach().float().sigmoid().cpu().numpy())
                severity_probabilities.append(out["logit_complication"].detach().float().sigmoid().cpu().numpy())

    summary = {("loss" if k == "loss_total" else k): v / n_patients for k, v in sums.items()}
    summary.update(probability_metrics(
        np.concatenate(y_true), np.concatenate(probabilities),
        np.concatenate(diagnosis_probabilities) if diagnosis_probabilities else None,
        np.concatenate(severity_probabilities) if severity_probabilities else None,
    ))
    return summary, np.concatenate(y_true), np.concatenate(y_pred)


# =============================================================================
# FIT
# =============================================================================

def _fmt_line(tag: str, loss: float, m: dict) -> str:
    return (f"  {tag:<5} | loss {loss:.4f} | acc {m['accuracy']:.4f} | prec {m['precision']:.4f} | "
            f"sens {m['sensitivity']:.4f} | spec {m['specificity']:.4f} | F1 {m['f1']:.4f}")


def fit(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
        optimizer: torch.optim.Optimizer, device: Union[str, torch.device], epochs: int,
        monitor: str = "val_f1", checkpoint_path: Optional[PathLike] = "best_model.pt",
        restore_best: bool = True, use_amp: Optional[bool] = None,
        verbose: bool = True, early_stopping_patience: Optional[int] = None,
        early_stopping_min_delta: float = 0.0, *,
        epoch_runner=None, metric_fn=None,
        early_stopping_rule: str = "previous_epoch_val_loss",
        checkpoint_tie_break: Optional[str] = None) -> pd.DataFrame:
    """Latih `epochs` epoch; tiap epoch lakukan train lalu validasi dan laporkan metrik.

    Argumen penting:
        monitor         : "val_loss" (minimum terbaik), val_auroc (macro OvR tiga kelas),
                          val_diagnosis_auroc / val_severity_auroc, atau val_accuracy / val_precision /
                          val_sensitivity / val_specificity / val_f1 (maksimum terbaik).
        checkpoint_path : file untuk menyimpan bobot epoch terbaik (None = tidak disimpan).
        restore_best    : True = bobot terbaik dimuat kembali ke `model` setelah selesai.
        use_amp         : mixed precision; None = otomatis (aktif hanya di CUDA).
        early_stopping_patience : None = nonaktif; N = berhenti setelah N transisi
                          epoch berturut-turut tanpa penurunan val_loss dari epoch
                          SEBELUMNYA (bukan dari best loss). Epoch pertama baseline.
        early_stopping_min_delta : penurunan harus > nilai ini untuk reset hitungan.
                          Default 0.0: setiap penurunan strict diakui.

    early_stopping_rule='previous_epoch_val_loss' mempertahankan perilaku lama.
    'best_monitor' menghitung patience sejak perbaikan terbaik metrik `monitor`.
    checkpoint_tie_break='val_loss': jika nilai monitor sama persis, pilih loss
    lebih rendah; tie-break ini tidak mereset patience best_monitor.
    `epoch_runner` dan `metric_fn` opsional untuk task lain (misalnya binary);
    default tetap runner dan metrik hierarkis tiga kelas.

    Mengembalikan DataFrame `history` (satu baris per epoch). Epoch terbaik tersimpan di
    history.attrs["best_epoch"] dan history.attrs["best_value"].
    """
    device = torch.device(device)
    epoch_runner = run_epoch if epoch_runner is None else epoch_runner
    metric_fn = compute_metrics if metric_fn is None else metric_fn
    if early_stopping_patience is not None and (
        isinstance(early_stopping_patience, bool)
        or not isinstance(early_stopping_patience, int)
        or early_stopping_patience < 1
    ):
        raise ValueError("early_stopping_patience harus integer >= 1 atau None")
    if not np.isfinite(early_stopping_min_delta) or early_stopping_min_delta < 0:
        raise ValueError("early_stopping_min_delta harus finite dan >= 0")
    if early_stopping_rule not in {"previous_epoch_val_loss", "best_monitor"}:
        raise ValueError("early_stopping_rule tidak dikenal")
    if checkpoint_tie_break not in {None, "val_loss"}:
        raise ValueError("checkpoint_tie_break harus None atau val_loss")
    model.to(device)
    if monitor != "val_loss" and monitor not in _MAXIMIZE:
        raise ValueError(f"monitor='{monitor}' tidak dikenal. Pilihan: "
                         f"{['val_loss'] + sorted(_MAXIMIZE)}")
    maximize = monitor in _MAXIMIZE
    if checkpoint_path is not None:
        # Buat folder induk di awal (torch.save tidak membuatnya) agar kegagalan
        # tidak baru ketahuan setelah satu epoch selesai.
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    if use_amp is None:
        use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    best_value = -np.inf if maximize else np.inf
    best_epoch = 0
    best_state = None
    rows: List[dict] = []
    previous_val_loss = None
    epochs_without_loss_decrease = 0
    last_loss_decrease_epoch = None
    early_stopping_triggered = False
    stopping_best = -np.inf if maximize else np.inf
    epochs_without_monitor_improvement = 0
    last_monitor_improvement_epoch = None
    best_checkpoint_loss = np.inf

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        tr_sum, tr_true, tr_pred = epoch_runner(model, train_loader, device, optimizer, scaler, use_amp)
        va_sum, va_true, va_pred = epoch_runner(model, val_loader, device, None, None, use_amp)
        tr_m, va_m = metric_fn(tr_true, tr_pred), metric_fn(va_true, va_pred)

        row = {"epoch": epoch,
               "lr": optimizer.param_groups[0]["lr"],
               "seconds": time.perf_counter() - t0}
        for prefix, summ, met in (("train", tr_sum, tr_m), ("val", va_sum, va_m)):
            row[f"{prefix}_loss"] = summ["loss"]
            row.update({f"{prefix}_{k}": met[k] for k in METRIC_KEYS})
            row.update({f"{prefix}_{k}": v for k, v in summ.items() if k != "loss"})
        if early_stopping_patience is not None and early_stopping_rule == "previous_epoch_val_loss":
            current_loss = row["val_loss"]
            if not np.isfinite(current_loss):
                raise FloatingPointError(f"val_loss tidak finite pada epoch {epoch}: {current_loss}")
            if previous_val_loss is None:
                last_loss_decrease_epoch = epoch
            elif previous_val_loss - current_loss > early_stopping_min_delta:
                epochs_without_loss_decrease = 0
                last_loss_decrease_epoch = epoch
            else:
                epochs_without_loss_decrease += 1
            previous_val_loss = current_loss
            row["epochs_without_loss_decrease"] = epochs_without_loss_decrease
        rows.append(row)

        if monitor not in row or not np.isfinite(row[monitor]):
            raise ValueError(f"Monitor {monitor} tidak tersedia atau tidak finite pada epoch {epoch}; "
                             "periksa definisi metrik dan kelas dalam validation.")
        value = row[monitor]
        improved = value > best_value if maximize else value < best_value
        tie_improved = (checkpoint_tie_break == "val_loss" and value == best_value
                        and np.isfinite(row["val_loss"]) and row["val_loss"] < best_checkpoint_loss)
        if early_stopping_patience is not None and early_stopping_rule == "best_monitor":
            stop_improved = (value > stopping_best + early_stopping_min_delta if maximize
                             else value < stopping_best - early_stopping_min_delta)
            if stop_improved:
                stopping_best = value
                epochs_without_monitor_improvement = 0
                last_monitor_improvement_epoch = epoch
            else:
                epochs_without_monitor_improvement += 1
            row["epochs_without_monitor_improvement"] = epochs_without_monitor_improvement
        if improved or tie_improved:
            best_value, best_epoch = value, epoch
            best_checkpoint_loss = row["val_loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if checkpoint_path is not None:
                torch.save({"epoch": epoch, "monitor": monitor, "value": value,
                            "model_state": best_state,
                            "config": getattr(model, "cfg", None) and model.cfg.to_dict()},
                           checkpoint_path)

        if verbose:
            print(f"Epoch {epoch:02d}/{epochs} | lr {row['lr']:.1e} | {row['seconds']:.1f} dtk")
            print(_fmt_line("train", row["train_loss"], tr_m)
                  + (f" | AUROC {row['train_auroc']:.4f}" if "train_auroc" in row else ""))
            print(_fmt_line("val", row["val_loss"], va_m)
                  + (f" | AUROC {row['val_auroc']:.4f}" if "val_auroc" in row else "")
                  + ("  *terbaik*" if improved or tie_improved else ""))

        if (early_stopping_patience is not None
                and (epochs_without_monitor_improvement if early_stopping_rule == "best_monitor"
                     else epochs_without_loss_decrease) >= early_stopping_patience):
            early_stopping_triggered = True
            if verbose:
                if early_stopping_rule == "best_monitor":
                    print(f"Early stopping pada epoch {epoch}: {monitor} tidak memperbaiki best historis "
                          f"selama {epochs_without_monitor_improvement} epoch (min_delta={early_stopping_min_delta:g}).")
                else:
                    print(f"Early stopping pada epoch {epoch}: val_loss tidak turun dari epoch "
                          f"sebelumnya selama {epochs_without_loss_decrease} epoch berturut-turut "
                          f"(min_delta={early_stopping_min_delta:g}).")
            break

    if restore_best and best_state is not None:
        model.load_state_dict(best_state)
    if verbose:
        print(f"\nSelesai. Epoch terbaik: {best_epoch} ({monitor} = {best_value:.4f})"
              + (f" -> {checkpoint_path}" if checkpoint_path is not None else ""))

    history = pd.DataFrame(rows).set_index("epoch")
    history.attrs.update(best_epoch=best_epoch, best_value=best_value, monitor=monitor)
    history.attrs.update(early_stopping_patience=early_stopping_patience,
                         early_stopping_min_delta=early_stopping_min_delta,
                         early_stopping_rule=early_stopping_rule,
                         early_stopping_triggered=early_stopping_triggered,
                         stopped_early=early_stopping_triggered and len(rows) < epochs,
                         epochs_ran=len(rows),
                         last_loss_decrease_epoch=last_loss_decrease_epoch)
    history.attrs.update(early_stopping_monitor=monitor if early_stopping_rule == "best_monitor" else "val_loss",
                         last_monitor_improvement_epoch=last_monitor_improvement_epoch,
                         checkpoint_tie_break=checkpoint_tie_break,
                         best_checkpoint_val_loss=best_checkpoint_loss)
    return history


# =============================================================================
# VISUALISASI & LAPORAN
# =============================================================================

def plot_history(history: pd.DataFrame):
    """Kurva loss dan 5 metrik, train vs validasi, per epoch."""
    import matplotlib.pyplot as plt

    names = [("loss", "Loss")] + [(k, k.capitalize() if k != "f1" else "F1-score") for k in METRIC_KEYS]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    for ax, (key, title) in zip(axes.ravel(), names):
        ax.plot(history.index, history[f"train_{key}"], marker="o", ms=3, label="train")
        ax.plot(history.index, history[f"val_{key}"], marker="o", ms=3, label="val")
        best = history.attrs.get("best_epoch")
        if best in history.index:
            ax.axvline(best, color="gray", ls=":", lw=1)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    return fig


@torch.no_grad()
def predict_loader(model: nn.Module, loader: DataLoader,
                   device: Union[str, torch.device]) -> Tuple[np.ndarray, np.ndarray]:
    """Prediksi kelas seluruh loader (mode eval) -> (y_true_kelas, y_pred_kelas)."""
    device = torch.device(device)
    model.to(device).eval()
    y_true, y_pred = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch["images"], batch["image_mask"], batch["tabular"])
        y_true.append(labels_to_class(batch["y_appendicitis"], batch["y_complicated"]).cpu().numpy())
        y_pred.append(model.predict_classes(out).cpu().numpy())
    return np.concatenate(y_true), np.concatenate(y_pred)


def detailed_report(model: nn.Module, loader: DataLoader, device: Union[str, torch.device]) -> dict:
    """Laporan rinci pada satu loader (mode eval): confusion matrix, metrik per kelas,
    dan metrik biner Level 1 (apendisitis vs bukan) serta Level 2 (complicated vs
    uncomplicated pada pasien yang sebenarnya apendisitis). Hasil juga dicetak."""
    y_true, y_pred = predict_loader(model, loader, device)
    overall = compute_metrics(y_true, y_pred)

    level1 = binary_metrics(y_true > 0, y_pred > 0)
    is_app = y_true > 0
    level2 = binary_metrics(y_true[is_app] == 2, y_pred[is_app] == 2)

    cm = pd.DataFrame(overall["confusion_matrix"],
                      index=[f"asli: {c}" for c in CLASS_NAMES],
                      columns=[f"prediksi: {c}" for c in CLASS_NAMES])
    print("Confusion matrix (baris = kelas asli, kolom = prediksi)")
    print(cm.to_string(), "\n")
    print("Metrik per kelas (satu-lawan-sisa)")
    print(overall["per_class"].round(4).to_string(), "\n")
    print("Macro 3 kelas   :", {k: round(overall[k], 4) for k in METRIC_KEYS})
    print("Level 1 (apendisitis vs bukan)             :", {k: round(v, 4) for k, v in level1.items()})
    print("Level 2 (complicated vs uncomplicated)     :", {k: round(v, 4) for k, v in level2.items()})
    return {"overall": overall, "level1": level1, "level2": level2, "confusion_matrix": cm}


def load_best(model: nn.Module, checkpoint_path: PathLike,
              device: Union[str, torch.device] = "cpu") -> dict:
    """Muat bobot dari checkpoint `fit` ke `model`. Mengembalikan isi checkpoint (epoch, value, ...)."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    return ckpt


# =============================================================================
# SMOKE TEST  (python training.py)
# =============================================================================

def _smoke_test() -> None:
    """Uji metrik (dibandingkan dengan scikit-learn) dan loop `fit` pada data sintetis."""
    import tempfile
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

    from multimodal_mil_model import ModelConfig, MultimodalMILModel, OptimizerConfig, build_optimizer, collate_bags

    # 1) metrik vs scikit-learn
    rng = np.random.default_rng(0)
    y_t = rng.integers(0, 3, 200)
    y_p = np.where(rng.random(200) < 0.6, y_t, rng.integers(0, 3, 200))
    m = compute_metrics(y_t, y_p)
    p, r, f, _ = precision_recall_fscore_support(y_t, y_p, average="macro", zero_division=0)
    assert abs(m["accuracy"] - accuracy_score(y_t, y_p)) < 1e-12
    assert abs(m["precision"] - p) < 1e-12 and abs(m["sensitivity"] - r) < 1e-12
    assert abs(m["f1"] - np.mean([2 * a * b / (a + b) if a + b else 0
                                  for a, b in zip(*precision_recall_fscore_support(y_t, y_p, zero_division=0)[:2])])) < 1e-12
    assert np.array_equal(m["confusion_matrix"], confusion_matrix(y_t, y_p))
    cm = m["confusion_matrix"]
    spec0 = (cm.sum() - cm[0].sum() - cm[:, 0].sum() + cm[0, 0]) / (cm.sum() - cm[0].sum())   # spesifisitas kelas 0 manual
    assert abs(m["per_class"]["specificity"].iloc[0] - spec0) < 1e-12
    # kelas yang tidak ada di data asli tidak ikut macro; 0/0 -> 0 tanpa error
    m2 = compute_metrics(np.array([0, 0, 1, 1]), np.array([0, 0, 1, 0]))
    assert m2["accuracy"] == 0.75 and m2["per_class"].loc[CLASS_NAMES[2], "support"] == 0
    b = binary_metrics(np.array([1, 1, 0, 0]), np.array([1, 0, 0, 0]))
    assert b["sensitivity"] == 0.5 and b["specificity"] == 1.0 and b["precision"] == 1.0
    print("[smoke] metrik cocok dengan scikit-learn: OK")

    # 2) fit pada data sintetis: 3 kelas, dua epoch, semua head
    torch.manual_seed(0)
    n_feat, n = 10, 24
    labels3 = torch.arange(n) % 3

    class Toy(torch.utils.data.Dataset):
        def __len__(self): return n
        def __getitem__(self, i):
            k = int(labels3[i])
            return {"images": torch.randn(1 + i % 3, 3, 32, 32) + k,
                    "tabular": torch.randn(n_feat) + k,
                    "y_appendicitis": int(k > 0), "y_complicated": (k - 1) if k > 0 else -1}

    loader = DataLoader(Toy(), batch_size=6, shuffle=True, collate_fn=collate_bags, drop_last=True)
    val_loader = DataLoader(Toy(), batch_size=6, shuffle=False, collate_fn=collate_bags)

    with tempfile.TemporaryDirectory() as tmp:
        for preset in ("P", "B5"):
            cfg = ModelConfig.from_preset(preset, num_tabular_features=n_feat, pretrained=False,
                                          tabular_encoder="mlp")
            model = MultimodalMILModel(cfg)
            opt = build_optimizer(model, OptimizerConfig(lr=1e-3))
            ckpt = Path(tmp) / f"{preset}.pt"
            hist = fit(model, loader, val_loader, opt, device="cpu", epochs=2,
                       monitor="val_f1", checkpoint_path=ckpt, verbose=(preset == "P"))
            assert len(hist) == 2 and ckpt.exists()
            for col in ["train_loss", "val_loss"] + [f"{s}_{k}" for s in ("train", "val") for k in METRIC_KEYS]:
                assert col in hist.columns and np.isfinite(hist[col]).all(), col
            assert hist.attrs["best_epoch"] in (1, 2)
            info = load_best(MultimodalMILModel(cfg), ckpt)
            assert info["epoch"] == hist.attrs["best_epoch"]
            rep = detailed_report(model, val_loader, "cpu") if preset == "P" else None
        fig = plot_history(hist)
        assert len(fig.axes) >= 6
    try:
        fit(model, loader, val_loader, opt, "cpu", 1, monitor="val_foo")
        raise AssertionError("monitor tidak valid seharusnya ditolak")
    except ValueError:
        pass
    print("[smoke] fit / checkpoint / laporan / plot: OK")
    print("[smoke] SEMUA UJI LULUS")


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    _smoke_test()
