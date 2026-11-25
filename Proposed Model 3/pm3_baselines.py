"""Tabular-only baselines (B1) and Clean-28 vs Leaky-53 proof (Target #3) on identical 5 folds."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    recall_score,
    roc_auc_score,
)

from pm3_data import ClinicalPipeline, LeakyClinicalPipeline
from pm3_train import calibrate_thresholds_nested_oof, compute_metrics, save_json


def _make_classifiers(seed: int = 42):
    return {
        "LogReg_balanced": lambda: LogisticRegression(max_iter=1500, class_weight="balanced", C=0.5, random_state=seed),
        "RF_balanced": lambda: RandomForestClassifier(
            n_estimators=300, max_depth=6, class_weight="balanced", random_state=seed
        ),
        "HistGB_balanced": lambda: HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.05, max_iter=150, class_weight="balanced", random_state=seed
        ),
    }


def run_tabular_baselines(
    cohort: pd.DataFrame,
    folds: List[dict],
    feature_mode: str = "clean_28",
    output_dir: str | Path | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Evaluate 3-class, Hierarchical, Binary Diagnosis, and Binary Severity tabular baselines."""
    if feature_mode not in ("clean_28", "leaky_53"):
        raise ValueError("feature_mode harus 'clean_28' atau 'leaky_53'.")
    pipe_cls = ClinicalPipeline if feature_mode == "clean_28" else LeakyClinicalPipeline

    fold_data = []
    for f in folds:
        tr_df = cohort.loc[f["train"]]
        va_df = cohort.loc[f["val"]]
        pipe = pipe_cls().fit(tr_df)
        fold_data.append(
            {
                "fold": f["fold"],
                "pids_va": va_df.index.to_numpy(int),
                "X_tr": pipe.transform(tr_df),
                "y_tr": tr_df["label"].to_numpy(int),
                "X_va": pipe.transform(va_df),
                "y_va": va_df["label"].to_numpy(int),
            }
        )

    rows = []
    clfs = _make_classifiers(seed=seed)

    # 1) Direct 3-class classifiers
    for name, factory in clfs.items():
        y_all, p_all, pred_all = [], [], []
        for fd in fold_data:
            clf = factory().fit(fd["X_tr"], fd["y_tr"])
            y_all.append(fd["y_va"])
            p_all.append(clf.predict_proba(fd["X_va"]))
            pred_all.append(clf.predict(fd["X_va"]))
        y_all = np.concatenate(y_all)
        p_all = np.concatenate(p_all)
        pred_all = np.concatenate(pred_all)
        m = compute_metrics(y_all, p_all, pred_all, task_mode="softmax")
        rows.append(
            {
                "feature_mode": feature_mode,
                "task": "3class_direct",
                "model": name,
                "accuracy": m["accuracy"],
                "macro_f1": m["macro_f1"],
                "auroc": m["multiclass_auroc"],
                "aupr": m["macro_aupr"],
                "recall_no": m["sensitivity_class_0"],
                "recall_unc": m["sensitivity_class_1"],
                "recall_comp": m["sensitivity_class_2"],
            }
        )

    # 2) Hierarchical 2-stage binary classifiers
    for name, factory in clfs.items():
        oof_rows = []
        for fd in fold_data:
            c_diag = factory().fit(fd["X_tr"], (fd["y_tr"] > 0).astype(int))
            m_app = fd["y_tr"] > 0
            c_comp = factory().fit(fd["X_tr"][m_app], (fd["y_tr"][m_app] == 2).astype(int))
            pd_ = c_diag.predict_proba(fd["X_va"])[:, 1]
            qc_ = c_comp.predict_proba(fd["X_va"])[:, 1]
            for pid, yt, p, q in zip(fd["pids_va"], fd["y_va"], pd_, qc_):
                oof_rows.append(
                    {
                        "patient_id": int(pid),
                        "fold": fd["fold"],
                        "label": int(yt),
                        "p_diag": float(p),
                        "q_comp": float(q),
                        "p_0": float(1.0 - p),
                        "p_1": float(p * (1.0 - q)),
                        "p_2": float(p * q),
                        "pred": int(0 if p < 0.5 else (1 if q < 0.5 else 2)),
                    }
                )
        oof_df = pd.DataFrame(oof_rows)
        oof_df, m_cal = calibrate_thresholds_nested_oof(oof_df, "hierarchical")
        m_fix = compute_metrics(
            oof_df["label"].to_numpy(),
            oof_df[["p_0", "p_1", "p_2"]].to_numpy(),
            oof_df["pred"].to_numpy(),
            "hierarchical",
        )
        rows.append(
            {
                "feature_mode": feature_mode,
                "task": "3class_hierarchical",
                "model": f"Hier_{name}",
                "accuracy": m_cal["accuracy"],
                "macro_f1": m_cal["macro_f1"],
                "macro_f1_fixed05": m_fix["macro_f1"],
                "auroc": m_cal["multiclass_auroc"],
                "aupr": m_cal["macro_aupr"],
                "recall_no": m_cal["sensitivity_class_0"],
                "recall_unc": m_cal["sensitivity_class_1"],
                "recall_comp": m_cal["sensitivity_class_2"],
            }
        )

    # 3) Binary Diagnosis (Appendicitis vs No) & Binary Severity (Complicated vs All)
    for task_label, y_fn in (("binary_diag", lambda y: (y > 0).astype(int)), ("binary_sev", lambda y: (y == 2).astype(int))):
        for name, factory in clfs.items():
            y_all, p_all, pred_all = [], [], []
            for fd in fold_data:
                clf = factory().fit(fd["X_tr"], y_fn(fd["y_tr"]))
                prob = clf.predict_proba(fd["X_va"])[:, 1]
                y_all.append(y_fn(fd["y_va"]))
                p_all.append(prob)
                pred_all.append((prob >= 0.5).astype(int))
            y_all = np.concatenate(y_all)
            p_all = np.concatenate(p_all)
            pred_all = np.concatenate(pred_all)
            rows.append(
                {
                    "feature_mode": feature_mode,
                    "task": task_label,
                    "model": name,
                    "accuracy": float(accuracy_score(y_all, pred_all)),
                    "macro_f1": float(f1_score(y_all, pred_all, average="macro", zero_division=0)),
                    "f1_pos": float(f1_score(y_all, pred_all, pos_label=1, zero_division=0)),
                    "auroc": float(roc_auc_score(y_all, p_all)),
                    "aupr": float(average_precision_score(y_all, p_all)),
                    "sensitivity": float(recall_score(y_all, pred_all, pos_label=1, zero_division=0)),
                }
            )

    res = pd.DataFrame(rows)
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        res.to_csv(out / f"tabular_baselines_{feature_mode}.csv", index=False)
    return res


def run_leakage_comparison(
    cohort: pd.DataFrame, folds: List[dict], output_dir: str | Path | None = None, seed: int = 42
) -> pd.DataFrame:
    """Side-by-side comparison of Clean 28 ED features vs Leaky 53 features (Target #3 proof)."""
    clean_df = run_tabular_baselines(cohort, folds, feature_mode="clean_28", output_dir=output_dir, seed=seed)
    leaky_df = run_tabular_baselines(cohort, folds, feature_mode="leaky_53", output_dir=output_dir, seed=seed)
    merged = leaky_df.merge(
        clean_df,
        on=["task", "model"],
        suffixes=("_leaky53", "_clean28"),
    )
    merged["delta_auroc_leakage"] = merged["auroc_leaky53"] - merged["auroc_clean28"]
    merged["delta_macro_f1_leakage"] = merged["macro_f1_leaky53"] - merged["macro_f1_clean28"]
    if output_dir is not None:
        out = Path(output_dir)
        merged.to_csv(out / "leakage_vs_clean_comparison.csv", index=False)
        save_json(out / "leakage_summary.json", merged.to_dict(orient="records"))
    return merged
