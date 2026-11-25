"""Isolated preprocessing, cohort loading, curated 8 concepts, and USG bag pipeline for Proposed Model 3."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.impute import KNNImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder, RobustScaler, StandardScaler
from torch.utils.data import Dataset

_RAM_IMAGE_CACHE: Dict[Path, Tuple[np.ndarray, np.ndarray]] = {}


def _load_cached_npz(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    hit = _RAM_IMAGE_CACHE.get(path)
    if hit is not None:
        return hit
    with np.load(path) as arr:
        pair = (arr["image"].copy(), arr["mask"].copy())
    _RAM_IMAGE_CACHE[path] = pair
    return pair

ID_COL = "US_Number"
CLASS_NAMES = ("No Appendicitis", "Uncomplicated", "Complicated")

NUMERIC = [
    "Age", "Height", "Weight", "Body_Temperature", "WBC_Count",
    "Neutrophil_Percentage", "RBC_Count", "Hemoglobin", "RDW",
    "Thrombocyte_Count", "CRP", "BMI",
]
URINE = ["Ketones_in_Urine", "RBC_in_Urine", "WBC_in_Urine"]
BINARY = {
    "Sex": {"female": 0, "male": 1},
    **{
        c: {"no": 0, "yes": 1}
        for c in (
            "Migratory_Pain",
            "Lower_Right_Abd_Pain",
            "Contralateral_Rebound_Tenderness",
            "Coughing_Pain",
            "Nausea",
            "Loss_of_Appetite",
            "Dysuria",
            "Psoas_Sign",
            "Ipsilateral_Rebound_Tenderness",
            "Neutrophilia",
        )
    },
}
FEATURES = [
    "Age", "Sex", "Height", "Weight", "Body_Temperature", "Migratory_Pain",
    "Lower_Right_Abd_Pain", "Contralateral_Rebound_Tenderness", "Coughing_Pain",
    "Nausea", "Loss_of_Appetite", "Dysuria", "Stool", "Peritonitis",
    "Psoas_Sign", "Ipsilateral_Rebound_Tenderness", "WBC_Count",
    "Neutrophil_Percentage", "Neutrophilia", "RBC_Count", "Hemoglobin",
    "RDW", "Thrombocyte_Count", "CRP", *URINE, "BMI",
]
PLAUSIBLE = {
    "Age": (0, 18),
    "Height": (40, 200),
    "Weight": (1.5, 150),
    "BMI": (10, 60),
    "Body_Temperature": (33, 43),
    "WBC_Count": (0.1, 100),
    "Neutrophil_Percentage": (0, 100),
    "RBC_Count": (1.5, 8),
    "Hemoglobin": (3, 25),
    "RDW": (8, 30),
    "Thrombocyte_Count": (5, 1500),
    "CRP": (0, 1000),
}
GUIDE = [
    "Age", "Sex", "Body_Temperature", "Lower_Right_Abd_Pain",
    "Contralateral_Rebound_Tenderness", "Peritonitis", "WBC_Count", "CRP",
]
STOOL = {"normal": 0, "diarrhea": 1, "constipation": 2, "constipation, diarrhea": 3}

# Curated 8 sonographic concepts for Proposed Model 3
CONCEPTS_PM3 = (
    "Appendix_on_US",
    "Appendix_Diameter",
    "Target_Sign",
    "Pathological_Lymph_Nodes",
    "Appendix_Wall_Layers",
    "Surrounding_Tissue_Reaction",
    "Free_Fluids",
    "Severe_Complication_Signs",
)
SEVERE_CONCEPT_COLS = (
    "Perforation",
    "Appendicular_Abscess",
    "Appendicolith",
    "Ileus",
    "Conglomerate_of_Bowel_Loops",
)

BAG_MODES = ("all", "appendix_single", "appendix_plus_1_2")


def load_cohort(path: str | Path) -> pd.DataFrame:
    """Load clean 578-patient Regensburg pediatric appendicitis cohort."""
    raw = pd.read_csv(path)
    if len(raw) != 579 or raw[ID_COL].duplicated().any():
        raise ValueError("Sumber harus berisi tepat 579 pasien unik.")
    df = raw.loc[raw[ID_COL].ne(557)].copy()
    valid = df.Diagnosis.isin(["no appendicitis", "appendicitis"])
    valid &= df.Diagnosis.eq("no appendicitis") | df.Severity.isin(["uncomplicated", "complicated"])
    valid &= ~(df.Diagnosis.eq("no appendicitis") & df.Severity.eq("complicated"))
    if not valid.all():
        raise ValueError("Ground truth tidak valid ditemukan pada kohort.")
    df[ID_COL] = df[ID_COL].astype(int)
    df["label"] = np.select(
        [df.Diagnosis.eq("no appendicitis"), df.Severity.eq("uncomplicated")],
        [0, 1],
        2,
    )
    df = df.sort_values(ID_COL).set_index(ID_COL, drop=False)
    if len(df) != 578 or df.label.value_counts().sort_index().tolist() != [196, 286, 96]:
        raise AssertionError("Distribusi label kohort tidak sesuai 578 pasien [196, 286, 96].")
    return df


def make_stratified_folds(cohort: pd.DataFrame, seed: int = 1, n_splits: int = 5) -> List[dict]:
    """Patient-level stratified K-fold splits with strict zero overlap."""
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = [
        {
            "fold": k,
            "train": cohort.iloc[tr][ID_COL].astype(int).tolist(),
            "val": cohort.iloc[va][ID_COL].astype(int).tolist(),
        }
        for k, (tr, va) in enumerate(splitter.split(cohort, cohort.label), 1)
    ]
    all_ids = set(cohort[ID_COL].astype(int))
    assert sorted(p for f in folds for p in f["val"]) == sorted(all_ids)
    assert all(not set(f["train"]) & set(f["val"]) and (set(f["train"]) | set(f["val"])) == all_ids for f in folds)
    return folds


class ClinicalPipeline:
    """Clean 28-feature pre-imaging ED clinical preprocessing fitted strictly on train fold."""

    def _encode(self, df: pd.DataFrame) -> pd.DataFrame:
        x = pd.DataFrame(index=df.index)
        for c in FEATURES:
            if c in NUMERIC:
                x[c] = pd.to_numeric(df[c], errors="raise").astype(float)
            else:
                s = df[c].astype("string").str.strip().str.lower().replace("not_examined", pd.NA)
                mapping = (
                    BINARY[c]
                    if c in BINARY
                    else STOOL
                    if c == "Stool"
                    else {"no": 0, "local": 1, "generalized": 2}
                    if c == "Peritonitis"
                    else {"no": 0, "+": 1, "++": 2, "+++": 3}
                )
                bad = set(s.dropna()) - set(mapping)
                if bad:
                    raise ValueError(f"Kategori tidak dikenal pada {c}: {bad}")
                x[c] = s.map(mapping).astype(float)

        calc = x.Weight / (x.Height / 100) ** 2
        x.loc[calc.notna() & ~calc.between(*PLAUSIBLE["BMI"]), "Weight"] = np.nan
        x.loc[(x.Age < 2) & (x.Height > 100), "Age"] = np.nan
        for c, (lo, hi) in PLAUSIBLE.items():
            if c != "BMI":
                x.loc[x[c].notna() & ~x[c].between(lo, hi), c] = np.nan
        x["BMI"] = x.Weight / (x.Height / 100) ** 2
        pct = x.Neutrophil_Percentage.notna()
        x.loc[pct, "Neutrophilia"] = (x.loc[pct, "Neutrophil_Percentage"] >= 75).astype(float)
        # MNAR urinalysis: missing encoded explicitly as -1.0
        x[URINE] = x[URINE].fillna(-1.0)
        return x

    def fit(self, train: pd.DataFrame) -> "ClinicalPipeline":
        self.train_ids_ = train[ID_COL].astype(int).tolist()
        self.donors_ = self._encode(train)
        self.distance_std_ = self.donors_[NUMERIC].std(ddof=0).replace(0, 1).fillna(1)
        self.missing_train_ = self.donors_.isna().sum().to_dict()
        filled = self._impute(self.donors_)
        filled["BMI"] = filled.Weight / (filled.Height / 100) ** 2
        filled[["WBC_Count", "CRP"]] = np.log1p(filled[["WBC_Count", "CRP"]])
        self.scaler_ = StandardScaler().fit(filled[NUMERIC])
        self.columns_ = filled.columns.tolist()
        return self

    def _impute(self, x: pd.DataFrame) -> pd.DataFrame:
        out = x.copy()
        for c in FEATURES:
            if c == "BMI" or not x[c].isna().any():
                continue
            donors = self.donors_.loc[self.donors_[c].notna()]
            if donors.empty:
                raise ValueError(f"Tidak ada donor train untuk {c}")
            guides = [g for g in (["Age", "Sex"] if c in ["Height", "Weight"] else GUIDE) if g != c]
            for idx in x.index[x[c].isna()]:
                q = x.loc[idx, guides].to_numpy(float)
                ref = donors[guides].to_numpy(float)
                observed = np.isfinite(ref) & np.isfinite(q)[None, :]
                delta = np.zeros_like(ref)
                for j, g in enumerate(guides):
                    if g in NUMERIC:
                        delta[:, j] = (ref[:, j] - q[j]) / self.distance_std_[g]
                    elif g == "Peritonitis":
                        delta[:, j] = (ref[:, j] - q[j]) / 2.0
                    else:
                        delta[:, j] = (ref[:, j] != q[j]).astype(float)
                n_obs = observed.sum(1)
                dist = np.sqrt((np.where(observed, delta, 0) ** 2).sum(1) * len(guides) / np.maximum(n_obs, 1))
                dist[n_obs == 0] = np.inf
                keep = np.flatnonzero(np.isfinite(dist))
                if len(keep) == 0:
                    raise ValueError(f"Tidak ada pemandu teramati untuk pasien {idx}, fitur {c}.")
                nearest = keep[np.argsort(dist[keep], kind="stable")[:5]]
                weights = 1.0 / (dist[nearest] + 1e-5)
                values = donors[c].to_numpy()[nearest]
                if c in NUMERIC:
                    out.at[idx, c] = np.average(values, weights=weights)
                elif c in BINARY:
                    out.at[idx, c] = float(np.average(values, weights=weights) >= 0.5)
                else:
                    levels = np.unique(values)
                    votes = [weights[values == level].sum() for level in levels]
                    out.at[idx, c] = levels[int(np.argmax(votes))]
        absent = x.Neutrophil_Percentage.isna()
        flag = x.Neutrophilia
        out.loc[absent & flag.eq(0), "Neutrophil_Percentage"] = out.loc[
            absent & flag.eq(0), "Neutrophil_Percentage"
        ].clip(upper=np.nextafter(75.0, 0))
        out.loc[absent & flag.eq(1), "Neutrophil_Percentage"] = out.loc[
            absent & flag.eq(1), "Neutrophil_Percentage"
        ].clip(lower=75.0)
        out["Neutrophilia"] = (out.Neutrophil_Percentage >= 75).astype(float)
        out["BMI"] = out.Weight / (out.Height / 100) ** 2
        return out

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "scaler_"):
            raise RuntimeError("Panggil fit() pada train terlebih dahulu.")
        x = self._impute(self._encode(df))
        x[["WBC_Count", "CRP"]] = np.log1p(x[["WBC_Count", "CRP"]])
        x[NUMERIC] = self.scaler_.transform(x[NUMERIC])
        arr = x[self.columns_].to_numpy(np.float32)
        if not np.isfinite(arr).all():
            raise ValueError("Ditemukan nilai non-finite pada hasil transform klinis.")
        return arr


class LeakyClinicalPipeline:
    """Reference 53-feature leaky tabular pipeline (LOS, scores, US findings, missing-category shortcut)
    used strictly for Target #3 leakage-vs-clean comparison on identical patient folds.
    """

    EXCLUDE = {ID_COL, "Diagnosis", "Severity", "Management", "Diagnosis_Presumptive", "label"}

    def fit(self, train: pd.DataFrame) -> "LeakyClinicalPipeline":
        self.feature_cols_ = [c for c in train.columns if c not in self.EXCLUDE]
        self.num_cols_ = [c for c in self.feature_cols_ if pd.api.types.is_numeric_dtype(train[c])]
        self.cat_cols_ = [c for c in self.feature_cols_ if c not in self.num_cols_]

        miss_rate = train[self.feature_cols_].isna().mean()
        self.knn_num_cols_ = [c for c in self.num_cols_ if miss_rate[c] <= 0.50]
        self.med_num_cols_ = [c for c in self.num_cols_ if miss_rate[c] > 0.50]
        self.knn_cat_cols_ = [c for c in self.cat_cols_ if miss_rate[c] <= 0.50]
        self.miss_cat_cols_ = [c for c in self.cat_cols_ if miss_rate[c] > 0.50]

        # Numeric >50% missing -> train median
        self.medians_ = {c: float(pd.to_numeric(train[c], errors="coerce").median()) for c in self.med_num_cols_}

        # Categorical >50% missing -> explicit "missing" category; <=50% missing -> keep NaN for KNN
        tr_cat_knn = train[self.knn_cat_cols_].astype("object").where(train[self.knn_cat_cols_].notna(), np.nan)
        self.knn_cat_enc_ = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=np.nan
        )
        tr_cat_knn_codes = self.knn_cat_enc_.fit_transform(tr_cat_knn) if self.knn_cat_cols_ else np.empty((len(train), 0))

        tr_num_knn = train[self.knn_num_cols_].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        knn_mat_tr = np.hstack([tr_num_knn, tr_cat_knn_codes])
        self.knn_scaler_ = RobustScaler().fit(knn_mat_tr)
        self.knn_imputer_ = KNNImputer(n_neighbors=5).fit(self.knn_scaler_.transform(knn_mat_tr))

        # Fit final RobustScaler on all numeric columns and OrdinalEncoder on high-missing categoricals
        tr_imputed_knn = self.knn_scaler_.inverse_transform(self.knn_imputer_.transform(self.knn_scaler_.transform(knn_mat_tr)))
        n_knn_num = len(self.knn_num_cols_)
        num_df = pd.DataFrame(index=train.index)
        for i, c in enumerate(self.knn_num_cols_):
            raw_v = tr_num_knn[:, i]
            num_df[c] = np.where(np.isfinite(raw_v), raw_v, tr_imputed_knn[:, i])
        for c in self.med_num_cols_:
            num_df[c] = pd.to_numeric(train[c], errors="coerce").fillna(self.medians_[c])
        self.final_num_scaler_ = RobustScaler().fit(num_df[self.num_cols_])

        tr_miss_cat = train[self.miss_cat_cols_].fillna("missing").astype(str)
        self.miss_cat_enc_ = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        if self.miss_cat_cols_:
            self.miss_cat_enc_.fit(tr_miss_cat)
        self.columns_ = list(self.feature_cols_)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        cat_knn = df[self.knn_cat_cols_].astype("object").where(df[self.knn_cat_cols_].notna(), np.nan)
        cat_knn_codes = self.knn_cat_enc_.transform(cat_knn) if self.knn_cat_cols_ else np.empty((len(df), 0))
        num_knn = df[self.knn_num_cols_].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        knn_mat = np.hstack([num_knn, cat_knn_codes])
        imp_mat = self.knn_scaler_.inverse_transform(self.knn_imputer_.transform(self.knn_scaler_.transform(knn_mat)))

        n_knn_num = len(self.knn_num_cols_)
        out = pd.DataFrame(index=df.index)
        for i, c in enumerate(self.knn_num_cols_):
            raw_v = num_knn[:, i]
            out[c] = np.where(np.isfinite(raw_v), raw_v, imp_mat[:, i])
        for c in self.med_num_cols_:
            out[c] = pd.to_numeric(df[c], errors="coerce").fillna(self.medians_[c])
        out[self.num_cols_] = self.final_num_scaler_.transform(out[self.num_cols_])

        for j, c in enumerate(self.knn_cat_cols_):
            raw_c = cat_knn_codes[:, j]
            filled_c = np.where(np.isfinite(raw_c), raw_c, np.round(imp_mat[:, n_knn_num + j]))
            n_cats = len(self.knn_cat_enc_.categories_[j])
            out[c] = np.clip(filled_c, -1, n_cats - 1)

        if self.miss_cat_cols_:
            miss_cat = df[self.miss_cat_cols_].fillna("missing").astype(str)
            miss_codes = self.miss_cat_enc_.transform(miss_cat)
            for j, c in enumerate(self.miss_cat_cols_):
                out[c] = miss_codes[:, j]

        arr = out[self.columns_].to_numpy(np.float32)
        if not np.isfinite(arr).all():
            raise ValueError("Ditemukan nilai non-finite pada hasil transform LeakyClinicalPipeline.")
        return arr


def encode_pm3_concepts(
    df: pd.DataFrame, unrecorded_smoothing: float = 0.05
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode 8 curated sonographic concepts.

    Returns:
        concepts_target: float32 [N, 8] with optional smoothing applied ONLY to unrecorded cells
                         of presence/absence concepts (observed "no" stays 0.0, observed "yes" stays 1.0).
        concepts_binary: float32 [N, 8] exact {0.0, 1.0} binary ground truth for evaluation.
        concept_mask:    float32 [N, 8] validity mask (1=include in loss, 0=exclude).
    """
    if not 0.0 <= unrecorded_smoothing < 0.5:
        raise ValueError("unrecorded_smoothing harus berada di [0.0, 0.5).")

    visible = df["Appendix_on_US"].eq("yes").to_numpy()
    targets, binaries, masks = [], [], []

    for name in CONCEPTS_PM3:
        if name == "Appendix_on_US":
            s = df[name]
            pos = s.eq("yes").to_numpy(dtype=np.float32)
            obs = s.notna().to_numpy(dtype=np.float32)
            tgt = pos.copy()
            mask = obs
        elif name == "Appendix_Diameter":
            d = pd.to_numeric(df[name], errors="coerce")
            pos = d.ge(6.0).to_numpy(dtype=np.float32)
            tgt = pos.copy()
            mask = (visible & d.notna().to_numpy()).astype(np.float32)
        elif name == "Appendix_Wall_Layers":
            s = df[name]
            pos = s.isin(["raised", "partially raised", "upset"]).to_numpy(dtype=np.float32)
            tgt = pos.copy()
            mask = (visible & s.notna().to_numpy()).astype(np.float32)
        elif name == "Severe_Complication_Signs":
            sub = df[list(SEVERE_CONCEPT_COLS)]
            pos_bool = (
                sub["Perforation"].isin(["yes", "suspected", "not excluded"])
                | sub["Appendicular_Abscess"].isin(["yes", "suspected"])
                | sub["Appendicolith"].isin(["yes", "suspected"])
                | sub["Ileus"].eq("yes")
                | sub["Conglomerate_of_Bowel_Loops"].eq("yes")
            )
            any_obs = sub.notna().any(axis=1)
            pos = pos_bool.to_numpy(dtype=np.float32)
            tgt = np.where((~pos_bool) & (~any_obs), unrecorded_smoothing, pos).astype(np.float32)
            mask = np.ones(len(df), dtype=np.float32)
        else:
            # Target_Sign, Pathological_Lymph_Nodes, Surrounding_Tissue_Reaction, Free_Fluids
            s = df[name]
            pos_bool = s.eq("yes")
            obs_bool = s.notna()
            pos = pos_bool.to_numpy(dtype=np.float32)
            tgt = np.where((~pos_bool) & (~obs_bool), unrecorded_smoothing, pos).astype(np.float32)
            mask = np.ones(len(df), dtype=np.float32)

        targets.append(tgt)
        binaries.append(pos)
        masks.append(mask)

    return np.stack(targets, axis=1), np.stack(binaries, axis=1), np.stack(masks, axis=1)


# ---------------------------------------------------------------------------
# Ultrasound image preprocessing, Telea inpainting, sector mask, & caching
# ---------------------------------------------------------------------------

@dataclass
class USConfig:
    image_width: int = 320
    image_height: int = 384
    overlay_sat: int = 40
    overlay_rel_sat: float = 0.35
    overlay_dilate: int = 2
    crop_threshold: int = 12
    crop_density_window: int = 31
    crop_density_frac: float = 0.55
    crop_min_side: int = 40
    crop_col_frac: float = 0.10
    crop_gap_rows: int = 4
    crop_text_block: int = 6
    clahe_clip_limit: float = 2.0
    clahe_grid: int = 8


def _cv():
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("Butuh opencv-python (pip install opencv-python)") from exc
    return cv2


def _overlay_mask(rgb: np.ndarray, cfg: USConfig, dilate: Optional[int] = None) -> np.ndarray:
    cv = _cv()
    a = rgb.astype(np.int16)
    mx, mn = a.max(2), a.min(2)
    m = ((mx - mn) > cfg.overlay_sat) & ((mx - mn) > cfg.overlay_rel_sat * np.maximum(mx, 1))
    m = m.astype(np.uint8)
    k = cfg.overlay_dilate if dilate is None else dilate
    return cv.dilate(m, np.ones((2 * k + 1, 2 * k + 1), np.uint8)) if k else m


def _is_line_row(row: np.ndarray, thr: int) -> bool:
    return (row > thr).mean() > 0.9 and row.std() < 25


def _is_ui_block(block: np.ndarray, thr: int) -> bool:
    b = block > thr
    f = b.mean()
    return f < 0.01 or (f < 0.2 and (block > 150).sum() / max(b.sum(), 1) > 0.25)


def detect_us_box(gray: np.ndarray, cfg: USConfig):
    cv = _cv()
    h, w = gray.shape
    thr, win = cfg.crop_threshold, cfg.crop_density_window
    b = (cv.GaussianBlur(gray, (5, 5), 0) > thr).astype(np.float32)
    dens = cv.boxFilter(b, -1, (win, win), normalize=True, borderType=cv.BORDER_CONSTANT)
    n, lab, st, _ = cv.connectedComponentsWithStats((dens > cfg.crop_density_frac).astype(np.uint8), 8)
    cand = [
        i for i in range(1, n)
        if st[i, cv.CC_STAT_WIDTH] >= cfg.crop_min_side and st[i, cv.CC_STAT_HEIGHT] >= cfg.crop_min_side
    ]
    if not cand:
        return (0, 0, w, h), False
    comp = (lab == max(cand, key=lambda i: st[i, cv.CC_STAT_AREA])).astype(np.uint8)
    cnts, _ = cv.findContours(comp, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    comp = np.zeros_like(comp)
    cv.drawContours(comp, cnts, -1, 1, thickness=cv.FILLED)
    comp = cv.dilate(comp, np.ones((win // 2, win // 2), np.uint8))
    ys, xs = np.nonzero(comp)
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1

    raw = cv.GaussianBlur(gray, (3, 3), 0) > thr
    col = raw[y0:y1, x0:x1].mean(0)
    while x1 - x0 > 50 and col[0] < cfg.crop_col_frac:
        x0 += 1
        col = col[1:]
    while x1 - x0 > 50 and col[-1] < cfg.crop_col_frac:
        x1 -= 1
        col = col[:-1]

    y_dense, y, empty = y1, y1, 0
    while y < h and empty < cfg.crop_gap_rows and not _is_line_row(gray[y, x0:x1], thr):
        empty = empty + 1 if _is_ui_block(gray[y : y + 1, x0:x1], thr) else 0
        y += 1
    y1 = max(y1, y - empty)
    for yy in range(max(y0, y_dense - win), y1):
        if _is_line_row(gray[yy, x0:x1], thr):
            y1 = yy
            break
    blk = cfg.crop_text_block
    while y1 - y0 > 50 and _is_ui_block(gray[y1 - blk : y1, x0:x1], thr):
        y1 -= 1
    while y1 - y0 > 50 and _is_ui_block(gray[y0 : y0 + blk, x0:x1], thr):
        y0 += 1
    return (x0, y0, x1, y1), (x1 - x0) * (y1 - y0) / (h * w) >= 0.10


def us_region_mask(crop: np.ndarray, cfg: USConfig) -> np.ndarray:
    cv = _cv()
    from scipy.ndimage import median_filter, minimum_filter

    b = (cv.GaussianBlur(crop, (5, 5), 0) > cfg.crop_threshold).astype(np.uint8)
    b = cv.morphologyEx(b, cv.MORPH_OPEN, np.ones((5, 5), np.uint8))
    h, w = b.shape
    has = b.any(0)
    top = np.where(has, b.argmax(0), h)
    top = minimum_filter(median_filter(top, size=15, mode="nearest"), size=31, mode="nearest")
    m = (np.arange(h)[:, None] >= top[None, :]).astype(np.uint8)
    if not m.any():
        m[:] = 1
    return m


def letterbox(img: np.ndarray, width: int, height: int, nearest: bool = False) -> np.ndarray:
    cv = _cv()
    ch, cw = img.shape
    s = min(width / cw, height / ch)
    rw, rh = max(1, round(cw * s)), max(1, round(ch * s))
    interp = cv.INTER_NEAREST if nearest else (cv.INTER_AREA if s < 1 else cv.INTER_LINEAR)
    canvas = np.zeros((height, width), dtype=img.dtype)
    top, left = (height - rh) // 2, (width - rw) // 2
    canvas[top : top + rh, left : left + rw] = cv.resize(img, (rw, rh), interpolation=interp)
    return canvas


def artifact_mask(rgb: np.ndarray, cfg: USConfig) -> np.ndarray:
    cv = _cv()
    gray = cv.cvtColor(rgb, cv.COLOR_RGB2GRAY)
    white = (gray >= 245).astype(np.uint8)
    n, labels, stats, _ = cv.connectedComponentsWithStats(white, 8)
    candidates = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        text = 3 <= area <= 180 and 2 <= w <= 18 and 4 <= h <= 22 and area / (w * h) < 0.85
        line = area >= 12 and (w >= 18 and h <= 3 or h >= 18 and w <= 3)
        if text or line:
            candidates.append((i, x, y, w, h, line))
    mask = _overlay_mask(rgb, cfg)
    for i, x, y, w, h, line in candidates:
        aligned = (
            sum(abs(y - yy) <= 4 and 0 < abs(x - xx) <= 40 for j, xx, yy, ww, hh, ll in candidates if j != i)
            >= 2
        )
        if line or aligned:
            mask[labels == i] = 1
    return cv.dilate(mask, np.ones((3, 3), np.uint8))


def preprocess_ultrasound(path: str | Path, cfg: USConfig, return_stages: bool = False):
    cv = _cv()
    with Image.open(path) as im:
        rgb = np.asarray(im.convert("RGB"))
    original = cv.cvtColor(rgb, cv.COLOR_RGB2GRAY)
    artifacts = artifact_mask(rgb, cfg)
    gray = cv.inpaint(original, artifacts, 3, cv.INPAINT_TELEA) if artifacts.any() else original.copy()
    (x0, y0, x1, y1), _ = detect_us_box(gray, cfg)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Crop kosong pada citra {path}")
    crop = gray[y0:y1, x0:x1]
    region = us_region_mask(crop, cfg)
    contrast = cv.createCLAHE(
        clipLimit=cfg.clahe_clip_limit,
        tileGridSize=(cfg.clahe_grid, cfg.clahe_grid),
    ).apply(crop)
    image = letterbox(contrast, cfg.image_width, cfg.image_height)
    mask = letterbox(region, cfg.image_width, cfg.image_height, nearest=True)
    if return_stages:
        return image, mask, {
            "original": original,
            "artifact_mask": artifacts * 255,
            "Telea": gray,
            "crop": crop,
            "CLAHE": contrast,
            "final": image,
            "sector_mask": mask * 255,
        }
    return image, mask


def build_image_cache(
    manifest: pd.DataFrame, image_root: str | Path, cfg: USConfig, cache_dir: str | Path
) -> Dict[str, Path]:
    """Build deterministic disk cache of preprocessed uint8 USG images + sector masks."""
    code_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    cfg_hash = hashlib.sha256(json.dumps(asdict(cfg), sort_keys=True).encode()).hexdigest()
    directory = Path(cache_dir) / f"{code_hash[:10]}_{cfg_hash[:10]}"
    directory.mkdir(parents=True, exist_ok=True)
    cache = {}
    for i, row in enumerate(manifest.itertuples(index=False), 1):
        src = Path(image_root) / row.path
        stat = src.stat()
        key = hashlib.sha256(f"{src.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode()).hexdigest()[:12]
        dest = directory / f"{row.image_id}_{key}.npz"
        if not dest.exists():
            image, mask = preprocess_ultrasound(src, cfg)
            np.savez_compressed(dest, image=image, mask=mask)
        cache[str(row.image_id)] = dest
        if i % 250 == 0:
            print(f"Preprocessing/cache PM3: {i}/{len(manifest)}", flush=True)
    return cache


# ---------------------------------------------------------------------------
# View / Bag Filtering & Anatomy-Preserving Augmentations
# ---------------------------------------------------------------------------

def is_appendix_label(label) -> bool:
    """True when filename label indicates an actual appendix view."""
    if pd.isna(label):
        return False
    s = str(label).lower()
    if any(neg in s for neg in ("not visible", "not vis", "no appendix", "not appendix", "possibily", "possibly", "like")):
        return False
    return bool(re.search(r"\bap", s))


def images_by_patient(manifest: pd.DataFrame, bag_mode: str = "all") -> Dict[int, List[str]]:
    """Build patient -> ordered image_id bags under one of three view-selection modes.

    Modes:
        - "all": All views sorted by (US_Number, view, panel, image_id).
        - "appendix_single": 1 appendix-labeled image per patient (fallback: first image if 0 appendix views).
        - "appendix_plus_1_2": 1st appendix-labeled image + up to 2 earliest other images (total 1..3 images).
    """
    if bag_mode not in BAG_MODES:
        raise ValueError(f"bag_mode harus salah satu dari {BAG_MODES}, bukan {bag_mode!r}.")
    ordered = manifest.sort_values([ID_COL, "view", "panel", "image_id"]).copy()
    ordered["_is_app"] = ordered["label"].apply(is_appendix_label)
    bags: Dict[int, List[str]] = {}
    for pid, grp in ordered.groupby(ID_COL):
        all_ids = grp["image_id"].astype(str).tolist()
        app_ids = grp.loc[grp["_is_app"], "image_id"].astype(str).tolist()
        if bag_mode == "all":
            sel = all_ids
        elif bag_mode == "appendix_single":
            sel = app_ids[:1] if app_ids else all_ids[:1]
        else:
            # appendix_plus_1_2: 1st appendix view + up to 2 earliest non-duplicate views
            if app_ids:
                first_app = app_ids[:1]
                others = [iid for iid in all_ids if iid not in first_app][:2]
                sel = first_app + others
            else:
                sel = all_ids[:3]
        bags[int(pid)] = sel
    return bags


def bag_mode_summary(manifest: pd.DataFrame) -> pd.DataFrame:
    """Summarize total images and per-patient bag sizes across all bag modes."""
    rows = []
    for mode in BAG_MODES:
        bags = images_by_patient(manifest, bag_mode=mode)
        sizes = [len(v) for v in bags.values()]
        rows.append({
            "bag_mode": mode,
            "n_patients": len(bags),
            "total_images": int(sum(sizes)),
            "mean_images": float(np.mean(sizes)),
            "median_images": float(np.median(sizes)),
            "min_images": int(min(sizes)),
            "max_images": int(max(sizes)),
        })
    return pd.DataFrame(rows)


class BagTransform:
    """Fast OpenCV anatomy-preserving USG transform (no vertical flip) with ImageNet normalization."""

    def __init__(self, train: bool = False):
        self.train = train
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        img_np = image
        msk_np = mask
        if self.train:
            cv = _cv()
            h, w = img_np.shape
            if float(np.random.random()) < 0.5:
                img_np = cv.flip(img_np, 1)
                msk_np = cv.flip(msk_np, 1)
            angle = float(np.random.uniform(-10.0, 10.0))
            scale = float(np.random.uniform(0.92, 1.08))
            tx = float(np.random.uniform(-0.05, 0.05)) * w
            ty = float(np.random.uniform(-0.05, 0.05)) * h
            mat = cv.getRotationMatrix2D((w * 0.5, h * 0.5), angle, scale)
            mat[0, 2] += tx
            mat[1, 2] += ty
            img_np = cv.warpAffine(img_np, mat, (w, h), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT, borderValue=0)
            msk_np = cv.warpAffine(msk_np, mat, (w, h), flags=cv.INTER_NEAREST, borderMode=cv.BORDER_CONSTANT, borderValue=0)

            x_f = img_np.astype(np.float32) * (1.0 / 255.0)
            gamma = float(np.random.uniform(0.85, 1.15))
            contrast = float(np.random.uniform(0.88, 1.12))
            brightness = float(np.random.uniform(-0.08, 0.08))
            np.clip(x_f, 1e-6, 1.0, out=x_f)
            np.power(x_f, gamma, out=x_f)
            x_f = (x_f - 0.5) * contrast + 0.5 + brightness
            if float(np.random.random()) < 0.30:
                x_f += np.random.normal(0.0, 0.015, size=x_f.shape).astype(np.float32)
            np.clip(x_f, 0.0, 1.0, out=x_f)
            x_f *= (msk_np > 0).astype(np.float32)
        else:
            x_f = img_np.astype(np.float32) * (1.0 / 255.0)

        x_norm = (x_f[None, :, :] - self.mean) / self.std
        m_t = torch.from_numpy((msk_np > 0).astype(np.float32))[None, :, :]
        return torch.from_numpy(np.ascontiguousarray(x_norm)), m_t


class PM3PatientBagDataset(Dataset):
    """Patient-level multimodal bag dataset for Proposed Model 3."""

    def __init__(
        self,
        ids: List[int],
        cohort: pd.DataFrame,
        clinical_pipeline,
        bags: Optional[Dict[int, List[str]]],
        cache: Optional[Dict[str, Path]],
        train: bool = False,
        include_images: bool = True,
        unrecorded_smoothing: float = 0.05,
    ):
        self.pids = [int(i) for i in ids]
        self.rows = cohort.loc[self.pids]
        self.tabular = clinical_pipeline.transform(self.rows)
        self.labels = self.rows.label.to_numpy(int)
        self.concepts_target, self.concepts_binary, self.concept_mask = encode_pm3_concepts(
            self.rows, unrecorded_smoothing=unrecorded_smoothing
        )
        self.bags = bags
        self.cache = cache
        self.include_images = include_images
        self.transform = BagTransform(train=train)

    def __len__(self) -> int:
        return len(self.pids)

    def __getitem__(self, i: int) -> dict:
        pid = self.pids[i]
        out = {
            "patient_id": pid,
            "tabular": torch.from_numpy(self.tabular[i]),
            "label": int(self.labels[i]),
            "concepts_target": torch.from_numpy(self.concepts_target[i]),
            "concepts_binary": torch.from_numpy(self.concepts_binary[i]),
            "concept_mask": torch.from_numpy(self.concept_mask[i]),
        }
        if self.include_images:
            if self.bags is None or self.cache is None:
                raise ValueError("bags dan cache wajib disediakan saat include_images=True.")
            pairs = []
            iids = self.bags[pid]
            if not iids:
                raise ValueError(f"Pasien {pid} tidak memiliki citra pada bag.")
            for iid in iids:
                im_arr, mk_arr = _load_cached_npz(self.cache[iid])
                pairs.append(self.transform(im_arr, mk_arr))
            out["image_ids"] = iids
            out["images"] = torch.stack([p[0] for p in pairs])
            out["pixel_masks"] = torch.stack([p[1] for p in pairs])
        else:
            out["image_ids"] = []
            out["images"] = None
            out["pixel_masks"] = None
        return out


def collate_patients(items: List[dict]) -> List[dict]:
    """Keep patient bags unpadded as a list; each patient runs as 1 microbatch on GPU."""
    return items
