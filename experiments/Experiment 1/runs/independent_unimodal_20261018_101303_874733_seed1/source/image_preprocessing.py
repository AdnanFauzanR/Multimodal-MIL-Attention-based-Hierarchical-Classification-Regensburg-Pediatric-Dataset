#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
image_preprocessing.py
======================

Preprocessing citra USG untuk model multimodal MIL, mengikuti bagian
"Preprocessing (for ResNet-18 Input)" pada paper referensi
(Barthwal et al., *Multimodal Explainable AI Approach to Pediatric Appendicitis
Detection*, Egyptian Pediatric Association Gazette, 2026).

DAFTAR ISI
----------
1. Langkah preprocessing (sesuai paper)
2. Alur pemakaian
3. Keputusan yang tidak ditentukan paper
4. Isi file

-------------------------------------------------------------------------------
1. LANGKAH PREPROCESSING (sesuai paper)
-------------------------------------------------------------------------------

   Semua split (train, val, test) - deterministik:
     1. Grayscale             : citra dibaca sebagai 1 kanal (mode "L").
     2. Letterbox 224 x 224   : sisi terpanjang di-resize ke 224 dengan rasio aspek
                                dipertahankan, sisanya di-zero-padding (hitam), citra
                                di tengah. (Tidak ada distorsi/peregangan.)
     3. Skala [0, 1]          : nilai piksel dibagi 255.
     4. Duplikasi 3 kanal     : kanal grayscale disalin 3x agar cocok dengan CNN
                                ImageNet (tensor 3 x 224 x 224).
     5. Standardisasi         : (x - mean) / std, mean & std dihitung HANYA dari citra
                                train (anti kebocoran). Karena 3 kanal identik, mean/std
                                sama untuk ketiganya.

   Hanya TRAIN - augmentasi acak, tiap citra diacak sendiri-sendiri, diterapkan
   pada langkah 3 (ruang [0, 1], sebelum standardisasi):
     - rotasi          sampai +/- 10 derajat
     - geser (shift)   sampai 5% dari ukuran
     - zoom            +/- 10%  (skala 0.9 - 1.1)
     - brightness      ringan (faktor 1 +/- 0.1)
     - noise Gaussian  ringan (sigma 0.01 pada skala [0, 1])
     - TANPA flip horizontal (menjaga orientasi anatomi, sesuai paper)
     Area kosong akibat rotasi/geser/zoom diisi hitam (0), sama dengan padding.

-------------------------------------------------------------------------------
2. ALUR PEMAKAIAN
-------------------------------------------------------------------------------

   from pathlib import Path
   import pandas as pd
   from image_preprocessing import (ImagePreprocessConfig, compute_train_stats,
                                    save_stats, load_stats, ImageTransform,
                                    PatientBagDataset)
   from multimodal_mil_model import collate_bags

   cfg = ImagePreprocessConfig()                       # 224, augmentasi sesuai paper
   SPLIT_DIR = Path("dataset_split")                   # hasil sel split di notebook

   # (a) Statistik dari citra TRAIN saja, lalu simpan agar val/test/inference
   #     memakai angka yang sama (jangan dihitung ulang di val/test!)
   train_manifest = pd.read_csv(SPLIT_DIR / "train" / "image_manifest.csv")
   stats = compute_train_stats([SPLIT_DIR / p for p in train_manifest["path"]], cfg.image_size)
   save_stats(stats, "image_stats.json")               # nanti: stats = load_stats("image_stats.json")

   # (b) Transform: train memakai augmentasi, val/test tidak
   train_tf = ImageTransform(cfg, stats, train=True)
   eval_tf  = ImageTransform(cfg, stats, train=False)

   # (c) Dataset per split -> tiap item = satu pasien (bag), siap untuk collate_bags
   train_ds = PatientBagDataset(
       tabular=X_train, labels=y_train, patient_ids=train_imp["US_Number"],
       manifest=train_manifest, image_root=SPLIT_DIR, transform=train_tf)
   # val/test: manifest masing-masing split + transform=eval_tf

   Untuk satu citra saja (mis. inference): load_letterboxed(path, cfg.image_size)
   lalu eval_tf(...) -> tensor (3, 224, 224).

-------------------------------------------------------------------------------
3. KEPUTUSAN YANG TIDAK DITENTUKAN PAPER
-------------------------------------------------------------------------------

   * Besar noise (sigma 0.01) dan brightness (+/- 0.1): paper hanya menulis
     "low-level" dan "mild". Nilai dapat diubah di ImagePreprocessConfig.
   * Urutan augmentasi vs standardisasi: di sini augmentasi dulu (ruang [0, 1]),
     baru standardisasi.
   * Mean/std dihitung atas seluruh piksel citra train yang SUDAH di-letterbox,
     termasuk area padding hitam (karena itulah masukan sebenarnya ke langkah 5).
   * Paper menyebut "ROI-based preprocessing" untuk menonjolkan area anatomi, tanpa
     menjelaskan caranya. TIDAK diimplementasikan di sini. Catatan: citra mentah
     berisi tampilan layar mesin USG (teks, skala, ikon), jadi cropping area USG
     mungkin bermanfaat; ini keputusan terpisah.
   * Citra berwarna (penanda ukur cyan/oranye) menjadi abu-abu pada langkah 1,
     sehingga penanda itu tetap ada sebagai intensitas, bukan warna.

-------------------------------------------------------------------------------
4. ISI FILE
-------------------------------------------------------------------------------
   ImagePreprocessConfig -> letterbox / load_letterboxed -> ImageStats /
   compute_train_stats / save_stats / load_stats -> ImageTransform ->
   PatientBagDataset -> _smoke_test (jalankan: python image_preprocessing.py)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2

PathLike = Union[str, Path]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
NORMALIZATIONS = ("train_stats", "imagenet")


# =============================================================================
# KONFIGURASI
# =============================================================================

@dataclass
class ImagePreprocessConfig:
    """Pengaturan preprocessing. Default = nilai dari paper (jika disebut)."""

    image_size: int = 224
    """Sisi citra keluaran (224 x 224, sesuai paper)."""

    normalization: str = "train_stats"
    """"train_stats": standardisasi dengan mean/std citra train (paper).
    "imagenet": pakai mean/std ImageNet (alternatif untuk backbone pretrained)."""

    # --- Augmentasi (hanya dipakai bila ImageTransform(..., train=True)) -----
    rotation_degrees: float = 10.0
    """Rotasi acak dalam [-r, +r] derajat (paper: 10)."""
    translate_frac: float = 0.05
    """Geser acak maksimum sebagai fraksi ukuran citra (paper: 5%)."""
    scale_range: Tuple[float, float] = (0.9, 1.1)
    """Zoom acak (paper: +/- 10%)."""
    brightness: float = 0.1
    """Faktor brightness acak dalam [1 - b, 1 + b] ("mild"); 0 = nonaktif."""
    noise_std: float = 0.01
    """Sigma noise Gaussian pada skala [0, 1] ("low-level"); 0 = nonaktif."""

    def __post_init__(self) -> None:
        if self.normalization not in NORMALIZATIONS:
            raise ValueError(f"normalization='{self.normalization}' tidak dikenal. "
                             f"Pilihan: {list(NORMALIZATIONS)}")
        if self.image_size < 8:
            raise ValueError("image_size terlalu kecil")
        self.scale_range = tuple(self.scale_range)

    def to_dict(self) -> dict:
        return asdict(self)


# =============================================================================
# LANGKAH DETERMINISTIK: GRAYSCALE + LETTERBOX
# =============================================================================

def letterbox(img: Image.Image, size: int) -> Image.Image:
    """Resize sisi terpanjang ke `size` (rasio aspek tetap), tempel di tengah kanvas
    hitam `size x size` (zero-padding). Mengembalikan citra mode "L"."""
    img = img.convert("L")                                  # langkah 1: grayscale
    w, h = img.size
    scale = size / max(w, h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    img = img.resize((new_w, new_h), Image.BILINEAR)        # BILINEAR di PIL sudah antialias saat mengecil
    canvas = Image.new("L", (size, size), 0)
    canvas.paste(img, ((size - new_w) // 2, (size - new_h) // 2))
    return canvas


def load_letterboxed(path: PathLike, size: int = 224) -> torch.Tensor:
    """Baca file citra -> tensor uint8 (1, size, size) hasil grayscale + letterbox."""
    with Image.open(path) as im:
        canvas = letterbox(im, size)
    return torch.from_numpy(np.asarray(canvas, dtype=np.uint8).copy()).unsqueeze(0)


# =============================================================================
# STATISTIK TRAIN (mean / std)
# =============================================================================

@dataclass
class ImageStats:
    """Mean dan std per kanal (3 kanal) pada skala [0, 1]."""
    mean: Tuple[float, float, float]
    std: Tuple[float, float, float]


def compute_train_stats(train_paths: Sequence[PathLike], image_size: int = 224) -> ImageStats:
    """Hitung mean/std piksel dari citra TRAIN saja (setelah grayscale+letterbox, skala [0, 1]).

    Akumulasi dalam float64 per citra, jadi hemat memori. Karena kanal diduplikasi,
    mean/std satu angka dipakai untuk ketiga kanal.

    PENTING: panggil hanya dengan path citra TRAIN, lalu pakai hasilnya untuk
    val/test (jangan hitung ulang di sana).
    """
    if len(train_paths) == 0:
        raise ValueError("train_paths kosong")
    total = 0
    s1 = 0.0   # jumlah piksel
    s2 = 0.0   # jumlah kuadrat piksel
    for p in train_paths:
        x = load_letterboxed(p, image_size).double() / 255.0
        total += x.numel()
        s1 += x.sum().item()
        s2 += (x * x).sum().item()
    mean = s1 / total
    std = max((s2 / total - mean ** 2), 0.0) ** 0.5
    return ImageStats(mean=(mean,) * 3, std=(std,) * 3)


def save_stats(stats: ImageStats, path: PathLike) -> None:
    Path(path).write_text(json.dumps(asdict(stats), indent=2), encoding="utf-8")


def load_stats(path: PathLike) -> ImageStats:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return ImageStats(mean=tuple(d["mean"]), std=tuple(d["std"]))


# =============================================================================
# TRANSFORM: AUGMENTASI (TRAIN) + SKALA + 3 KANAL + STANDARDISASI
# =============================================================================

class AddGaussianNoise:
    """Tambah noise Gaussian N(0, std^2) pada tensor."""

    def __init__(self, std: float):
        self.std = std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.randn_like(x) * self.std


class ImageTransform:
    """Tensor uint8 (1, H, W) hasil load_letterboxed -> tensor float (3, H, W).

    train=True menambahkan augmentasi acak; train=False sepenuhnya deterministik.
    Instance dengan train=True sebaiknya HANYA dipakai pada dataset train.
    """

    def __init__(self, cfg: ImagePreprocessConfig, stats: Optional[ImageStats], train: bool):
        self.cfg, self.train = cfg, train

        if cfg.normalization == "imagenet":
            mean, std = IMAGENET_MEAN, IMAGENET_STD
        else:
            if stats is None:
                raise ValueError("normalization='train_stats' memerlukan `stats` "
                                 "(compute_train_stats / load_stats)")
            mean, std = stats.mean, stats.std
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

        self.affine = v2.RandomAffine(
            degrees=cfg.rotation_degrees,
            translate=(cfg.translate_frac, cfg.translate_frac),
            scale=cfg.scale_range,
            fill=0,                         # area kosong = hitam, sama dengan padding
        ) if train else None
        self.jitter = v2.ColorJitter(brightness=cfg.brightness) if (train and cfg.brightness > 0) else None
        self.noise = AddGaussianNoise(cfg.noise_std) if (train and cfg.noise_std > 0) else None

    def __call__(self, gray_u8: torch.Tensor) -> torch.Tensor:
        x = gray_u8.float() / 255.0                          # langkah 3: skala [0, 1]
        if self.train:
            x = self.affine(x)                               # rotasi + geser + zoom (tanpa flip)
            if self.jitter is not None:
                x = self.jitter(x)                           # brightness ringan
            if self.noise is not None:
                x = self.noise(x)                            # noise Gaussian ringan
            x = x.clamp(0.0, 1.0)
        x = x.expand(3, -1, -1)                              # langkah 4: duplikasi ke 3 kanal
        return (x - self.mean) / self.std                    # langkah 5: standardisasi


# =============================================================================
# DATASET: SATU PASIEN = SATU BAG
# =============================================================================

class PatientBagDataset(Dataset):
    """Dataset yang mengembalikan satu pasien (bag) per item.

    Item = dict yang cocok dengan `collate_bags` di multimodal_mil_model.py:
        "images"         : (n, 3, H, W) float  (n = jumlah citra pasien)
        "tabular"        : (F,) float
        "y_appendicitis" : int
        "y_complicated"  : int  (-1 bila bukan apendisitis)
        "patient_id"     : int  (US_Number; tidak dipakai collate_bags, berguna untuk analisis)

    Argumen (semua sejajar baris-per-baris untuk tabular, labels, patient_ids):
        tabular     : DataFrame/array (N, F) hasil One-Hot (X_train, dst.)
        labels      : DataFrame dengan kolom y_appendicitis, y_complicated
        patient_ids : US_Number tiap baris
        manifest    : DataFrame split ini (kolom US_Number, path, view, panel, image_id)
        image_root  : folder tempat kolom `path` di manifest berlaku (mis. dataset_split/)
        transform   : ImageTransform (train=True untuk train, False untuk val/test)
        cache       : True = simpan hasil grayscale+letterbox (uint8, kecil) di memori
                      agar epoch berikutnya tidak membaca disk; augmentasi tetap acak.
                      (Dengan DataLoader num_workers > 0, tiap worker punya cache sendiri.)

    Urutan citra dalam bag diurutkan tetap menurut (view, panel, image_id),
    penting bila agregasi memakai LSTM.
    """

    def __init__(self, tabular, labels: pd.DataFrame, patient_ids: Sequence[int],
                 manifest: pd.DataFrame, image_root: PathLike, transform: ImageTransform,
                 cache: bool = True):
        self.tabular = torch.as_tensor(np.asarray(tabular, dtype=np.float32))
        self.y_app = torch.as_tensor(np.asarray(labels["y_appendicitis"]), dtype=torch.long)
        self.y_comp = torch.as_tensor(np.asarray(labels["y_complicated"]), dtype=torch.long)
        self.patient_ids = [int(p) for p in patient_ids]
        if not (len(self.tabular) == len(self.y_app) == len(self.y_comp) == len(self.patient_ids)):
            raise ValueError("tabular, labels, dan patient_ids harus sama panjang")

        self.transform = transform
        self.image_size = transform.cfg.image_size
        self.cache: Optional[Dict[Path, torch.Tensor]] = {} if cache else None

        root = Path(image_root)
        ordered = manifest.sort_values(["view", "panel", "image_id"])
        by_patient = {int(k): [root / p for p in g["path"]] for k, g in ordered.groupby("US_Number")}
        missing = [pid for pid in self.patient_ids if pid not in by_patient]
        if missing:
            raise ValueError(f"{len(missing)} pasien tidak punya citra di manifest, mis. {missing[:5]}")
        self.bags: List[List[Path]] = [by_patient[pid] for pid in self.patient_ids]

    def __len__(self) -> int:
        return len(self.patient_ids)

    def _gray(self, path: Path) -> torch.Tensor:
        if self.cache is None:
            return load_letterboxed(path, self.image_size)
        if path not in self.cache:
            self.cache[path] = load_letterboxed(path, self.image_size)
        return self.cache[path]

    def __getitem__(self, idx: int) -> dict:
        images = torch.stack([self.transform(self._gray(p)) for p in self.bags[idx]])
        return {
            "images": images,
            "tabular": self.tabular[idx],
            "y_appendicitis": int(self.y_app[idx]),
            "y_complicated": int(self.y_comp[idx]),
            "patient_id": self.patient_ids[idx],
        }


# =============================================================================
# SMOKE TEST  (python image_preprocessing.py)
# =============================================================================

def _smoke_test() -> None:
    """Uji dengan citra sintetis (tidak bergantung pada dataset asli)."""
    import tempfile
    from torch.utils.data import DataLoader

    torch.manual_seed(0)
    cfg = ImagePreprocessConfig(image_size=64)

    # 1) letterbox: ukuran akhir, rasio aspek, padding hitam, posisi tengah
    wide = Image.new("L", (300, 100), 200)                  # landscape
    out = np.asarray(letterbox(wide, 64))
    assert out.shape == (64, 64)
    rows = np.where(out.max(axis=1) > 0)[0]
    assert abs((rows.max() - rows.min() + 1) - round(100 * 64 / 300)) <= 1       # tinggi = 100*64/300
    assert out[0].max() == 0 and out[-1].max() == 0                               # padding atas/bawah hitam
    assert abs((rows.min() + rows.max()) / 2 - 31.5) <= 1                         # di tengah
    tall = np.asarray(letterbox(Image.new("RGB", (100, 300), (255, 0, 0)), 64))   # RGB -> grayscale
    assert tall.shape == (64, 64) and tall[:, 0].max() == 0
    print("[smoke] letterbox: OK")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "train" / "images").mkdir(parents=True)

        # citra sintetis tak simetris: setengah kiri terang, setengah kanan gelap
        def make(name: str, w: int, h: int) -> str:
            arr = np.zeros((h, w), dtype=np.uint8)
            arr[:, : w // 2] = 220
            arr[:, w // 2:] = 30
            rel = f"train/images/{name}.png"
            Image.fromarray(arr).save(tmp / rel)
            return rel

        sizes = {1: [(120, 80)], 2: [(80, 120), (100, 100), (90, 60)], 3: [(150, 90), (70, 70)]}
        rows_m = []
        for pid, shp in sizes.items():
            for v, (w, h) in enumerate(shp, 1):
                rows_m.append({"US_Number": pid, "view": v, "panel": 1,
                               "image_id": f"S{pid:04d}_V{v:02d}_P01", "path": make(f"p{pid}v{v}", w, h)})
        manifest = pd.DataFrame(rows_m)

        # 2) statistik train: sesuai perhitungan manual, simpan/muat konsisten
        paths = [tmp / p for p in manifest["path"]]
        stats = compute_train_stats(paths, cfg.image_size)
        ref = torch.cat([load_letterboxed(p, cfg.image_size).double().flatten() / 255 for p in paths])
        assert abs(stats.mean[0] - ref.mean().item()) < 1e-9
        assert abs(stats.std[0] - ref.std(unbiased=False).item()) < 1e-9
        save_stats(stats, tmp / "stats.json")
        assert load_stats(tmp / "stats.json") == stats
        print(f"[smoke] statistik train: mean={stats.mean[0]:.4f} std={stats.std[0]:.4f}  OK")

        # 3) transform eval: bentuk, 3 kanal identik, deterministik, ternormalisasi (~mean 0, std 1)
        eval_tf = ImageTransform(cfg, stats, train=False)
        g = load_letterboxed(paths[0], cfg.image_size)
        a, b = eval_tf(g), eval_tf(g)
        assert a.shape == (3, 64, 64) and a.dtype == torch.float32
        assert torch.equal(a, b) and torch.equal(a[0], a[1]) and torch.equal(a[1], a[2])
        allx = torch.cat([eval_tf(load_letterboxed(p, cfg.image_size)).flatten() for p in paths])
        assert abs(allx.mean().item()) < 1e-4 and abs(allx.std(unbiased=False).item() - 1) < 1e-4
        print("[smoke] transform eval (deterministik, 3 kanal, ter-standardisasi): OK")

        # 4) transform train: acak, tetap di [0, 1] sebelum standardisasi, tanpa flip horizontal
        ident = ImageStats(mean=(0.0,) * 3, std=(1.0,) * 3)          # standardisasi netral -> lihat ruang [0, 1]
        train_tf = ImageTransform(cfg, ident, train=True)
        samples = torch.stack([train_tf(g) for _ in range(40)])
        assert not torch.equal(samples[0], samples[1])                # acak
        assert samples.min() >= 0.0 and samples.max() <= 1.0          # clamp
        xs = torch.arange(64, dtype=torch.float32)
        cx = (samples[:, 0].sum(dim=1) * xs).sum(dim=1) / samples[:, 0].sum(dim=(1, 2))   # pusat massa x
        cx0 = (eval_tf.__class__(cfg, ident, train=False)(g)[0].sum(dim=0) * xs).sum() / \
              eval_tf.__class__(cfg, ident, train=False)(g)[0].sum()
        assert (cx - cx0).abs().max() < 8, "pusat massa bergeser terlalu jauh (flip/geser berlebihan?)"
        assert (cx < 32).float().mean() >= 0.95 or cx0 < 32           # sisi terang tidak pindah ke kanan
        print("[smoke] transform train (acak, [0,1], tanpa flip): OK")

        # 5) dataset + collate_bags + model, ujung ke ujung
        import multimodal_mil_model as mm
        n_feat = 12
        ids = [1, 2, 3]
        labels = pd.DataFrame({"y_appendicitis": [0, 1, 1], "y_complicated": [-1, 0, 1]})
        ds = PatientBagDataset(torch.randn(3, n_feat), labels, ids, manifest, tmp, train_tf)
        item = ds[1]
        assert item["images"].shape == (3, 3, 64, 64) and item["tabular"].shape == (n_feat,)
        assert [len(ds[i]["images"]) for i in range(3)] == [1, 3, 2]
        assert len(ds.cache) > 0                                      # cache terisi
        batch = next(iter(DataLoader(ds, batch_size=3, collate_fn=mm.collate_bags)))
        model = mm.MultimodalMILModel(mm.ModelConfig.from_preset(
            "P", num_tabular_features=n_feat, pretrained=False, tabular_encoder="mlp")).train()
        out = model(batch["images"], batch["image_mask"], batch["tabular"])
        loss, _ = model.compute_loss(out, batch["y_appendicitis"], batch["y_complicated"])
        loss.backward()
        try:
            PatientBagDataset(torch.randn(1, n_feat), labels.iloc[:1], [99], manifest, tmp, train_tf)
            raise AssertionError("pasien tanpa citra seharusnya ditolak")
        except ValueError:
            pass
        print("[smoke] dataset -> collate_bags -> model: OK")

    print("[smoke] SEMUA UJI LULUS")


if __name__ == "__main__":
    _smoke_test()
