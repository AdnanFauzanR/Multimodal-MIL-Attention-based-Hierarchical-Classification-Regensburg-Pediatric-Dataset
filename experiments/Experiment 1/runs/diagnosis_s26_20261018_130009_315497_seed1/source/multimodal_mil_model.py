#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
multimodal_mil_model.py
=======================

Model multimodal (citra USG + data klinis) berbasis *Multiple Instance Learning*
(MIL) untuk klasifikasi hierarkis apendisitis pediatrik.

Satu **pasien** = satu **bag**: 1 data klinis + n >= 1 citra USG (n berbeda-beda).
Modul ini hanya berisi MODEL (+ fungsi loss & utilitas batching). Loop training dan
inference ditulis terpisah di notebook/skrip experiment.

DAFTAR ISI
----------
1.  Arsitektur
2.  Opsi yang bisa diatur (ModelConfig)
3.  Preset konfigurasi pembanding (B3-B7 dan P)
4.  Kontrak input / output
5.  Contoh pemakaian (termasuk GPU)
6.  Catatan penting
7.  Isi file (urutan kelas/fungsi)

-------------------------------------------------------------------------------
1. ARSITEKTUR (model usulan "P": dual attention + dual head)
-------------------------------------------------------------------------------

   Bag citra X1..Xn --> Backbone CNN (dipakai bersama) --> h1..hn   (fitur tiap citra)
                                                              |
                                  +---------------------------+---------------------+
                                  |                                                 |
                        Attention-MIL "diagnosis"                        Attention-MIL "komplikasi"
                        zD = sum_j aD_j * h_j                            zC = sum_j aC_j * h_j
                                  |                                                 |
   Data klinis c --> Encoder tabular --> e --+--> concat(zD, e)             concat(zC, e) <--+-- e
                     (TabNet / MLP)                   |                                 |
                                              Fusion MLP (diagnosis)           Fusion MLP (komplikasi)
                                                      |                                 |
                                              Head sigmoid                      Head sigmoid
                                              p = P(A | X, c)                   q = P(C | A, X, c)

   * A = apendisitis, C = komplikasi. Prediksi 3 kelas disusun hierarkis:
         P(no appendicitis)         = 1 - p
         P(uncomplicated)           = p * (1 - q)
         P(complicated)             = p * q
   * Level 2 (q) hanya dilatih pada pasien apendisitis (lihat compute_loss).

   Variasi lain hanyalah mengganti sebagian blok:

   +--------------------+-------------------------------------------------------+
   | Blok               | Pilihan                                               |
   +--------------------+-------------------------------------------------------+
   | backbone           | resnet18/34/50/101/152, efficientnet_b0..b7,          |
   |                    | efficientnet_v2_s/m/l, mobilenet_v2,                  |
   |                    | mobilenet_v3_small/large                              |
   | pretrained         | True (bobot ImageNet) / False (acak)                  |
   | aggregation        | "attention" | "lstm" | "mean"                         |
   | attention_fusion   | "shared" (1 attention) | "dual" (2 attention)         |
   |                    | -> hanya berlaku jika aggregation == "attention"      |
   | head               | "dual_head" (2 sigmoid) | "softmax" (3 kelas langsung)|
   | tabular_encoder    | "tabnet" | "mlp"                                      |
   | use_dropout        | False = matikan semua dropout                         |
   | dropout_rate       | satu angka untuk semua dropout (menimpa per komponen);|
   |                    | None = pakai image_/attention_/lstm_/mlp_/fusion_dropout|
   | OptimizerConfig    | name: "adam" | "adamw" | "sgd" | "rmsprop"; lr;        |
   |                    | weight_decay; backbone_lr_multiplier; momentum; ...   |
   +--------------------+-------------------------------------------------------+

   Cara tiap blok bekerja:

   * Attention-MIL (Ilse et al., 2018, versi *gated*): tiap citra diberi skor
         s_j = w^T ( tanh(V h_j) * sigmoid(U h_j) ),   a_j = softmax_j(s_j)
     lalu z = sum_j a_j h_j. Citra padding diberi skor -inf (bobot 0).
       - "shared": SATU attention menghasilkan SATU z yang dipakai kedua tugas.
       - "dual"  : DUA attention terpisah, zD untuk diagnosis dan zC untuk
                   komplikasi (tiap tugas boleh menyorot citra yang berbeda).
   * LSTM pooling: urutan h_1..h_n masuk LSTM, state akhir = z.
     (LSTM peka urutan, jadi urutan citra dalam bag mengikuti urutan dataset.)
   * Mean pooling: z = rata-rata h_j (baseline paling sederhana).
     Untuk "lstm" dan "mean" tidak ada attention terpisah: zD == zC (dipakai bersama).
   * head="dual_head": dua Fusion MLP + dua head sigmoid (diagram di atas).
   * head="softmax"  : SATU Fusion MLP + SATU head 3 kelas
                       [no appendicitis, uncomplicated, complicated].
       - Bila attention "dual", masukan fusion = concat(zD, zC, e)
         (kedua ringkasan citra dipakai, hanya headnya satu).
         Selain itu masukannya concat(z, e).
   * TabNet (Arik & Pfister, 2021): diimplementasikan murni PyTorch di file ini
     (tanpa library tambahan): 5 decision step, sparsemax feature mask, ghost
     batch norm. Keluarannya embedding e berdimensi `tabular_embed_dim`, plus
     loss sparsitas (opsional) dan mask fitur untuk interpretasi.

-------------------------------------------------------------------------------
2. OPSI (ModelConfig)  - lihat docstring tiap field di kelas ModelConfig
-------------------------------------------------------------------------------

-------------------------------------------------------------------------------
3. PRESET (sesuai tabel "Konfigurasi pembanding dan evaluasi")
-------------------------------------------------------------------------------

   B3 : mean pooling                         + dual head
   B4 : LSTM pooling                         + dual head
   B5 : shared attention                     + softmax
   B6 : shared attention                     + dual head
   B7 : dual attention                       + softmax
   P  : dual attention                       + dual head   (model usulan)

   B1 (klinis saja) dan B2 (citra saja) adalah ablasi satu modalitas dan BELUM
   dibuat; modul ini fokus pada model multimodal.

-------------------------------------------------------------------------------
4. KONTRAK INPUT / OUTPUT
-------------------------------------------------------------------------------

   forward(images, image_mask, tabular, return_attention=False)

     images     : (B, N, 3, H, W) float. N = jumlah citra terbanyak dalam batch;
                  bag yang lebih pendek DIPADDING di BELAKANG (isi bebas, mis. 0).
                  Preprocessing (letterbox 224, [0,1], standardisasi, augmentasi)
                  disediakan oleh image_preprocessing.py (PatientBagDataset).
     image_mask : (B, N) bool. True = citra asli, False = padding.
                  Tiap bag minimal 1 citra asli; padding harus di belakang.
     tabular    : (B, F) float, F = num_tabular_features. Pakai X_train/X_val/X_test
                  hasil preprocessing train (numerik terskala + kategori encoded).

   Keluaran: dict
     head="dual_head": "logit_diagnosis" (B,), "logit_complication" (B,)
     head="softmax"  : "logits" (B, 3)
     selalu          : "tab_sparsity_loss" (skalar, hanya TabNet; selain itu None)
     return_attention=True menambah: "attention_diagnosis", "attention_complication"
                  (B, N; bobot tiap citra, 0 untuk padding; None untuk LSTM) dan
                  "tabular_masks" (B, n_steps, F) bila TabNet.

   Label (sesuai pengkodean di notebook):
     y_appendicitis (B,) long : 0 / 1
     y_complicated  (B,) long : 0 / 1, atau -1 bila y_appendicitis == 0 (diabaikan)

-------------------------------------------------------------------------------
5. CONTOH PEMAKAIAN
-------------------------------------------------------------------------------

   import torch
   from torch.utils.data import DataLoader
   from multimodal_mil_model import (ModelConfig, MultimodalMILModel, OptimizerConfig,
                                     build_optimizer, collate_bags)

   device = "cuda" if torch.cuda.is_available() else "cpu"

   cfg = ModelConfig.from_preset("P", num_tabular_features=X_train.shape[1],
                                 backbone="resnet18", pretrained=True,
                                 tabular_encoder="tabnet")
   model = MultimodalMILModel(cfg).to(device)

   # Dataset-mu mengembalikan per pasien:
   #   {"images": Tensor(n, 3, H, W), "tabular": Tensor(F,),
   #    "y_appendicitis": int, "y_complicated": int}
   loader = DataLoader(train_dataset, batch_size=8, shuffle=True,
                       collate_fn=collate_bags, drop_last=True)

   # Optimizer: name "adam" | "adamw" | "sgd" | "rmsprop"; backbone boleh lr lebih kecil
   optim = build_optimizer(model, OptimizerConfig(name="adamw", lr=1e-4, weight_decay=1e-4,
                                                  backbone_lr_multiplier=0.1))
   scaler = torch.amp.GradScaler(enabled=(device == "cuda"))     # mixed precision

   model.train()
   for batch in loader:
       batch = {k: v.to(device) for k, v in batch.items()}
       with torch.autocast(device_type=device, enabled=(device == "cuda")):
           out = model(batch["images"], batch["image_mask"], batch["tabular"])
           loss, parts = model.compute_loss(out, batch["y_appendicitis"],
                                            batch["y_complicated"])
       optim.zero_grad(set_to_none=True)
       scaler.scale(loss).backward()
       scaler.step(optim); scaler.update()

   model.eval()
   with torch.no_grad():
       out = model(images, image_mask, tabular, return_attention=True)
       probs = model.predict_proba(out)          # (B, 3), urutan = CLASS_NAMES
       pred = probs.argmax(1)

-------------------------------------------------------------------------------
6. CATATAN PENTING
-------------------------------------------------------------------------------

   * Batch size minimal 2 saat training (BatchNorm); gunakan drop_last=True.
   * Kedua encoder diawali BatchNorm1d pada input. Scaling numerik train-only
     (mis. RobustScaler) tetap dapat diterapkan sebelumnya untuk mengelola
     skala/outlier; BatchNorm tidak menggantikan encoding kategori.
   * Backbone pretrained butuh unduhan bobot sekali (butuh internet).
   * `freeze_backbone=True` membekukan backbone (BN-nya ikut mode eval); berguna
     untuk data kecil atau tahap awal fine-tuning.
   * Memori: semua citra dalam batch lewat backbone sekaligus
     (jumlah total citra = jumlah True di image_mask). Bag terbesar di dataset ini
     ~15 citra, jadi kecilkan batch_size / resolusi bila GPU kehabisan memori.
   * Build PyTorch bertanda "+cpu" tidak memakai GPU; untuk CUDA pasang build CUDA.

-------------------------------------------------------------------------------
7. ISI FILE
-------------------------------------------------------------------------------
   Konstanta -> ModelConfig -> build_backbone -> sparsemax & blok TabNet ->
   TabNetEncoder / MLPEncoder -> AttentionMIL / LSTMPooling / MeanPooling /
   BagAggregator -> MultimodalMILModel -> labels_to_class, collate_bags ->
   count_parameters -> _smoke_test (jalankan: python multimodal_mil_model.py)
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from itertools import chain
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.nn.utils.rnn import pack_padded_sequence

# =============================================================================
# KONSTANTA
# =============================================================================

SUPPORTED_BACKBONES: Tuple[str, ...] = (
    "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
    "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3",
    "efficientnet_b4", "efficientnet_b5", "efficientnet_b6", "efficientnet_b7",
    "efficientnet_v2_s", "efficientnet_v2_m", "efficientnet_v2_l",
    "mobilenet_v2", "mobilenet_v3_small", "mobilenet_v3_large",
)
AGGREGATIONS = ("attention", "lstm", "mean")
ATTENTION_FUSIONS = ("shared", "dual")
HEADS = ("dual_head", "softmax")
TABULAR_ENCODERS = ("tabnet", "mlp")
OPTIMIZERS = ("adam", "adamw", "sgd", "rmsprop")

# Semua field dropout per komponen di ModelConfig (diatur global lewat use_dropout / dropout_rate)
DROPOUT_FIELDS = ("image_dropout", "attention_dropout", "lstm_dropout", "mlp_dropout", "fusion_dropout")

# Urutan kelas pada predict_proba / head softmax
CLASS_NAMES = ("no appendicitis", "uncomplicated_appendicitis", "complicated_appendicitis")

# Normalisasi ImageNet untuk backbone pretrained (terapkan di pipeline data)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# =============================================================================
# KONFIGURASI
# =============================================================================

@dataclass
class ModelConfig:
    """Seluruh opsi model. Nilai default = konfigurasi model usulan "P"."""

    # --- Input tabular ------------------------------------------------------
    num_tabular_features: int = 145
    """Jumlah kolom tabular setelah preprocessing (X_train.shape[1])."""

    # --- Dropout (pengaturan global) ----------------------------------------
    use_dropout: bool = True
    """False: matikan SEMUA dropout (tingkat per komponen di bawah dianggap 0)."""
    dropout_rate: Optional[float] = None
    """Jika diisi (0 <= r < 1): satu angka untuk SEMUA dropout (image, attention, lstm,
    mlp, fusion) dan menimpa tingkat per komponen. None = pakai tingkat per komponen.
    (TabNet tidak memakai dropout.)"""

    # --- Backbone citra -----------------------------------------------------
    backbone: str = "resnet18"
    """Salah satu dari SUPPORTED_BACKBONES."""
    pretrained: bool = True
    """True: bobot ImageNet; False: inisialisasi acak."""
    freeze_backbone: bool = False
    """True: backbone tidak dilatih (BN tetap mode eval)."""
    backbone_trainable_stage: Optional[str] = None
    """None: perilaku freeze_backbone lama. 'layer4': hanya blok terakhir ResNet
    dilatih, blok sebelumnya beserta BN tetap beku/eval. Memerlukan freeze_backbone=False."""
    image_embed_dim: Optional[int] = 256
    """Fitur backbone (512-2048 dim) diproyeksikan Linear+ReLU+Dropout ke dimensi ini
    (h_j). None = tanpa proyeksi (h_j = keluaran backbone apa adanya)."""
    image_dropout: float = 0.25
    """Dropout setelah proyeksi h_j."""

    # --- Agregasi MIL -------------------------------------------------------
    aggregation: str = "attention"
    """"attention" | "lstm" | "mean"."""
    attention_fusion: str = "dual"
    """"shared" | "dual". Hanya dipakai jika aggregation == "attention"."""
    attention_dim: int = 128
    """Dimensi tersembunyi attention (ukuran V dan U)."""
    attention_gated: bool = True
    """True: attention gated (tanh * sigmoid); False: tanh saja."""
    attention_dropout: float = 0.0
    """Dropout di dalam attention."""
    lstm_hidden: int = 256
    """Ukuran hidden LSTM (aggregation == "lstm")."""
    lstm_layers: int = 1
    lstm_bidirectional: bool = False
    lstm_dropout: float = 0.0
    """Dropout antar-layer LSTM (hanya aktif jika lstm_layers > 1)."""

    # --- Encoder tabular ----------------------------------------------------
    tabular_encoder: str = "tabnet"
    """"tabnet" | "mlp"."""
    tabular_embed_dim: int = 64
    """Dimensi embedding klinis e (paper: 64)."""
    mlp_hidden_dims: Tuple[int, ...] = (128,)
    """Lapisan tersembunyi MLP (tabular_encoder == "mlp")."""
    mlp_dropout: float = 0.2
    tabnet_n_steps: int = 5
    """Jumlah decision step TabNet (paper: 5)."""
    tabnet_gamma: float = 1.5
    """Koefisien relaksasi: >1 membolehkan fitur dipakai di beberapa step."""
    tabnet_n_shared: int = 2
    """Jumlah GLU layer yang dibagi antar-step."""
    tabnet_n_independent: int = 2
    """Jumlah GLU layer khusus tiap step."""
    tabnet_virtual_batch_size: int = 128
    tabnet_momentum: float = 0.02

    # --- Fusion & head ------------------------------------------------------
    fusion_hidden_dims: Tuple[int, ...] = (256,)
    """Lapisan Fusion MLP. Kosong () = langsung head."""
    fusion_dropout: float = 0.3
    head: str = "dual_head"
    """"dual_head" | "softmax"."""

    # --- Loss ---------------------------------------------------------------
    complication_loss_weight: float = 1.0
    """Bobot loss level 2 (komplikasi) pada head dual_head."""
    tabnet_sparsity_weight: float = 1e-3
    """Bobot loss sparsitas TabNet (default TabNet: 1e-3)."""

    def __post_init__(self) -> None:
        def check(name: str, value: str, allowed: Sequence[str]) -> None:
            if value not in allowed:
                raise ValueError(f"{name}='{value}' tidak dikenal. Pilihan: {list(allowed)}")

        check("backbone", self.backbone, SUPPORTED_BACKBONES)
        check("aggregation", self.aggregation, AGGREGATIONS)
        check("attention_fusion", self.attention_fusion, ATTENTION_FUSIONS)
        check("head", self.head, HEADS)
        check("tabular_encoder", self.tabular_encoder, TABULAR_ENCODERS)
        if self.backbone_trainable_stage is not None:
            if self.backbone_trainable_stage != "layer4" or not self.backbone.startswith("resnet"):
                raise ValueError("backbone_trainable_stage='layer4' hanya didukung untuk ResNet")
            if self.freeze_backbone:
                raise ValueError("Gunakan freeze_backbone=False untuk fine-tuning layer4")
        if self.num_tabular_features < 1:
            raise ValueError("num_tabular_features harus >= 1")
        if self.tabnet_n_shared + self.tabnet_n_independent < 1:
            raise ValueError("tabnet_n_shared + tabnet_n_independent harus >= 1")
        self.mlp_hidden_dims = tuple(self.mlp_hidden_dims)
        self.fusion_hidden_dims = tuple(self.fusion_hidden_dims)

        # Dropout: saklar global > satu angka global > tingkat per komponen
        if self.dropout_rate is not None and not 0.0 <= self.dropout_rate < 1.0:
            raise ValueError("dropout_rate harus di rentang [0, 1)")
        if not self.use_dropout:
            for f in DROPOUT_FIELDS:
                setattr(self, f, 0.0)
        elif self.dropout_rate is not None:
            for f in DROPOUT_FIELDS:
                setattr(self, f, self.dropout_rate)
        for f in DROPOUT_FIELDS:
            if not 0.0 <= getattr(self, f) < 1.0:
                raise ValueError(f"{f} harus di rentang [0, 1)")

    @property
    def dual_attention(self) -> bool:
        """True jika ada DUA attention terpisah (diagnosis & komplikasi)."""
        return self.aggregation == "attention" and self.attention_fusion == "dual"

    @classmethod
    def from_preset(cls, name: str, **overrides) -> "ModelConfig":
        """Buat konfigurasi dari kode pembanding ("B3".."B7", "P").

        `overrides` menimpa nilai apa pun, mis.
            ModelConfig.from_preset("P", num_tabular_features=145, backbone="resnet50")
        """
        if name in ("B1", "B2"):
            raise NotImplementedError(
                f"{name} adalah ablasi satu modalitas; modul ini baru mendukung model multimodal."
            )
        if name not in PRESETS:
            raise ValueError(f"Preset '{name}' tidak dikenal. Pilihan: {list(PRESETS)}")
        return cls(**{**PRESETS[name], **overrides})

    def to_dict(self) -> dict:
        return asdict(self)


PRESETS: Dict[str, dict] = {
    "B3": dict(aggregation="mean", head="dual_head"),
    "B4": dict(aggregation="lstm", head="dual_head"),
    "B5": dict(aggregation="attention", attention_fusion="shared", head="softmax"),
    "B6": dict(aggregation="attention", attention_fusion="shared", head="dual_head"),
    "B7": dict(aggregation="attention", attention_fusion="dual", head="softmax"),
    "P":  dict(aggregation="attention", attention_fusion="dual", head="dual_head"),
}


# =============================================================================
# BACKBONE CITRA
# =============================================================================

def build_backbone(name: str, pretrained: bool) -> Tuple[nn.Module, int]:
    """Bangun ekstraktor fitur citra dari torchvision.

    Mengembalikan (modul, dim_fitur). Modul menerima (M, 3, H, W) dan mengembalikan
    (M, dim_fitur): bagian klasifikasi (fc / classifier) dibuang dan diganti global
    average pooling, sehingga ukuran citra input bebas (mis. 224x224).
    """
    if name not in SUPPORTED_BACKBONES:
        raise ValueError(f"Backbone '{name}' tidak didukung. Pilihan: {list(SUPPORTED_BACKBONES)}")

    net = torchvision.models.get_model(name, weights="DEFAULT" if pretrained else None)

    if name.startswith("resnet"):
        dim = net.fc.in_features
        body = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool,
                             net.layer1, net.layer2, net.layer3, net.layer4)
    elif name.startswith("efficientnet"):
        dim = net.classifier[1].in_features
        body = net.features
    elif name == "mobilenet_v2":
        dim = net.classifier[1].in_features
        body = net.features
    else:  # mobilenet_v3_small / large
        dim = net.classifier[0].in_features
        body = net.features

    return nn.Sequential(body, nn.AdaptiveAvgPool2d(1), nn.Flatten(1)), dim


# =============================================================================
# TABNET (implementasi murni PyTorch)
# =============================================================================

def sparsemax(z: torch.Tensor) -> torch.Tensor:
    """Sparsemax (Martins & Astudillo, 2016) sepanjang dim terakhir.

    Seperti softmax, tetapi keluarannya bisa tepat 0, sehingga mask fitur TabNet
    benar-benar memilih sebagian fitur. Jumlah tiap baris = 1.
    """
    z_sorted, _ = torch.sort(z, dim=-1, descending=True)
    k = torch.arange(1, z.size(-1) + 1, device=z.device, dtype=z.dtype)
    z_cumsum = z_sorted.cumsum(dim=-1)
    in_support = (1 + k * z_sorted) > z_cumsum          # titik yang masuk "support"
    k_support = in_support.sum(dim=-1, keepdim=True)    # ukuran support
    tau = (z_cumsum.gather(-1, k_support - 1) - 1) / k_support.to(z.dtype)
    return torch.clamp(z - tau, min=0)


class GhostBatchNorm(nn.Module):
    """BatchNorm yang dihitung per 'virtual batch' (ghost batch) saat training.

    Membuat statistik BN lebih stabil/regularisasi lebih kuat pada batch besar.
    Saat eval atau batch <= virtual_batch_size, sama dengan BatchNorm biasa.
    """

    def __init__(self, num_features: int, virtual_batch_size: int = 128, momentum: float = 0.02):
        super().__init__()
        self.virtual_batch_size = virtual_batch_size
        self.bn = nn.BatchNorm1d(num_features, momentum=momentum)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and x.size(0) > self.virtual_batch_size:
            n_chunks = math.ceil(x.size(0) / self.virtual_batch_size)
            return torch.cat([self.bn(c) for c in x.chunk(n_chunks, dim=0)], dim=0)
        return self.bn(x)


class GLULayer(nn.Module):
    """Linear -> GhostBN -> GLU (separuh keluaran menjadi gerbang sigmoid)."""

    def __init__(self, in_dim: int, out_dim: int, virtual_batch_size: int, momentum: float,
                 fc: Optional[nn.Linear] = None):
        super().__init__()
        self.out_dim = out_dim
        self.fc = fc if fc is not None else nn.Linear(in_dim, 2 * out_dim, bias=False)
        self.bn = GhostBatchNorm(2 * out_dim, virtual_batch_size, momentum)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(self.fc(x))
        return x[:, : self.out_dim] * torch.sigmoid(x[:, self.out_dim:])


class TabNetFeatureTransformer(nn.Module):
    """Tumpukan GLU: Linear shared antar-step, setiap GLU memiliki BN sendiri.

    Layer pertama mengubah dimensi (tanpa residual); layer berikutnya memakai
    sambungan residual dengan penskalaan sqrt(0.5) agar varians stabil.
    """

    def __init__(self, in_dim: int, out_dim: int, shared_layers: nn.ModuleList,
                 n_independent: int, virtual_batch_size: int, momentum: float):
        super().__init__()
        # Input awal dan input masked memiliki distribusi berbeda. Hanya Linear
        # yang dibagi; affine parameters dan running statistics BN harus terpisah.
        self.shared = nn.ModuleList([
            GLULayer(in_dim if i == 0 else out_dim, out_dim,
                     virtual_batch_size, momentum, fc=fc)
            for i, fc in enumerate(shared_layers)
        ])
        independent = []
        for i in range(n_independent):
            first_overall = (len(shared_layers) == 0 and i == 0)
            independent.append(GLULayer(in_dim if first_overall else out_dim, out_dim,
                                        virtual_batch_size, momentum))
        self.independent = nn.ModuleList(independent)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = math.sqrt(0.5)
        for i, layer in enumerate(chain(self.shared, self.independent)):
            out = layer(x)
            x = out if i == 0 else (x + out) * scale
        return x


class TabNetAttentiveTransformer(nn.Module):
    """Menghasilkan mask fitur step ini: sparsemax( GhostBN(Linear(a)) * prior )."""

    def __init__(self, n_a: int, input_dim: int, virtual_batch_size: int, momentum: float):
        super().__init__()
        self.fc = nn.Linear(n_a, input_dim, bias=False)
        self.bn = GhostBatchNorm(input_dim, virtual_batch_size, momentum)

    def forward(self, prior: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return sparsemax(self.bn(self.fc(a)) * prior)


class TabNetEncoder(nn.Module):
    """TabNet encoder (Arik & Pfister, 2021) -> embedding klinis e.

    Alur per decision step:
        1. mask M   = attentive_transformer(prior, a)      (fitur mana yang dilihat)
        2. prior    = (gamma - M) * prior                  (fitur yang sudah dipakai
                                                            dikurangi jatahnya)
        3. out      = feature_transformer(M * x)           -> pecah jadi [d | a_next]
        4. kontribusi step = ReLU(d)
    Embedding e = jumlah kontribusi semua step (dimensi n_d = embed_dim).
    """

    def __init__(self, input_dim: int, embed_dim: int, n_steps: int, gamma: float,
                 n_shared: int, n_independent: int, virtual_batch_size: int, momentum: float):
        super().__init__()
        self.input_dim, self.n_d, self.n_a = input_dim, embed_dim, embed_dim
        self.n_steps, self.gamma = n_steps, gamma
        width = self.n_d + self.n_a

        self.initial_bn = nn.BatchNorm1d(input_dim, momentum=momentum)  # normalisasi input

        shared = nn.ModuleList([
            nn.Linear(input_dim if i == 0 else width, 2 * width, bias=False)
            for i in range(n_shared)
        ])
        self.initial_transformer = TabNetFeatureTransformer(
            input_dim, width, shared, n_independent, virtual_batch_size, momentum)
        self.feature_transformers = nn.ModuleList([
            TabNetFeatureTransformer(input_dim, width, shared, n_independent,
                                     virtual_batch_size, momentum)
            for _ in range(n_steps)
        ])
        self.attentive_transformers = nn.ModuleList([
            TabNetAttentiveTransformer(self.n_a, input_dim, virtual_batch_size, momentum)
            for _ in range(n_steps)
        ])

    def forward(self, x: torch.Tensor, return_masks: bool = False):
        """Mengembalikan (embedding (B, n_d), sparsity_loss skalar, masks (B, n_steps, F) | None)."""
        x = self.initial_bn(x)
        prior = torch.ones_like(x)
        a = self.initial_transformer(x)[:, self.n_d:]

        embedding = x.new_zeros(x.size(0), self.n_d)
        sparsity = x.new_zeros(())
        masks: List[torch.Tensor] = []

        for step in range(self.n_steps):
            mask = self.attentive_transformers[step](prior, a)
            # entropi mask (dihitung float32 agar log stabil di mixed precision)
            m32 = mask.float()
            sparsity = sparsity + torch.mean(torch.sum(-m32 * torch.log(m32 + 1e-15), dim=1))
            prior = (self.gamma - mask) * prior

            out = self.feature_transformers[step](mask * x)
            embedding = embedding + F.relu(out[:, : self.n_d])
            a = out[:, self.n_d:]
            if return_masks:
                masks.append(mask)

        sparsity = sparsity / self.n_steps
        return embedding, sparsity, (torch.stack(masks, dim=1) if return_masks else None)


class MLPEncoder(nn.Module):
    """Encoder tabular sederhana: BN input -> [Linear -> ReLU -> Dropout]* -> Linear -> ReLU."""

    def __init__(self, input_dim: int, embed_dim: int, hidden_dims: Sequence[int], dropout: float):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(input_dim)
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, embed_dim), nn.ReLU()]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, return_masks: bool = False):
        """Antarmuka sama dengan TabNetEncoder: (embedding, None, None)."""
        return self.net(self.input_bn(x)), None, None


# =============================================================================
# AGREGASI MIL
# =============================================================================

class AttentionMIL(nn.Module):
    """Attention-MIL (gated) terhadap tumpukan fitur citra dengan mask padding.

        s_j = w^T ( tanh(V h_j) * sigmoid(U h_j) )      (gated; tanpa sigmoid jika gated=False)
        a   = softmax_j(s_j)  hanya atas citra asli
        z   = sum_j a_j h_j
    """

    def __init__(self, in_dim: int, attn_dim: int, gated: bool, dropout: float):
        super().__init__()
        self.V = nn.Linear(in_dim, attn_dim)
        self.U = nn.Linear(in_dim, attn_dim) if gated else None
        self.w = nn.Linear(attn_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.out_dim = in_dim

    def forward(self, h: torch.Tensor, mask: torch.Tensor):
        """h: (B, N, F); mask: (B, N) bool -> (z (B, F), alpha (B, N))."""
        a = torch.tanh(self.V(h))
        if self.U is not None:
            a = a * torch.sigmoid(self.U(h))
        scores = self.w(self.dropout(a)).squeeze(-1)                  # (B, N)
        scores = scores.masked_fill(~mask, float("-inf"))             # padding -> bobot 0
        alpha = torch.softmax(scores, dim=1)
        z = torch.bmm(alpha.unsqueeze(1), h).squeeze(1)               # (B, F)
        return z, alpha


class LSTMPooling(nn.Module):
    """Agregasi dengan LSTM: z = state tersembunyi terakhir (hanya langkah asli, bukan padding)."""

    def __init__(self, in_dim: int, hidden: int, layers: int, bidirectional: bool, dropout: float):
        super().__init__()
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=layers, batch_first=True,
                            bidirectional=bidirectional, dropout=dropout if layers > 1 else 0.0)
        self.out_dim = hidden * (2 if bidirectional else 1)

    def forward(self, h: torch.Tensor, mask: torch.Tensor):
        lengths = mask.sum(dim=1).cpu()
        packed = pack_padded_sequence(h, lengths, batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)       # h_n: (layers * arah, B, hidden)
        z = torch.cat([h_n[-2], h_n[-1]], dim=1) if self.bidirectional else h_n[-1]
        return z, None                        # LSTM tidak punya bobot per-citra


class MeanPooling(nn.Module):
    """Rata-rata fitur citra asli (padding diabaikan)."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.out_dim = in_dim

    def forward(self, h: torch.Tensor, mask: torch.Tensor):
        m = mask.unsqueeze(-1).to(h.dtype)
        count = m.sum(dim=1).clamp(min=1)
        z = (h * m).sum(dim=1) / count
        alpha = mask.to(h.dtype) / count                               # bobot seragam 1/n
        return z, alpha


class BagAggregator(nn.Module):
    """Pembungkus yang memilih agregasi sesuai config dan selalu mengembalikan
    ringkasan bag untuk dua tugas: (z_diagnosis, z_complication, alpha_D, alpha_C).

    Jika tidak ada attention terpisah (shared / lstm / mean), z_complication adalah
    tensor yang SAMA dengan z_diagnosis.
    """

    def __init__(self, cfg: ModelConfig, in_dim: int):
        super().__init__()
        self.dual = cfg.dual_attention
        self.second: Optional[nn.Module] = None
        if cfg.aggregation == "attention":
            self.first = AttentionMIL(in_dim, cfg.attention_dim, cfg.attention_gated, cfg.attention_dropout)
            if self.dual:
                self.second = AttentionMIL(in_dim, cfg.attention_dim, cfg.attention_gated, cfg.attention_dropout)
        elif cfg.aggregation == "lstm":
            self.first = LSTMPooling(in_dim, cfg.lstm_hidden, cfg.lstm_layers,
                                     cfg.lstm_bidirectional, cfg.lstm_dropout)
        else:
            self.first = MeanPooling(in_dim)
        self.out_dim: int = self.first.out_dim

    def forward(self, h: torch.Tensor, mask: torch.Tensor):
        z_d, alpha_d = self.first(h, mask)
        if self.second is None:
            return z_d, z_d, alpha_d, alpha_d
        z_c, alpha_c = self.second(h, mask)
        return z_d, z_c, alpha_d, alpha_c


# =============================================================================
# MODEL UTAMA
# =============================================================================

def _mlp(in_dim: int, hidden_dims: Sequence[int], dropout: float) -> Tuple[nn.Sequential, int]:
    """[Linear -> ReLU -> Dropout]* ; mengembalikan (modul, dim_keluaran)."""
    layers: List[nn.Module] = []
    prev = in_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
        prev = h
    return nn.Sequential(*layers), prev


def configure_backbone_training(backbone: nn.Module, cfg: ModelConfig) -> None:
    """Keep legacy full/frozen modes; optionally train only ResNet layer4."""
    backbone.requires_grad_(not cfg.freeze_backbone)
    if cfg.backbone_trainable_stage == "layer4":
        backbone.requires_grad_(False)
        backbone[0][7].requires_grad_(True)  # body: conv,bn,relu,pool,layer1..4


def set_backbone_mode(backbone: nn.Module, cfg: ModelConfig, mode: bool) -> None:
    if cfg.freeze_backbone:
        backbone.eval()
    elif cfg.backbone_trainable_stage == "layer4":
        backbone.eval()
        backbone[0][7].train(mode)
    else:
        backbone.train(mode)


class MultimodalMILModel(nn.Module):
    """Model multimodal MIL (citra USG + data klinis). Lihat docstring modul."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # --- Cabang citra: backbone (+ proyeksi) -> h_j -----------------------
        self.backbone, backbone_dim = build_backbone(cfg.backbone, cfg.pretrained)
        configure_backbone_training(self.backbone, cfg)
        if cfg.image_embed_dim is None:
            self.image_proj: nn.Module = nn.Identity()
            feat_dim = backbone_dim
        else:
            self.image_proj = nn.Sequential(nn.Linear(backbone_dim, cfg.image_embed_dim),
                                            nn.ReLU(), nn.Dropout(cfg.image_dropout))
            feat_dim = cfg.image_embed_dim

        # --- Agregasi bag -> z --------------------------------------------------
        self.aggregator = BagAggregator(cfg, feat_dim)
        bag_dim = self.aggregator.out_dim

        # --- Cabang tabular -> e ------------------------------------------------
        if cfg.tabular_encoder == "tabnet":
            self.tabular = TabNetEncoder(
                cfg.num_tabular_features, cfg.tabular_embed_dim, cfg.tabnet_n_steps,
                cfg.tabnet_gamma, cfg.tabnet_n_shared, cfg.tabnet_n_independent,
                cfg.tabnet_virtual_batch_size, cfg.tabnet_momentum)
        else:
            self.tabular = MLPEncoder(cfg.num_tabular_features, cfg.tabular_embed_dim,
                                      cfg.mlp_hidden_dims, cfg.mlp_dropout)
        emb_dim = cfg.tabular_embed_dim

        # --- Fusion + head ------------------------------------------------------
        if cfg.head == "dual_head":
            self.fusion_diagnosis, fus_dim = _mlp(bag_dim + emb_dim, cfg.fusion_hidden_dims, cfg.fusion_dropout)
            self.fusion_complication, _ = _mlp(bag_dim + emb_dim, cfg.fusion_hidden_dims, cfg.fusion_dropout)
            self.head_diagnosis = nn.Linear(fus_dim, 1)       # logit p = P(A | X, c)
            self.head_complication = nn.Linear(fus_dim, 1)    # logit q = P(C | A, X, c)
        else:
            n_bag_vectors = 2 if cfg.dual_attention else 1    # dual attention: pakai zD dan zC
            self.fusion, fus_dim = _mlp(n_bag_vectors * bag_dim + emb_dim,
                                        cfg.fusion_hidden_dims, cfg.fusion_dropout)
            self.head_softmax = nn.Linear(fus_dim, len(CLASS_NAMES))

    # ------------------------------------------------------------------ mode
    def train(self, mode: bool = True) -> "MultimodalMILModel":
        """Seperti nn.Module.train, tetapi backbone beku tetap di mode eval
        (agar statistik BatchNorm-nya tidak berubah)."""
        super().train(mode)
        set_backbone_mode(self.backbone, self.cfg, mode)
        return self

    # --------------------------------------------------------------- citra
    def encode_images(self, images: torch.Tensor, image_mask: torch.Tensor) -> torch.Tensor:
        """(B, N, 3, H, W) + mask -> h (B, N, F). Hanya citra asli yang lewat backbone;
        posisi padding diisi nol."""
        flat = images[image_mask]                                  # (M, 3, H, W), M = total citra asli
        feats = self.image_proj(self.backbone(flat))               # (M, F)
        h = feats.new_zeros(image_mask.size(0), image_mask.size(1), feats.size(-1))
        h[image_mask] = feats
        return h

    # ------------------------------------------------------------- forward
    def forward(self, images: torch.Tensor, image_mask: torch.Tensor, tabular: torch.Tensor,
                return_attention: bool = False) -> Dict[str, Optional[torch.Tensor]]:
        self._check_inputs(images, image_mask, tabular)

        h = self.encode_images(images, image_mask)                                   # (B, N, F)
        e, tab_loss, tab_masks = self.tabular(tabular, return_masks=return_attention)  # (B, E)
        z_d, z_c, alpha_d, alpha_c = self.aggregator(h, image_mask)

        out: Dict[str, Optional[torch.Tensor]] = {"tab_sparsity_loss": tab_loss}
        if self.cfg.head == "dual_head":
            f_d = self.fusion_diagnosis(torch.cat([z_d, e], dim=1))
            f_c = self.fusion_complication(torch.cat([z_c, e], dim=1))
            out["logit_diagnosis"] = self.head_diagnosis(f_d).squeeze(-1)            # (B,)
            out["logit_complication"] = self.head_complication(f_c).squeeze(-1)      # (B,)
        else:
            parts = [z_d, z_c, e] if self.cfg.dual_attention else [z_d, e]
            out["logits"] = self.head_softmax(self.fusion(torch.cat(parts, dim=1)))  # (B, 3)

        if return_attention:
            out["attention_diagnosis"] = alpha_d
            out["attention_complication"] = alpha_c
            out["tabular_masks"] = tab_masks
        return out

    @staticmethod
    def _check_inputs(images: torch.Tensor, image_mask: torch.Tensor, tabular: torch.Tensor) -> None:
        if images.dim() != 5:
            raise ValueError(f"images harus (B, N, 3, H, W), dapat {tuple(images.shape)}")
        if image_mask.shape != images.shape[:2] or image_mask.dtype != torch.bool:
            raise ValueError("image_mask harus bool dengan shape (B, N)")
        if tabular.size(0) != images.size(0):
            raise ValueError("Batch size images dan tabular berbeda")
        if not image_mask.any(dim=1).all():
            raise ValueError("Ada bag tanpa citra asli (image_mask semua False)")
        if (image_mask[:, 1:] & ~image_mask[:, :-1]).any():
            raise ValueError("Padding harus di BELAKANG (citra asli di awal tiap bag)")

    # ------------------------------------------------------------- prediksi
    @staticmethod
    def _class_probs_from(out: Dict[str, Optional[torch.Tensor]]) -> torch.Tensor:
        if "logits" in out:
            return torch.softmax(out["logits"].float(), dim=-1)
        p = torch.sigmoid(out["logit_diagnosis"].float())
        q = torch.sigmoid(out["logit_complication"].float())
        return torch.stack([1 - p, p * (1 - q), p * q], dim=-1)

    def predict_proba(self, out: Dict[str, Optional[torch.Tensor]]) -> torch.Tensor:
        """Probabilitas 3 kelas (B, 3), urutan CLASS_NAMES, dari keluaran forward.

        dual_head : [1-p, p(1-q), p*q] (hierarkis)   softmax : softmax(logits)
        """
        return self._class_probs_from(out)

    def predict_classes(self, out: Dict[str, Optional[torch.Tensor]], threshold: float = 0.5) -> torch.Tensor:
        """Kelas prediksi (B,) berindeks CLASS_NAMES.

        dual_head : keputusan hierarkis. p < threshold -> 0 (no appendicitis);
                    selain itu 1 + (q >= threshold) -> 1 (uncomplicated) / 2 (complicated).
                    (Berbeda dari argmax predict_proba, yang bisa memilih kelas 0 walau p > 0.5.)
        softmax   : argmax logits.
        """
        if "logits" in out:
            return out["logits"].argmax(dim=-1)
        p = torch.sigmoid(out["logit_diagnosis"].float())
        q = torch.sigmoid(out["logit_complication"].float())
        return torch.where(p < threshold, torch.zeros_like(p, dtype=torch.long),
                           1 + (q >= threshold).long())

    # ----------------------------------------------------------------- loss
    def compute_loss(self, out: Dict[str, Optional[torch.Tensor]], y_appendicitis: torch.Tensor,
                     y_complicated: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Loss total + rincian (untuk logging).

        dual_head : BCE(level 1, semua pasien)
                    + complication_loss_weight * BCE(level 2, hanya y_complicated >= 0)
        softmax   : cross-entropy 3 kelas (label dari labels_to_class)
        TabNet    : + tabnet_sparsity_weight * loss sparsitas mask
        """
        cfg = self.cfg
        parts: Dict[str, float] = {}

        if cfg.head == "dual_head":
            loss_d = F.binary_cross_entropy_with_logits(
                out["logit_diagnosis"].float(), y_appendicitis.float())
            valid = y_complicated >= 0                      # pasien apendisitis saja
            if valid.any():
                loss_c = F.binary_cross_entropy_with_logits(
                    out["logit_complication"][valid].float(), y_complicated[valid].float())
            else:                                           # batch tanpa apendisitis
                loss_c = out["logit_complication"].sum() * 0.0
            total = loss_d + cfg.complication_loss_weight * loss_c
            parts.update(loss_diagnosis=loss_d.item(), loss_complication=loss_c.item())
        else:
            target = labels_to_class(y_appendicitis, y_complicated)
            total = F.cross_entropy(out["logits"].float(), target)
            parts["loss_ce"] = total.item()

        if out.get("tab_sparsity_loss") is not None and cfg.tabnet_sparsity_weight > 0:
            total = total + cfg.tabnet_sparsity_weight * out["tab_sparsity_loss"]
            parts["loss_tab_sparsity"] = out["tab_sparsity_loss"].item()

        parts["loss_total"] = total.item()
        return total, parts


# =============================================================================
# UTILITAS
# =============================================================================

def labels_to_class(y_appendicitis: torch.Tensor, y_complicated: torch.Tensor) -> torch.Tensor:
    """Dua kolom label hierarkis -> indeks kelas 3 (0=no, 1=uncomplicated, 2=complicated)."""
    return torch.where(y_appendicitis == 0, torch.zeros_like(y_appendicitis),
                       1 + y_complicated.clamp(min=0))


def collate_bags(batch: List[dict]) -> Dict[str, torch.Tensor]:
    """collate_fn DataLoader: menumpuk pasien dengan jumlah citra berbeda.

    Tiap elemen batch: {"images": (n, 3, H, W), "tabular": (F,),
                        "y_appendicitis": int, "y_complicated": int}
    Bag dipadding nol di belakang; image_mask menandai citra asli.
    Kunci hasil sama dengan argumen forward (images, image_mask, tabular).
    """
    n_max = max(item["images"].size(0) for item in batch)
    ref = batch[0]["images"]
    images = ref.new_zeros(len(batch), n_max, *ref.shape[1:])
    mask = torch.zeros(len(batch), n_max, dtype=torch.bool)
    for i, item in enumerate(batch):
        n = item["images"].size(0)
        images[i, :n] = item["images"]
        mask[i, :n] = True
    return {
        "images": images,
        "image_mask": mask,
        "tabular": torch.stack([item["tabular"] for item in batch]),
        "y_appendicitis": torch.tensor([int(item["y_appendicitis"]) for item in batch], dtype=torch.long),
        "y_complicated": torch.tensor([int(item["y_complicated"]) for item in batch], dtype=torch.long),
    }


@dataclass
class OptimizerConfig:
    """Pengaturan optimizer dan learning rate (dipakai oleh build_optimizer)."""

    name: str = "adamw"
    """"adam" | "adamw" | "sgd" | "rmsprop"."""
    lr: float = 1e-4
    """Learning rate dasar (untuk bagian model selain backbone)."""
    weight_decay: float = 1e-4
    """Regularisasi L2. Pada "adam" ini L2 biasa (terikat ke gradien); pada "adamw"
    berupa decoupled weight decay (umumnya lebih baik)."""
    backbone_lr_multiplier: float = 1.0
    """lr backbone = lr * pengali ini. Untuk fine-tuning backbone pretrained, nilai
    kecil (mis. 0.1) menjaga bobot ImageNet agar tidak rusak, sementara bagian baru
    (attention, tabular, fusion, head) tetap memakai lr penuh."""
    momentum: float = 0.9
    """Hanya "sgd" dan "rmsprop"."""
    nesterov: bool = True
    """Hanya "sgd" (momentum Nesterov)."""
    betas: Tuple[float, float] = (0.9, 0.999)
    """Hanya "adam" dan "adamw"."""
    no_decay_bias_norm: bool = True
    """True: bias dan parameter BatchNorm/LayerNorm (tensor 1-D) tidak kena weight decay."""

    def __post_init__(self) -> None:
        if self.name not in OPTIMIZERS:
            raise ValueError(f"optimizer '{self.name}' tidak dikenal. Pilihan: {list(OPTIMIZERS)}")
        if self.lr <= 0:
            raise ValueError("lr harus > 0")
        if self.backbone_lr_multiplier < 0:
            raise ValueError("backbone_lr_multiplier harus >= 0")


def build_optimizer(model: nn.Module, cfg: OptimizerConfig) -> torch.optim.Optimizer:
    """Bangun optimizer dengan param group terpisah.

    Parameter dibagi menurut (backbone atau bukan) x (kena weight decay atau tidak),
    sehingga backbone bisa memakai lr berbeda (backbone_lr_multiplier). Parameter
    yang dibekukan (requires_grad=False, mis. freeze_backbone=True) dilewati.
    Tiap group diberi kunci "name" untuk memudahkan logging / scheduler.
    """
    buckets: Dict[Tuple[bool, bool], List[nn.Parameter]] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_backbone = name.startswith("backbone.")
        no_decay = cfg.no_decay_bias_norm and p.ndim <= 1
        buckets.setdefault((is_backbone, no_decay), []).append(p)

    groups = []
    for (is_backbone, no_decay), params in sorted(buckets.items()):
        groups.append({
            "params": params,
            "lr": cfg.lr * (cfg.backbone_lr_multiplier if is_backbone else 1.0),
            "weight_decay": 0.0 if no_decay else cfg.weight_decay,
            "name": f"{'backbone' if is_backbone else 'head'}{'_no_decay' if no_decay else ''}",
        })
    if not groups:
        raise ValueError("Tidak ada parameter yang dapat dilatih (semua requires_grad=False)")

    if cfg.name == "adam":
        return torch.optim.Adam(groups, lr=cfg.lr, betas=cfg.betas)
    if cfg.name == "adamw":
        return torch.optim.AdamW(groups, lr=cfg.lr, betas=cfg.betas)
    if cfg.name == "sgd":
        return torch.optim.SGD(groups, lr=cfg.lr, momentum=cfg.momentum, nesterov=cfg.nesterov)
    return torch.optim.RMSprop(groups, lr=cfg.lr, momentum=cfg.momentum)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Jumlah parameter total/terlatih dan per komponen utama."""
    def n(mod: nn.Module, trainable_only: bool = False) -> int:
        return sum(p.numel() for p in mod.parameters() if p.requires_grad or not trainable_only)

    result = {"total": n(model), "trainable": n(model, True)}
    for name, child in model.named_children():
        result[name] = n(child)
    return result


# =============================================================================
# SMOKE TEST  (python multimodal_mil_model.py)
# =============================================================================

def _smoke_test() -> None:
    """Uji cepat di CPU/GPU dengan data acak (tanpa mengunduh bobot pretrained)."""
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_feat, sizes, hw = 20, [1, 3, 2], 64
    B, N = len(sizes), max(sizes)
    print(f"[smoke] device={device}")

    images = torch.randn(B, N, 3, hw, hw, device=device)
    mask = torch.zeros(B, N, dtype=torch.bool, device=device)
    for i, s in enumerate(sizes):
        mask[i, :s] = True
    images = images * mask[..., None, None, None]
    tab = torch.randn(B, n_feat, device=device)
    y_app = torch.tensor([0, 1, 1], device=device)
    y_comp = torch.tensor([-1, 0, 1], device=device)

    # 1) semua preset x kedua encoder tabular
    for preset in PRESETS:
        for enc in TABULAR_ENCODERS:
            cfg = ModelConfig.from_preset(preset, num_tabular_features=n_feat, pretrained=False,
                                          tabular_encoder=enc, tabnet_n_steps=2)
            model = MultimodalMILModel(cfg).to(device).train()
            out = model(images, mask, tab, return_attention=True)
            loss, parts = model.compute_loss(out, y_app, y_comp)
            loss.backward()
            probs = model.predict_proba(out)
            assert probs.shape == (B, 3) and torch.allclose(probs.sum(1), torch.ones(B, device=device), atol=1e-5)
            for part in (model.backbone, model.tabular, model.aggregator):   # gradien harus sampai
                if not any(True for _ in part.parameters()):                 # mean pooling: tanpa parameter
                    continue
                assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in part.parameters()), \
                    f"{preset}/{enc}: tidak ada gradien di {type(part).__name__}"
            if out["attention_diagnosis"] is not None:      # bobot: jumlah 1, padding 0
                a = out["attention_diagnosis"]
                assert torch.allclose(a.sum(1), torch.ones(B, device=device), atol=1e-5)
                assert (a[~mask] == 0).all()
            print(f"[smoke] {preset:>2} / {enc:6s}  loss={parts['loss_total']:.3f}  "
                  f"params={count_parameters(model)['total']:,}")

    # 2) invariansi padding: pasien yang sama harus berhasil identik sendirian maupun dalam batch
    for preset in ("P", "B4", "B3", "B7"):
        cfg = ModelConfig.from_preset(preset, num_tabular_features=n_feat, pretrained=False,
                                      tabular_encoder="mlp")
        model = MultimodalMILModel(cfg).to(device).eval()
        with torch.no_grad():
            batch_out = model.predict_proba(model(images, mask, tab))
            solo = model.predict_proba(model(images[1:2, :3], mask[1:2, :3], tab[1:2]))
        assert torch.allclose(batch_out[1], solo[0], atol=1e-4), f"{preset}: padding mengubah hasil"
    print("[smoke] invarian terhadap padding: OK")

    # 3) semua backbone: dimensi fitur & forward
    for name in SUPPORTED_BACKBONES:
        net, dim = build_backbone(name, pretrained=False)
        net = net.to(device).eval()
        with torch.no_grad():
            y = net(torch.randn(2, 3, 64, 64, device=device))
        assert y.shape == (2, dim), (name, y.shape, dim)
    print(f"[smoke] {len(SUPPORTED_BACKBONES)} backbone: OK")

    # 4) freeze_backbone: parameter beku & tetap eval saat train()
    cfg = ModelConfig.from_preset("P", num_tabular_features=n_feat, pretrained=False, freeze_backbone=True)
    model = MultimodalMILModel(cfg).to(device).train()
    assert not any(p.requires_grad for p in model.backbone.parameters())
    assert not model.backbone.training
    print("[smoke] freeze_backbone: OK")

    # 5) dropout global: saklar, satu angka, dan tingkat per komponen
    c_off = ModelConfig(use_dropout=False, dropout_rate=0.5)
    assert all(getattr(c_off, f) == 0.0 for f in DROPOUT_FIELDS)            # saklar mengalahkan angka
    c_all = ModelConfig(dropout_rate=0.4, fusion_dropout=0.1)
    assert all(getattr(c_all, f) == 0.4 for f in DROPOUT_FIELDS)            # angka global menimpa per komponen
    c_each = ModelConfig(fusion_dropout=0.1, mlp_dropout=0.0)
    assert c_each.fusion_dropout == 0.1 and c_each.image_dropout == 0.25    # default per komponen utuh
    m_off = MultimodalMILModel(ModelConfig(num_tabular_features=n_feat, pretrained=False,
                                           tabular_encoder="mlp", use_dropout=False))
    assert all(d.p == 0.0 for d in m_off.modules() if isinstance(d, nn.Dropout))
    m_on = MultimodalMILModel(ModelConfig(num_tabular_features=n_feat, pretrained=False,
                                          tabular_encoder="mlp", dropout_rate=0.5))
    assert {d.p for d in m_on.modules() if isinstance(d, nn.Dropout)} == {0.5}
    try:
        ModelConfig(dropout_rate=1.0)
        raise AssertionError("dropout_rate=1.0 seharusnya ditolak")
    except ValueError:
        pass
    print("[smoke] dropout: OK")

    # 6) semua optimizer: satu langkah update, lr backbone terpisah, bias/BN tanpa weight decay
    for opt_name in OPTIMIZERS:
        cfg = ModelConfig.from_preset("P", num_tabular_features=n_feat, pretrained=False, tabular_encoder="mlp")
        model = MultimodalMILModel(cfg).to(device).train()
        opt = build_optimizer(model, OptimizerConfig(name=opt_name, lr=1e-3, weight_decay=1e-2,
                                                     backbone_lr_multiplier=0.1))
        lrs = {g["name"]: g["lr"] for g in opt.param_groups}
        assert abs(lrs["backbone"] - 1e-4) < 1e-12 and abs(lrs["head"] - 1e-3) < 1e-12, lrs
        assert all(g["weight_decay"] == 0.0 for g in opt.param_groups if g["name"].endswith("no_decay"))
        before = model.head_diagnosis.weight.detach().clone()
        out = model(images, mask, tab)
        model.compute_loss(out, y_app, y_comp)[0].backward()
        opt.step()
        assert not torch.equal(before, model.head_diagnosis.weight), f"{opt_name}: bobot tidak berubah"
    frozen = MultimodalMILModel(ModelConfig.from_preset("P", num_tabular_features=n_feat, pretrained=False,
                                                        freeze_backbone=True))
    assert not any(g["name"].startswith("backbone") for g in
                   build_optimizer(frozen, OptimizerConfig()).param_groups)    # backbone beku dilewati
    print(f"[smoke] optimizer {list(OPTIMIZERS)}: OK")
    print("[smoke] SEMUA UJI LULUS")


if __name__ == "__main__":
    _smoke_test()
