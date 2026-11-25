"""Modular FiLM + Scaled Dot-Product Cross-Attention MIL + Trimmed EfficientNet-B0 for Proposed Model 3."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

CLASS_IDS = (0, 1, 2)
BACKBONE_MODES = ("stage7_8", "stage7_only", "stage8_only", "frozen")
AGGREGATOR_MODES = ("cross_attn", "mv_mean", "mv_lstm")
MODALITY_MODES = ("multimodal", "image_only", "tabular_only")
FUSION_MODES = ("film", "gated", "concat")
TASK_MODES = ("hierarchical", "softmax", "binary_diag", "binary_sev")


@dataclass
class ModelConfig:
    clinical_dim: int = 28
    num_concepts: int = 8
    modality: str = "multimodal"
    backbone_mode: str = "stage7_8"
    aggregator: str = "cross_attn"
    fusion_mode: str = "film"
    task_mode: str = "hierarchical"
    use_concepts: bool = True
    instance_film: bool = True
    clinical_query: bool = True
    clinical_skip: bool = True
    attn_heads: int = 4
    attn_dim: int = 128
    attn_temperature: float = 0.7
    attn_dropout: float = 0.1
    mil_dropout: float = 0.35
    clinical_dropout: float = 0.25
    modality_dropout: float = 0.15
    pretrained: bool = True
    image_microbatch: int = 16
    gradient_checkpointing: bool = False
    diagnosis_threshold: float = 0.50
    complication_threshold: float = 0.50

    def __post_init__(self):
        if self.modality not in MODALITY_MODES:
            raise ValueError(f"modality harus dalam {MODALITY_MODES}")
        if self.backbone_mode not in BACKBONE_MODES:
            raise ValueError(f"backbone_mode harus dalam {BACKBONE_MODES}")
        if self.aggregator not in AGGREGATOR_MODES:
            raise ValueError(f"aggregator harus dalam {AGGREGATOR_MODES}")
        if self.fusion_mode not in FUSION_MODES:
            raise ValueError(f"fusion_mode harus dalam {FUSION_MODES}")
        if self.task_mode not in TASK_MODES:
            raise ValueError(f"task_mode harus dalam {TASK_MODES}")
        if self.clinical_dim < 1 or self.image_microbatch < 1:
            raise ValueError("clinical_dim dan image_microbatch harus positif.")
        if self.attn_dim % self.attn_heads != 0:
            raise ValueError("attn_dim harus habis dibagi attn_heads.")
        if self.attn_temperature <= 0:
            raise ValueError("attn_temperature harus positif.")
        if not 0 < self.diagnosis_threshold < 1 or not 0 < self.complication_threshold < 1:
            raise ValueError("Threshold diagnosis dan komplikasi harus dalam (0, 1).")


class EfficientNetB0Encoder(nn.Module):
    """Trimmed EfficientNet-B0 encoder with locked BatchNorm and masked spatial pooling."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if cfg.pretrained else None
        net = efficientnet_b0(weights=weights)
        self.backbone_mode = cfg.backbone_mode
        self.checkpointing = cfg.gradient_checkpointing

        if cfg.backbone_mode == "stage7_8":
            self.frozen = net.features[:7]
            self.trainable_stages = nn.ModuleList([net.features[7], net.features[8]])
            self.out_channels = 1280
        elif cfg.backbone_mode == "stage7_only":
            # Truncate features[8] so output is 320 channels directly from stage 7.
            self.frozen = net.features[:7]
            self.trainable_stages = nn.ModuleList([net.features[7]])
            self.out_channels = 320
        elif cfg.backbone_mode == "stage8_only":
            self.frozen = net.features[:8]
            self.trainable_stages = nn.ModuleList([net.features[8]])
            self.out_channels = 1280
        elif cfg.backbone_mode == "frozen":
            self.frozen = net.features
            self.trainable_stages = nn.ModuleList()
            self.out_channels = 1280
        else:
            raise ValueError(f"backbone_mode tidak dikenal: {cfg.backbone_mode}")

        self.frozen.requires_grad_(False)
        self.last_conv_module = self.trainable_stages[-1] if len(self.trainable_stages) else self.frozen[-1]
        self._warmup_frozen = False

    def set_backbone_frozen(self, frozen: bool):
        self._warmup_frozen = bool(frozen)
        for stage in self.trainable_stages:
            stage.requires_grad_(not self._warmup_frozen)

    def train(self, mode: bool = True):
        super().train(mode)
        self.frozen.eval()
        # Keep ALL BatchNorm2d layers in eval() so microbatch=2 never corrupts running stats.
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        return self

    def forward(self, images: torch.Tensor, masks: torch.Tensor, allow_checkpoint: bool = True) -> torch.Tensor:
        with torch.no_grad():
            x = self.frozen(images)
        for stage in self.trainable_stages:
            can_ckpt = (
                allow_checkpoint
                and self.training
                and self.checkpointing
                and any(p.requires_grad for p in stage.parameters())
            )
            if self._warmup_frozen:
                with torch.no_grad():
                    x = stage(x)
            elif can_ckpt:
                x = checkpoint(stage, x, use_reentrant=False)
            else:
                x = stage(x)
        x = x.float()
        mask = F.adaptive_avg_pool2d(masks.float(), x.shape[-2:])
        denom = mask.sum((2, 3)).clamp_min(1e-6)
        return (x * mask).sum((2, 3)) / denom


class ResidualFiLM(nn.Module):
    """Feature-wise Linear Modulation initialized to exact identity (delta_gamma=0, beta=0)."""

    def __init__(self, cond_dim: int, feat_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, 64),
            nn.GELU(),
            nn.Linear(64, 2 * feat_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, feat: torch.Tensor, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        params = self.net(cond)
        delta_gamma, beta = params.chunk(2, dim=-1)
        if feat.dim() == 2 and delta_gamma.dim() == 1:
            delta_gamma_b = delta_gamma.unsqueeze(0)
            beta_b = beta.unsqueeze(0)
        else:
            delta_gamma_b = delta_gamma
            beta_b = beta
        out = (1.0 + delta_gamma_b) * feat + beta_b
        return out, delta_gamma, beta


class ScaledDotProductCrossAttentionMIL(nn.Module):
    """Attention Is All You Need multi-head scaled dot-product cross-attention MIL pooling."""

    def __init__(
        self,
        dim: int = 128,
        num_heads: int = 4,
        temperature: float = 0.7,
        dropout: float = 0.1,
        clinical_dim: int = 32,
        clinical_query: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.temperature = temperature
        self.scale = math.sqrt(self.head_dim) * temperature

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

        self.q_base = nn.Parameter(torch.randn(dim) * 0.02)
        self.q_clin = nn.Linear(clinical_dim, dim) if clinical_query else None
        if self.q_clin is not None:
            nn.init.zeros_(self.q_clin.weight)
            nn.init.zeros_(self.q_clin.bias)

    def forward(
        self, h: torch.Tensor, e_clin: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # h: [K_i, D]
        k_len = h.shape[0]
        q_vec = self.q_base
        if e_clin is not None and self.q_clin is not None:
            q_vec = q_vec + self.q_clin(e_clin)

        q = self.q_proj(q_vec).view(self.num_heads, 1, self.head_dim)
        k = self.k_proj(h).view(k_len, self.num_heads, self.head_dim).transpose(0, 1)  # [H, K_i, d_k]
        v = self.v_proj(h).view(k_len, self.num_heads, self.head_dim).transpose(0, 1)  # [H, K_i, d_k]

        scores = torch.matmul(q.float(), k.float().transpose(-2, -1)).squeeze(1) / self.scale  # [H, K_i]
        weights_per_head = F.softmax(scores, dim=-1)  # [H, K_i]
        weights_drop = self.attn_drop(weights_per_head)
        ctx = torch.matmul(weights_drop.unsqueeze(1), v.float()).squeeze(1).reshape(self.dim)  # [D]

        weights_mean = weights_per_head.mean(dim=0)  # [K_i]
        residual = (weights_mean[:, None] * h.float()).sum(dim=0)
        z = self.norm(self.out_proj(ctx) + residual)
        return z, weights_mean


class MultiViewMeanAggregator(nn.Module):
    def forward(self, h: torch.Tensor, e_clin: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        k_len = h.shape[0]
        weights = torch.full((k_len,), 1.0 / k_len, device=h.device, dtype=torch.float32)
        return h.float().mean(dim=0), weights


class MultiViewLSTMAggregator(nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()
        self.lstm = nn.LSTM(dim, dim, num_layers=1, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, h: torch.Tensor, e_clin: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        k_len = h.shape[0]
        _, (hn, _) = self.lstm(h.float().unsqueeze(0))
        weights = torch.full((k_len,), 1.0 / k_len, device=h.device, dtype=torch.float32)
        return self.norm(hn[-1, 0]), weights


class GatedFusionBlock(nn.Module):
    def __init__(self, vis_dim: int, clin_dim: int, out_dim: int = 32):
        super().__init__()
        self.vis_proj = nn.Sequential(nn.Linear(vis_dim, out_dim), nn.LayerNorm(out_dim), nn.GELU())
        self.gate = nn.Linear(out_dim + clin_dim, out_dim)

    def forward(self, z_vis: torch.Tensor, e_clin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        v = self.vis_proj(z_vis)
        g = self.gate(torch.cat([v, e_clin], dim=-1)).sigmoid()
        return g * v + (1.0 - g) * e_clin, g


class ClinicalEncoder(nn.Module):
    def __init__(self, in_dim: int = 28, out_dim: int = 32, dropout: float = 0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())


class PM3Model(nn.Module):
    """Unified Proposed Model 3 supporting multimodal, image-only, tabular-only, and all ablations."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.is_dual_branch = cfg.task_mode == "hierarchical"

        # 1. Visual pipeline
        if cfg.modality in ("multimodal", "image_only"):
            self.encoder = EfficientNetB0Encoder(cfg)
            self.img_proj = nn.Sequential(
                nn.Linear(self.encoder.out_channels, cfg.attn_dim),
                nn.LayerNorm(cfg.attn_dim),
                nn.GELU(),
                nn.Dropout(cfg.mil_dropout),
            )
            self.agg_diag = self._make_aggregator(cfg)
            self.agg_comp = self._make_aggregator(cfg) if self.is_dual_branch else None

            if cfg.use_concepts:
                self.concept_head = nn.Linear(cfg.attn_dim, cfg.num_concepts)
            # Direct visual auxiliary supervision heads on pure unmodulated visual features
            if cfg.task_mode == "hierarchical":
                self.aux_img_diag = nn.Linear(cfg.attn_dim, 1)
                self.aux_img_comp = nn.Linear(cfg.attn_dim, 1)
            elif cfg.task_mode == "softmax":
                self.aux_img_softmax = nn.Linear(cfg.attn_dim, 3)
            else:
                self.aux_img_binary = nn.Linear(cfg.attn_dim, 1)

        # 2. Clinical pipeline
        if cfg.modality in ("multimodal", "tabular_only"):
            self.clinical_mlp = ClinicalEncoder(cfg.clinical_dim, 32, cfg.clinical_dropout)
            if cfg.clinical_skip:
                out_k = 3 if cfg.task_mode == "softmax" else 1
                self.clinical_skip_a = nn.Linear(cfg.clinical_dim, out_k)
                self.clinical_skip_b = nn.Linear(cfg.clinical_dim, 1) if self.is_dual_branch else None

        # 3. Fusion & final heads
        if cfg.modality == "multimodal":
            use_clin_q = cfg.clinical_query and cfg.aggregator == "cross_attn"
            if cfg.fusion_mode == "film":
                if cfg.instance_film:
                    self.inst_film = ResidualFiLM(32, cfg.attn_dim)
                self.bag_film_a = ResidualFiLM(32, cfg.attn_dim)
                self.bag_film_b = ResidualFiLM(32, cfg.attn_dim) if self.is_dual_branch else None
                self.vis_anchor_proj_a = nn.Sequential(nn.Linear(cfg.attn_dim, 32), nn.LayerNorm(32), nn.GELU())
                self.vis_anchor_proj_b = (
                    nn.Sequential(nn.Linear(cfg.attn_dim, 32), nn.LayerNorm(32), nn.GELU())
                    if self.is_dual_branch
                    else None
                )
                head_in_dim = 64  # [e_clin (32) ; vis_film_proj (32)]
            elif cfg.fusion_mode == "gated":
                self.gated_a = GatedFusionBlock(cfg.attn_dim, 32, 32)
                self.gated_b = GatedFusionBlock(cfg.attn_dim, 32, 32) if self.is_dual_branch else None
                head_in_dim = 64  # [e_clin (32) ; gated (32)] preserves clinical anchor
            else:  # concat
                self.vis_concat_a = nn.Sequential(nn.Linear(cfg.attn_dim, 32), nn.LayerNorm(32), nn.GELU())
                self.vis_concat_b = (
                    nn.Sequential(nn.Linear(cfg.attn_dim, 32), nn.LayerNorm(32), nn.GELU())
                    if self.is_dual_branch
                    else None
                )
                head_in_dim = 64
            self._use_clin_query = use_clin_q
        elif cfg.modality == "image_only":
            head_in_dim = cfg.attn_dim
        else:  # tabular_only
            head_in_dim = 32

        if cfg.task_mode == "hierarchical":
            self.head_diag = nn.Linear(head_in_dim, 1)
            self.head_comp = nn.Linear(head_in_dim, 1)
        elif cfg.task_mode == "softmax":
            self.head_softmax = nn.Linear(head_in_dim, 3)
        else:
            self.head_binary = nn.Linear(head_in_dim, 1)

    def _make_aggregator(self, cfg: ModelConfig) -> nn.Module:
        use_cq = cfg.modality == "multimodal" and cfg.clinical_query
        if cfg.aggregator == "cross_attn":
            return ScaledDotProductCrossAttentionMIL(
                dim=cfg.attn_dim,
                num_heads=cfg.attn_heads,
                temperature=cfg.attn_temperature,
                dropout=cfg.attn_dropout,
                clinical_dim=32,
                clinical_query=use_cq,
            )
        if cfg.aggregator == "mv_mean":
            return MultiViewMeanAggregator()
        if cfg.aggregator == "mv_lstm":
            return MultiViewLSTMAggregator(dim=cfg.attn_dim)
        raise ValueError(f"Aggregator tidak dikenal: {cfg.aggregator}")

    def set_backbone_frozen(self, frozen: bool):
        if hasattr(self, "encoder"):
            self.encoder.set_backbone_frozen(frozen)

    def encode_images_chunked(
        self,
        images: torch.Tensor,
        pixel_masks: torch.Tensor,
        allow_checkpoint: bool = True,
        pad_to_microbatch: bool = False,
    ) -> torch.Tensor:
        """Encode arbitrarily sized image batch in fixed-size chunks when pad_to_microbatch=True."""
        step = max(1, self.cfg.image_microbatch)
        n_orig = len(images)
        if pad_to_microbatch and (n_orig % step != 0):
            pad_n = step - (n_orig % step)
            pad_imgs = images.new_zeros((pad_n, *images.shape[1:]))
            pad_msks = pixel_masks.new_ones((pad_n, *pixel_masks.shape[1:]))
            images = torch.cat((images, pad_imgs), dim=0)
            pixel_masks = torch.cat((pixel_masks, pad_msks), dim=0)

        if len(images) <= step:
            feats = self.encoder(images, pixel_masks, allow_checkpoint=allow_checkpoint)
        else:
            parts = [
                self.encoder(
                    images[start : start + step],
                    pixel_masks[start : start + step],
                    allow_checkpoint=allow_checkpoint,
                )
                for start in range(0, len(images), step)
            ]
            feats = torch.cat(parts, dim=0)
        return self.img_proj(feats[:n_orig])

    def _fuse_branch(self, z_vis: torch.Tensor, e_clin: torch.Tensor, branch: str):
        if self.cfg.fusion_mode == "film":
            film = self.bag_film_a if branch == "a" else self.bag_film_b
            proj = self.vis_anchor_proj_a if branch == "a" else self.vis_anchor_proj_b
            z_mod, dg, beta = film(z_vis, e_clin)
            fused = torch.cat([e_clin, proj(z_mod)], dim=-1)
            return fused, {"delta_gamma": dg, "beta": beta, "gate": None}
        if self.cfg.fusion_mode == "gated":
            gated = self.gated_a if branch == "a" else self.gated_b
            g_out, g = gated(z_vis, e_clin)
            fused = torch.cat([e_clin, g_out], dim=-1)
            return fused, {"delta_gamma": None, "beta": None, "gate": g}
        proj = self.vis_concat_a if branch == "a" else self.vis_concat_b
        fused = torch.cat([e_clin, proj(z_vis)], dim=-1)
        return fused, {"delta_gamma": None, "beta": None, "gate": None}

    def forward_from_embeddings(
        self,
        h_raw: Optional[torch.Tensor],
        tabular: Optional[torch.Tensor],
    ) -> dict:
        """Run patient-level MIL, FiLM, concepts, and heads from pre-encoded visual embeddings."""
        out: Dict[str, object] = {}
        attn_dict: Dict[str, torch.Tensor] = {}
        film_stats: Dict[str, float] = {}
        collect_stats = not self.training

        # 1. Clinical stream + modality dropout
        e_clin = None
        tab_for_skip = None
        if self.cfg.modality in ("multimodal", "tabular_only"):
            if tabular is None:
                raise ValueError("Data tabular wajib ada.")
            tab_f = tabular.float()
            e_clin = self.clinical_mlp(tab_f)
            tab_for_skip = tab_f
            if self.cfg.modality == "multimodal" and self.training and self.cfg.modality_dropout > 0:
                keep = (torch.rand((), device=e_clin.device) >= self.cfg.modality_dropout).float()
                scale = 1.0 / (1.0 - self.cfg.modality_dropout)
                e_clin = e_clin * keep * scale
                tab_for_skip = tab_for_skip * keep * scale

        # 2. Visual stream
        if self.cfg.modality in ("multimodal", "image_only"):
            if h_raw is None:
                raise ValueError("Visual instance embeddings wajib ada.")

            # Pure unmodulated visual pooling for concept bottleneck & aux visual supervision
            z_raw_a, attn_raw_a = self.agg_diag(h_raw, None)
            if self.is_dual_branch:
                z_raw_b, attn_raw_b = self.agg_comp(h_raw, None)
                z_for_concepts = 0.5 * (z_raw_a + z_raw_b)
            else:
                z_raw_b, attn_raw_b = None, None
                z_for_concepts = z_raw_a

            if self.cfg.use_concepts:
                out["concept_logits"] = self.concept_head(z_for_concepts).float()

            if self.cfg.modality == "multimodal":
                if self.cfg.task_mode == "hierarchical":
                    out["aux_img_logit_diag"] = self.aux_img_diag(z_raw_a).float().squeeze()
                    out["aux_img_logit_comp"] = self.aux_img_comp(z_raw_b).float().squeeze()
                elif self.cfg.task_mode == "softmax":
                    out["aux_img_logits"] = self.aux_img_softmax(z_raw_a).float()
                else:
                    out["aux_img_logit"] = self.aux_img_binary(z_raw_a).float().squeeze()

                # Apply optional instance-level FiLM and clinical-guided queries for main fusion path
                if self.cfg.fusion_mode == "film" and self.cfg.instance_film:
                    h_main, dg_inst, b_inst = self.inst_film(h_raw, e_clin)
                    if collect_stats:
                        film_stats["inst_delta_gamma_l2"] = float(dg_inst.detach().norm())
                        film_stats["inst_beta_l2"] = float(b_inst.detach().norm())
                else:
                    h_main = h_raw

                cq = e_clin if (self.cfg.clinical_query and self.cfg.aggregator == "cross_attn") else None
                if (h_main is h_raw) and (cq is None):
                    z_a, attn_a = z_raw_a, attn_raw_a
                    z_b, attn_b = z_raw_b, attn_raw_b
                else:
                    z_a, attn_a = self.agg_diag(h_main, cq)
                    z_b, attn_b = self.agg_comp(h_main, cq) if self.is_dual_branch else (None, None)

                attn_dict["diag" if self.is_dual_branch else "unified"] = attn_a
                if self.is_dual_branch and attn_b is not None:
                    attn_dict["comp"] = attn_b

                feat_a, st_a = self._fuse_branch(z_a, e_clin, "a")
                if collect_stats:
                    if st_a["delta_gamma"] is not None:
                        film_stats["bag_diag_delta_gamma_l2"] = float(st_a["delta_gamma"].detach().norm())
                        film_stats["bag_diag_beta_l2"] = float(st_a["beta"].detach().norm())
                    if st_a["gate"] is not None:
                        film_stats["gate_diag_mean"] = float(st_a["gate"].detach().mean())

                if self.is_dual_branch:
                    feat_b, st_b = self._fuse_branch(z_b, e_clin, "b")
                    if collect_stats:
                        if st_b["delta_gamma"] is not None:
                            film_stats["bag_comp_delta_gamma_l2"] = float(st_b["delta_gamma"].detach().norm())
                            film_stats["bag_comp_beta_l2"] = float(st_b["beta"].detach().norm())
                        if st_b["gate"] is not None:
                            film_stats["gate_comp_mean"] = float(st_b["gate"].detach().mean())
                else:
                    feat_b = None
            else:
                attn_dict["diag" if self.is_dual_branch else "unified"] = attn_raw_a
                if self.is_dual_branch and attn_raw_b is not None:
                    attn_dict["comp"] = attn_raw_b
                feat_a, feat_b = z_raw_a, z_raw_b
        else:
            feat_a = e_clin
            feat_b = e_clin if self.is_dual_branch else None

        # 3. Task Heads + Optional Direct Clinical Skip
        if self.cfg.task_mode == "hierarchical":
            logit_diag = self.head_diag(feat_a).float().squeeze()
            logit_comp = self.head_comp(feat_b).float().squeeze()
            if self.cfg.modality in ("multimodal", "tabular_only") and self.cfg.clinical_skip:
                logit_diag = logit_diag + self.clinical_skip_a(tab_for_skip).float().squeeze()
                logit_comp = logit_comp + self.clinical_skip_b(tab_for_skip).float().squeeze()
            p, q = logit_diag.sigmoid(), logit_comp.sigmoid()
            probs = torch.stack((1.0 - p, p * (1.0 - q), p * q))
            pred = torch.where(
                p < self.cfg.diagnosis_threshold,
                0,
                torch.where(q < self.cfg.complication_threshold, 1, 2),
            )
            out.update({"logit_diag": logit_diag, "logit_comp": logit_comp, "p": p, "q": q, "probs": probs, "pred": pred})
        elif self.cfg.task_mode == "softmax":
            logits = self.head_softmax(feat_a).float()
            if self.cfg.modality in ("multimodal", "tabular_only") and self.cfg.clinical_skip:
                logits = logits + self.clinical_skip_a(tab_for_skip).float()
            probs = logits.softmax(dim=-1)
            pred = probs.argmax(dim=-1)
            out.update({"logits": logits, "probs": probs, "pred": pred})
        else:
            logit = self.head_binary(feat_a).float().squeeze()
            if self.cfg.modality in ("multimodal", "tabular_only") and self.cfg.clinical_skip:
                logit = logit + self.clinical_skip_a(tab_for_skip).float().squeeze()
            prob = logit.sigmoid()
            thr = self.cfg.diagnosis_threshold if self.cfg.task_mode == "binary_diag" else self.cfg.complication_threshold
            pred = (prob >= thr).long()
            out.update({"logit": logit, "prob": prob, "probs": torch.stack((1.0 - prob, prob)), "pred": pred})

        out["attention"] = attn_dict
        out["film_stats"] = film_stats
        return out

    def forward(
        self,
        images: Optional[torch.Tensor],
        pixel_masks: Optional[torch.Tensor],
        tabular: Optional[torch.Tensor],
        allow_checkpoint: bool = True,
    ) -> dict:
        h_raw = None
        if self.cfg.modality in ("multimodal", "image_only"):
            if images is None or pixel_masks is None:
                raise ValueError("Citra USG dan pixel_masks wajib ada.")
            h_raw = self.encode_images_chunked(images, pixel_masks, allow_checkpoint=allow_checkpoint)
        return self.forward_from_embeddings(h_raw, tabular)

    def parameter_groups(
        self, backbone_lr: float = 2e-5, attention_lr: float = 2e-4, head_lr: float = 4e-4
    ) -> List[dict]:
        """Partition parameters into discriminative AdamW groups without overlap or omission."""
        backbone_ids = set()
        backbone_params = []
        if hasattr(self, "encoder"):
            # Include all trainable_stages parameters even if currently warmup-frozen so AdamW keeps their slots
            for p in self.encoder.trainable_stages.parameters():
                backbone_params.append(p)
                backbone_ids.add(id(p))

        attn_modules = [
            m for name in ("img_proj", "agg_diag", "agg_comp", "concept_head")
            if hasattr(self, name) and (m := getattr(self, name)) is not None
        ]
        attn_ids = set()
        attn_params = []
        for m in attn_modules:
            for p in m.parameters():
                if id(p) not in backbone_ids and id(p) not in attn_ids and p.requires_grad:
                    attn_params.append(p)
                    attn_ids.add(id(p))

        clin_modules = [
            m for name in ("clinical_mlp", "clinical_skip_a", "clinical_skip_b")
            if hasattr(self, name) and (m := getattr(self, name)) is not None
        ]
        clin_ids = set()
        clin_params = []
        for m in clin_modules:
            for p in m.parameters():
                if id(p) not in backbone_ids and id(p) not in attn_ids and id(p) not in clin_ids and p.requires_grad:
                    clin_params.append(p)
                    clin_ids.add(id(p))

        used = backbone_ids | attn_ids | clin_ids
        head_params = [p for p in self.parameters() if id(p) not in used and p.requires_grad]

        groups = []
        if backbone_params:
            groups.append({"name": "backbone", "params": backbone_params, "lr": backbone_lr, "weight_decay": 1e-3})
        if attn_params:
            groups.append({"name": "attention_concept", "params": attn_params, "lr": attention_lr, "weight_decay": 1e-4})
        if clin_params:
            groups.append({"name": "clinical_encoder", "params": clin_params, "lr": head_lr, "weight_decay": 1e-2})
        if head_params:
            groups.append({"name": "fusion_heads", "params": head_params, "lr": head_lr, "weight_decay": 1e-4})
        return groups

    def config_dict(self) -> dict:
        return asdict(self.cfg)
