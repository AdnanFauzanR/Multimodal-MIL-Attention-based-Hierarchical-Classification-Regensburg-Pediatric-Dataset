"""Explainable AI (XAI) for Proposed Model 3: MIL attention, Grad-CAM, 8 concepts, FiLM, and Integrated Gradients."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

from pm3_data import CLASS_NAMES, CONCEPTS_PM3, FEATURES
from pm3_model import PM3Model


def _select_scalar_logit(out: dict, target: str) -> torch.Tensor:
    if "logit_diag" in out:
        return out["logit_diag"] if target == "diag" else out["logit_comp"]
    if "logits" in out:
        idx = int(out["pred"]) if target == "diag" else 2
        return out["logits"][idx]
    return out["logit"]


def compute_gradcam_for_patient(
    model: PM3Model, item: dict, device: torch.device, target: str = "diag"
) -> List[np.ndarray]:
    """Compute Grad-CAM heatmaps [K_i, H, W] over last_conv_module for all USG slices of a patient."""
    if not hasattr(model, "encoder"):
        raise ValueError("Grad-CAM hanya tersedia untuk model yang memiliki cabang citra.")
    model.eval()
    activations, gradients = [], []

    def fwd_hook(_mod, _inp, out_t):
        activations.append(out_t)
        out_t.retain_grad()

    def bwd_hook(_mod, _grad_in, grad_out):
        gradients.append(grad_out[0])

    hook_target = model.encoder.last_conv_module
    h_fwd = hook_target.register_forward_hook(fwd_hook)
    h_bwd = hook_target.register_full_backward_hook(bwd_hook)

    # Temporarily allow gradients through frozen encoder stages only if needed for hook activation grads
    try:
        imgs = item["images"].to(device).detach().requires_grad_(True)
        masks = item["pixel_masks"].to(device)
        tab = item["tabular"].to(device) if item["tabular"] is not None else None
        model.zero_grad(set_to_none=True)
        out = model(imgs, masks, tab, allow_checkpoint=False)
        logit = _select_scalar_logit(out, target)
        logit.backward()

        # Activations and gradients come in microbatch chunks in forward order / reverse backward order
        acts = torch.cat(activations, dim=0).detach()
        grads = torch.cat(list(reversed(gradients)), dim=0).detach()
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * acts).sum(dim=1, keepdim=True))
        cam_up = F.interpolate(cam, size=imgs.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
        cam_up = cam_up * (masks.squeeze(1) > 0).float()

        cams = []
        for c in cam_up.cpu().numpy():
            mx = float(c.max())
            cams.append(c / mx if mx > 1e-8 else c)
        return cams
    finally:
        h_fwd.remove()
        h_bwd.remove()
        model.zero_grad(set_to_none=True)


def compute_clinical_attribution(
    model: PM3Model,
    item: dict,
    device: torch.device,
    target: str = "diag",
    steps: int = 20,
    feature_names: List[str] | None = None,
) -> Dict[str, float]:
    """Integrated Gradients attribution across the 28 clinical features."""
    if item["tabular"] is None or model.cfg.modality == "image_only":
        return {}
    model.eval()
    names = list(feature_names or FEATURES)
    x = item["tabular"].to(device).float()
    baseline = torch.zeros_like(x)
    imgs = item["images"].to(device) if item["images"] is not None else None
    masks = item["pixel_masks"].to(device) if item["pixel_masks"] is not None else None

    total_grad = torch.zeros_like(x)
    for alpha in torch.linspace(0.0, 1.0, steps + 1, device=device)[1:]:
        x_step = (baseline + alpha * (x - baseline)).detach().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        out = model(imgs, masks, x_step, allow_checkpoint=False)
        logit = _select_scalar_logit(out, target)
        logit.backward()
        if x_step.grad is not None:
            total_grad += x_step.grad.detach()
    ig = ((x - baseline) * (total_grad / steps)).cpu().numpy()
    model.zero_grad(set_to_none=True)
    return {name: float(val) for name, val in zip(names, ig)}


def explain_patient(
    model: PM3Model,
    item: dict,
    device: torch.device,
    feature_names: List[str] | None = None,
    save_path: str | Path | None = None,
    target: str = "diag",
):
    """Generate clinical XAI dashboard (Top USG + Grad-CAM, Attention, 8 Concepts, Clinical IG, FiLM)."""
    model.eval()
    with torch.no_grad():
        imgs = item["images"].to(device) if item["images"] is not None else None
        masks = item["pixel_masks"].to(device) if item["pixel_masks"] is not None else None
        tab = item["tabular"].to(device) if item["tabular"] is not None else None
        out = model(imgs, masks, tab, allow_checkpoint=False)

    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)

    # Panel A1: Top-attended USG slice + Grad-CAM
    ax_img = fig.add_subplot(gs[0, 0])
    if hasattr(model, "encoder") and imgs is not None:
        cams = compute_gradcam_for_patient(model, item, device, target=target)
        attn_key = "diag" if "diag" in out["attention"] else "unified"
        a_w = out["attention"][attn_key].detach().cpu().numpy()
        top_idx = int(np.argmax(a_w))
        raw_slice = imgs[top_idx, 0].detach().cpu().numpy()
        raw_slice = (raw_slice - raw_slice.min()) / max(raw_slice.max() - raw_slice.min(), 1e-6)
        ax_img.imshow(raw_slice, cmap="gray")
        ax_img.imshow(cams[top_idx], cmap="jet", alpha=0.40)
        ax_img.set_title(f"Top Slice #{top_idx+1}/{len(a_w)} ({item['image_ids'][top_idx]})\nGrad-CAM ({target}) | Attn={a_w[top_idx]:.3f}")
    else:
        ax_img.text(0.5, 0.5, "Tabular-Only Mode", ha="center", va="center")
    ax_img.axis("off")

    # Panel A2: MIL Attention weights across views
    ax_attn = fig.add_subplot(gs[0, 1])
    if out["attention"]:
        x_pos = np.arange(len(item["image_ids"]))
        width = 0.35 if len(out["attention"]) == 2 else 0.6
        for i, (bname, w_t) in enumerate(out["attention"].items()):
            offset = (i - 0.5) * width if len(out["attention"]) == 2 else 0.0
            ax_attn.bar(x_pos + offset, w_t.detach().cpu().numpy(), width=width, label=bname)
        ax_attn.set_xticks(x_pos)
        ax_attn.set_xticklabels([f"V{i+1}" for i in x_pos], fontsize=8)
        ax_attn.set_ylim(0, 1.05)
        ax_attn.legend()
        ax_attn.set_title("MIL Attention Weights per USG View")
    else:
        ax_attn.axis("off")

    # Panel B: 8 Curated Sonographic Concepts
    ax_con = fig.add_subplot(gs[0, 2])
    if "concept_logits" in out:
        c_prob = out["concept_logits"].sigmoid().detach().cpu().numpy()
        c_gt = item["concepts_binary"].numpy()
        c_mask = item["concept_mask"].numpy()
        y_pos = np.arange(len(CONCEPTS_PM3))
        ax_con.barh(y_pos, c_prob, color="steelblue", alpha=0.85, label="Pred Prob")
        for yi, gt, mk in zip(y_pos, c_gt, c_mask):
            marker = "o" if mk > 0 else "x"
            col = "crimson" if gt >= 0.5 else "black"
            ax_con.scatter([gt], [yi], color=col, marker=marker, s=45, zorder=5)
        ax_con.set_yticks(y_pos)
        ax_con.set_yticklabels(CONCEPTS_PM3, fontsize=8)
        ax_con.set_xlim(0, 1.05)
        ax_con.axvline(0.5, ls="--", color="gray", lw=0.8)
        ax_con.set_title("8 Sonographic Concepts (Pred vs GT)")
    else:
        ax_con.text(0.5, 0.5, "Concepts Disabled", ha="center", va="center")
        ax_con.axis("off")

    # Panel C1: Top-10 Clinical Integrated Gradients
    ax_ig = fig.add_subplot(gs[1, :2])
    attr = compute_clinical_attribution(model, item, device, target=target, feature_names=feature_names)
    if attr:
        top10 = sorted(attr.items(), key=lambda kv: abs(kv[1]), reverse=True)[:10][::-1]
        names_10 = [k for k, _ in top10]
        vals_10 = [v for _, v in top10]
        colors = ["crimson" if v > 0 else "teal" for v in vals_10]
        ax_ig.barh(np.arange(len(top10)), vals_10, color=colors)
        ax_ig.set_yticks(np.arange(len(top10)))
        ax_ig.set_yticklabels(names_10, fontsize=9)
        ax_ig.axvline(0, color="black", lw=0.8)
        ax_ig.set_title(f"Top-10 Clinical Feature Attributions (Integrated Gradients, target={target})")
    else:
        ax_ig.axis("off")

    # Panel C2: Decision + FiLM summary
    ax_sum = fig.add_subplot(gs[1, 2])
    ax_sum.axis("off")
    lines = [
        f"Patient ID : {item['patient_id']}",
        f"True Label : {CLASS_NAMES[item['label']] if item['label'] < 3 else item['label']}",
        f"Pred Label : {CLASS_NAMES[int(out['pred'])] if len(out['probs']) == 3 else int(out['pred'])}",
        "",
    ]
    if "p" in out:
        p, q = float(out["p"]), float(out["q"])
        probs = out["probs"].detach().cpu().numpy()
        lines += [
            f"p_diag     : {p:.3f}",
            f"q_comp|app : {q:.3f}",
            f"P(No App)  : {probs[0]:.3f}",
            f"P(Uncomp)  : {probs[1]:.3f}",
            f"P(Comp)    : {probs[2]:.3f}",
        ]
    else:
        probs = out["probs"].detach().cpu().numpy()
        lines += [f"Probs: {np.round(probs, 3).tolist()}"]
    if out.get("film_stats"):
        lines.append("")
        lines.append("FiLM / Fusion Stats:")
        for k_s, v_s in out["film_stats"].items():
            lines.append(f"  {k_s}: {v_s:.3f}")
    ax_sum.text(0.02, 0.95, "\n".join(lines), va="top", family="monospace", fontsize=9.5)

    fig.suptitle(f"Proposed Model 3 — Clinical XAI Summary (Patient {item['patient_id']})", fontsize=12, fontweight="bold")
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig
