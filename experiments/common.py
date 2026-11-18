"""Shared code for the ABCDFSS output-stage experiments.

The functions in this module operate on the 13 layer predictions produced by
``core.denseaffinity.DAMatComparison.calc_q_pred_coarses``.  Keeping all
fusion / threshold / post-processing variants here makes it possible to run
every experiment offline from a cached set of layer predictions, without
re-running the (expensive) test-time adapter for each variant.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils import segutils  # noqa: E402

# ---------------------------------------------------------------------------
# Layer groups (Low / Mid / High = 4 / 6 / 3, as in HSNet / DCAMA / ABCDFSS)
# ---------------------------------------------------------------------------
STAGE_SLICES: Dict[str, slice] = {
    "low": slice(0, 4),
    "mid": slice(4, 10),
    "high": slice(10, 13),
}
STAGE_PAIRS: Dict[str, Tuple[str, str]] = {
    "low_mid": ("low", "mid"),
    "mid_high": ("mid", "high"),
    "low_high": ("low", "high"),
}

IMG_SIZE = 400
N_LAYERS = 13


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def git_commit(path: Optional[Path] = None) -> str:
    try:
        import subprocess

        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(path or REPO_ROOT),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def remote_url(path: Optional[Path] = None) -> str:
    try:
        import subprocess

        url = subprocess.check_output(
            ["git", "remote", "get-url", "origin"],
            cwd=str(path or REPO_ROOT),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if url.startswith("git@"):
            url = "https://" + url[4:].replace(":", "/", 1)
        if url.endswith(".git"):
            url = url[:-4]
        return url
    except Exception:
        return "https://github.com/tiudan147/BTL_XLA"


def commit_link(sha: str) -> str:
    return f"{remote_url()}/commit/{sha}"


def _check_layers(layers: torch.Tensor) -> torch.Tensor:
    if layers.ndim == 3:
        layers = layers.unsqueeze(0)
    if layers.ndim != 4:
        raise ValueError(f"Expected [B, 13, H, W] layer predictions, got {tuple(layers.shape)}")
    if layers.shape[1] != N_LAYERS:
        raise ValueError(
            f"Expected {N_LAYERS} layer predictions, got {layers.shape[1]}"
        )
    return layers


def _stage_maps(layers: torch.Tensor) -> torch.Tensor:
    """[B, 13, H, W] -> [B, 3, H, W] with one mean map per Low/Mid/High stage."""
    layers = _check_layers(layers)
    return torch.stack(
        [layers[:, STAGE_SLICES[name]].mean(dim=1) for name in ("low", "mid", "high")],
        dim=1,
    )


def upsample_map(map_t: torch.Tensor, size: Tuple[int, int] = (IMG_SIZE, IMG_SIZE)) -> torch.Tensor:
    """Bilinearly upsample [B, H, W] to ``size`` (same op as the baseline)."""
    if map_t.ndim == 2:
        map_t = map_t.unsqueeze(0)
    if tuple(map_t.shape[-2:]) == tuple(size):
        return map_t
    return F.interpolate(
        map_t.unsqueeze(1), size=size, mode="bilinear", align_corners=False
    ).squeeze(1)


# ---------------------------------------------------------------------------
# Fusion variants (group A)
# ---------------------------------------------------------------------------
def _otsu_eta_np(x: np.ndarray) -> float:
    """Otsu between-class variance divided by total variance (in [0, 1])."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    mn, mx = float(x.min()), float(x.max())
    if mx - mn < 1e-12:
        return 0.0
    x = (x - mn) / (mx - mn)
    hist, edges = np.histogram(x, bins=256, range=(0.0, 1.0))
    p = hist.astype(np.float64) / max(hist.sum(), 1)
    centers = (edges[:-1] + edges[1:]) * 0.5
    omega = np.cumsum(p)
    mu = np.cumsum(p * centers)
    mu_t = mu[-1]
    sigma_t = float(np.sum(p * (centers - mu_t) ** 2))
    if sigma_t < 1e-12:
        return 0.0
    denom = omega * (1.0 - omega)
    sigma_b = np.where(denom > 1e-12, (mu_t * omega - mu) ** 2 / np.maximum(denom, 1e-12), 0.0)
    return float(np.max(sigma_b) / sigma_t)


def _entropy_np(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    mn, mx = float(x.min()), float(x.max())
    if mx - mn < 1e-12:
        return 0.0
    x = (x - mn) / (mx - mn)
    hist, _ = np.histogram(x, bins=256, range=(0.0, 1.0))
    p = hist.astype(np.float64) / max(hist.sum(), 1)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def _adaptive_weights(stage_maps: torch.Tensor, kind: str, temperature: float) -> torch.Tensor:
    """Weight the three stage maps from per-stage confidence statistics.

    Args:
        stage_maps: [B, 3, H, W]
        kind: ``otsu``, ``contrast`` or ``entropy``.
        temperature: softmax sharpening; larger values give more weight to the
            most confident stage.
    """
    b = stage_maps.shape[0]
    scores = torch.zeros(b, 3, device=stage_maps.device, dtype=stage_maps.dtype)
    maps_np = stage_maps.detach().cpu().numpy()
    for bi in range(b):
        raw = []
        for si in range(3):
            m = maps_np[bi, si]
            if kind == "otsu":
                raw.append(_otsu_eta_np(m))
            elif kind == "contrast":
                raw.append(float(m.std()))
            elif kind == "entropy":
                raw.append(-_entropy_np(m))
            else:
                raise ValueError(f"Unknown adaptive fusion kind {kind!r}")
        raw = np.asarray(raw, dtype=np.float64)
        raw = raw - raw.min()
        if raw.max() > 1e-12:
            raw = raw / raw.max()
        scores[bi] = torch.from_numpy(raw).to(device=stage_maps.device, dtype=stage_maps.dtype)
    return torch.softmax(scores * float(temperature), dim=1)


def fuse_layers(
    layers: torch.Tensor,
    method: str = "mean",
    params: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """Fuse [B, 13, H, W] layer predictions into [B, H, W].

    Supported methods cover group A in docs/README.md:
      * A0 ``mean``
      * A1-A3 ``low`` / ``mid`` / ``high``
      * A4 ``low_mid`` / ``mid_high`` / ``low_high``
      * A5 ``weighted_stage`` with params ``weights=(wL, wM, wH)``
      * A6 ``adaptive_otsu`` / ``adaptive_contrast`` / ``adaptive_entropy``
      * A7 ``norm_minmax_mean`` / ``norm_zscore_mean``
      * A8 ``max`` / ``geomean`` / ``median``
    """
    params = params or {}
    layers = _check_layers(layers)

    if method == "mean":
        return layers.mean(dim=1)
    if method in STAGE_SLICES:
        return layers[:, STAGE_SLICES[method]].mean(dim=1)
    if method in STAGE_PAIRS:
        a, b = STAGE_PAIRS[method]
        return torch.cat(
            [layers[:, STAGE_SLICES[a]], layers[:, STAGE_SLICES[b]]], dim=1
        ).mean(dim=1)
    if method == "max":
        return layers.max(dim=1).values
    if method == "median":
        return layers.median(dim=1).values
    if method == "geomean":
        return torch.exp(torch.log(layers.clamp_min(1e-6)).mean(dim=1))
    if method in {"norm_minmax_mean", "norm_zscore_mean"}:
        if method == "norm_minmax_mean":
            mn = layers.amin(dim=(-2, -1), keepdim=True)
            mx = layers.amax(dim=(-2, -1), keepdim=True)
            normed = (layers - mn) / (mx - mn).clamp_min(1e-6)
        else:
            mean = layers.mean(dim=(-2, -1), keepdim=True)
            std = layers.std(dim=(-2, -1), keepdim=True)
            normed = (layers - mean) / std.clamp_min(1e-6)
        return normed.mean(dim=1)
    if method == "weighted_stage":
        weights = params.get("weights", (1.0, 1.0, 1.0))
        w = torch.as_tensor(weights, device=layers.device, dtype=layers.dtype).view(1, 3, 1, 1)
        stage_maps = _stage_maps(layers)
        return (stage_maps * w).sum(dim=1) / w.sum()
    if method.startswith("adaptive_"):
        kind = method[len("adaptive_") :]
        stage_maps = _stage_maps(layers)
        w = _adaptive_weights(
            stage_maps, kind, float(params.get("temperature", 10.0))
        ).view(-1, 3, 1, 1)
        return (stage_maps * w).sum(dim=1)
    raise ValueError(f"Unknown fusion method {method!r}")


# ---------------------------------------------------------------------------
# Threshold variants (group B)
# ---------------------------------------------------------------------------
def _scale_to_uint8(x: torch.Tensor) -> Tuple[np.ndarray, float, float]:
    x = x.detach().float()
    mn = float(x.min().item())
    mx = float(x.max().item())
    if mx - mn < 1e-12:
        return np.zeros(tuple(x.shape), dtype=np.uint8), mn, mx
    u8 = ((x - mn) / (mx - mn) * 255.0).clamp(0, 255).byte().cpu().numpy()
    return u8, mn, mx


def _otsu_threshold_torch(fused: torch.Tensor) -> torch.Tensor:
    # Reuse the baseline implementation (drop_least=0.05, OpenCV Otsu).
    return segutils.otsus(fused)[0]


def _multi_otsu_np(x: np.ndarray, classes: int = 3) -> float:
    """Threshold separating the highest class from the lower classes."""
    if classes != 3:
        raise ValueError("Only 3-class multi-Otsu is implemented")
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    mn, mx = float(x.min()), float(x.max())
    if mx - mn < 1e-12:
        return mx
    x = (x - mn) / (mx - mn)
    hist, edges = np.histogram(x, bins=256, range=(0.0, 1.0))
    p = hist.astype(np.float64) / max(hist.sum(), 1)
    centers = (edges[:-1] + edges[1:]) * 0.5
    omega = np.cumsum(p)
    mu = np.cumsum(p * centers)
    mu_t = mu[-1]
    total_var = float(np.sum(p * (centers - mu_t) ** 2))
    if total_var < 1e-12:
        return mx
    best_score, best_t2 = -1.0, 255
    # j indexes the upper threshold bin in 1..255
    for i in range(1, 255):
        w0 = omega[i - 1]
        if w0 <= 1e-12:
            continue
        m0 = mu[i - 1]
        j = np.arange(i + 1, 256)
        w1 = omega[j - 1] - omega[i - 1]
        m1 = mu[j - 1] - mu[i - 1]
        w2 = 1.0 - omega[j - 1]
        m2 = mu_t - mu[j - 1]
        score = np.full(j.shape, w0 * (m0 / w0 - mu_t) ** 2, dtype=np.float64)
        valid = w1 > 1e-12
        score[valid] += (m1[valid] ** 2) / w1[valid] - 2 * mu_t * m1[valid] + w1[valid] * mu_t**2
        valid = w2 > 1e-12
        score[valid] += (m2[valid] ** 2) / w2[valid] - 2 * mu_t * m2[valid] + w2[valid] * mu_t**2
        k = int(np.argmax(score))
        if score[k] > best_score:
            best_score = float(score[k])
            best_t2 = int(j[k])
    t = float(edges[best_t2])  # edges has 257 values
    return mn + t * (mx - mn)


def compute_threshold(
    fused: torch.Tensor,
    method: str,
    params: Optional[Dict[str, Any]] = None,
    support_fg_ratio: Optional[float] = None,
) -> Optional[torch.Tensor]:
    """Return a per-sample threshold [B], or ``None`` for local thresholding."""
    params = params or {}
    if fused.ndim == 2:
        fused = fused.unsqueeze(0)
    b = fused.shape[0]
    mean_t = fused.mean(dim=(-2, -1))

    if method == "mean":
        return mean_t
    if method == "otsus":
        return _otsu_threshold_torch(fused)
    if method == "pred_mean":
        return torch.maximum(_otsu_threshold_torch(fused), mean_t)
    if method == "multi_otsu":
        values = []
        for i in range(b):
            values.append(
                _multi_otsu_np(
                    fused[i].detach().cpu().numpy(),
                    classes=int(params.get("classes", 3)),
                )
            )
        return torch.tensor(values, device=fused.device, dtype=fused.dtype)
    if method == "quantile_support":
        ratio = float(support_fg_ratio if support_fg_ratio is not None else 0.5)
        ratio = min(max(1.0 - ratio, 0.0), 1.0)
        q = torch.quantile(fused.flatten(1), float(ratio), dim=1)
        return q
    if method == "lambda":
        lam = float(params.get("lambda", 0.5))
        otsu = _otsu_threshold_torch(fused)
        return lam * otsu + (1.0 - lam) * mean_t
    if method == "adaptive_local":
        return None
    raise ValueError(f"Unknown threshold method {method!r}")


def apply_threshold(
    fused: torch.Tensor,
    method: str,
    params: Optional[Dict[str, Any]] = None,
    support_fg_ratio: Optional[float] = None,
) -> torch.Tensor:
    """Return a binary prediction [B, H, W] (float)."""
    params = params or {}
    if fused.ndim == 2:
        fused = fused.unsqueeze(0)
    if method == "adaptive_local":
        block_size = int(params.get("block_size", 31))
        c = float(params.get("C", 0.0))
        if block_size % 2 == 0:
            block_size += 1
        preds = []
        for i in range(fused.shape[0]):
            u8, _, _ = _scale_to_uint8(fused[i])
            pred = cv2.adaptiveThreshold(
                u8,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                block_size,
                c,
            )
            preds.append(torch.from_numpy((pred > 0).astype(np.float32)))
        return torch.stack(preds, dim=0).to(fused.device)
    thresh = compute_threshold(fused, method, params, support_fg_ratio)
    if thresh is None:
        raise ValueError(f"Method {method!r} did not produce a scalar threshold")
    return (fused > thresh.view(-1, 1, 1)).float()


# ---------------------------------------------------------------------------
# Post-processing variants (group C)
# ---------------------------------------------------------------------------
def _pred_to_np(pred: torch.Tensor) -> np.ndarray:
    if pred.ndim == 3:
        pred = pred[0]
    return (pred.detach().cpu().numpy() > 0.5).astype(np.uint8)


def _np_to_pred(mask: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(mask.astype(np.float32)).to(device).unsqueeze(0)


def _morph_np(mask: np.ndarray, ops: Sequence[str], kernel: int = 3, min_area_frac: float = 0.005) -> np.ndarray:
    m = mask.astype(np.uint8)
    k = np.ones((int(kernel), int(kernel)), np.uint8)
    for op in ops:
        if op == "open":
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
        elif op == "close":
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
        elif op == "remove_small":
            n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
            min_area = max(1, int(min_area_frac * m.shape[0] * m.shape[1]))
            for lab in range(1, n):
                if stats[lab, cv2.CC_STAT_AREA] < min_area:
                    m[labels == lab] = 0
        elif op == "fill_holes":
            contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filled = np.zeros_like(m)
            cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
            m = filled
        else:
            raise ValueError(f"Unknown morphology operation {op!r}")
    return m


def apply_morph(
    pred: torch.Tensor,
    ops: Sequence[str],
    kernel: int = 3,
    min_area_frac: float = 0.005,
) -> torch.Tensor:
    m = _pred_to_np(pred)
    return _np_to_pred(_morph_np(m, ops, kernel, min_area_frac), pred.device)


def apply_guided(
    q_img: torch.Tensor,
    fused: torch.Tensor,
    threshold_method: str,
    threshold_params: Optional[Dict[str, Any]],
    support_fg_ratio: float,
    radius: int = 8,
    eps: float = 1e-3,
) -> torch.Tensor:
    if q_img.ndim == 3:
        q_img = q_img.unsqueeze(0)
    refined = segutils.guided_filter(q_img, fused, radius=radius, eps=eps)
    return apply_threshold(refined, threshold_method, threshold_params, support_fg_ratio)


def apply_guided_morph(
    q_img: torch.Tensor,
    fused: torch.Tensor,
    threshold_method: str,
    threshold_params: Optional[Dict[str, Any]],
    support_fg_ratio: float,
    radius: int = 8,
    eps: float = 1e-3,
    ops: Sequence[str] = ("open", "close", "remove_small", "fill_holes"),
    kernel: int = 3,
    min_area_frac: float = 0.005,
) -> torch.Tensor:
    pred = apply_guided(
        q_img, fused, threshold_method, threshold_params, support_fg_ratio, radius, eps
    )
    return apply_morph(pred, ops, kernel, min_area_frac)


def _threshold_fn(
    method: str,
    params: Optional[Dict[str, Any]],
    support_fg_ratio: float,
):
    def inner(fused: torch.Tensor, _s_masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        thresh = compute_threshold(fused, method, params, support_fg_ratio)
        if thresh is None:
            raise ValueError("CRF requires a scalar threshold method (not adaptive_local)")
        return thresh

    return inner


def apply_crf(
    q_img: torch.Tensor,
    fused: torch.Tensor,
    threshold_method: str,
    threshold_params: Optional[Dict[str, Any]],
    support_fg_ratio: float,
    iterations: int = 5,
) -> torch.Tensor:
    if q_img.ndim == 3:
        q_img = q_img.unsqueeze(0)
    crf = segutils.CRF(
        gaussian_stdxy=(1, 1),
        gaussian_compat=2,
        bilateral_stdxy=(35, 35),
        bilateral_compat=1,
        stdrgb=(13, 13, 13),
    )
    thresh_fn = _threshold_fn(threshold_method, threshold_params, support_fg_ratio)
    q = crf.iterrefine(iterations, q_img, fused, thresh_fn)
    return q.argmax(1).float()


def apply_postproc(
    fused: torch.Tensor,
    q_img: torch.Tensor,
    postproc: str,
    postproc_params: Optional[Dict[str, Any]],
    threshold_method: str,
    threshold_params: Optional[Dict[str, Any]],
    support_fg_ratio: float,
    iterations_override: Optional[int] = None,
) -> torch.Tensor:
    """Apply a non-dynamic post-processing method."""
    params = postproc_params or {}
    if postproc == "none":
        return apply_threshold(fused, threshold_method, threshold_params, support_fg_ratio)
    if postproc == "crf":
        iters = int(params.get("iterations", 5) if iterations_override is None else iterations_override)
        return apply_crf(
            q_img, fused, threshold_method, threshold_params, support_fg_ratio, iters
        )
    if postproc == "guided":
        return apply_guided(
            q_img,
            fused,
            threshold_method,
            threshold_params,
            support_fg_ratio,
            radius=int(params.get("radius", 8)),
            eps=float(params.get("eps", 1e-3)),
        )
    if postproc == "morph":
        base = apply_threshold(fused, threshold_method, threshold_params, support_fg_ratio)
        return apply_morph(
            base,
            params.get("ops", ("open", "close")),
            kernel=int(params.get("kernel", 3)),
            min_area_frac=float(params.get("min_area_frac", 0.005)),
        )
    if postproc == "guided_morph":
        return apply_guided_morph(
            q_img,
            fused,
            threshold_method,
            threshold_params,
            support_fg_ratio,
            radius=int(params.get("radius", 8)),
            eps=float(params.get("eps", 1e-3)),
            ops=params.get("ops", ("open", "close", "remove_small", "fill_holes")),
            kernel=int(params.get("kernel", 3)),
            min_area_frac=float(params.get("min_area_frac", 0.005)),
        )
    raise ValueError(f"Unknown post-processing method {postproc!r}")


def mask_iou(pred: torch.Tensor, gt: torch.Tensor) -> float:
    p = (pred.detach() > 0.5).float()
    g = (gt.detach() > 0.5).float()
    inter = float((p * g).sum().item())
    union = float(((p + g) > 0).sum().item())
    return inter / (union + 1e-8)


def dynamic_choice(
    pseudo_layers: torch.Tensor,
    s_img: torch.Tensor,
    s_mask: torch.Tensor,
    s_mask_aug_mean: float,
    fusion: str,
    fusion_params: Optional[Dict[str, Any]],
    threshold_method: str,
    threshold_params: Optional[Dict[str, Any]],
    postproc: str,
    postproc_params: Optional[Dict[str, Any]],
) -> bool:
    """Algorithm 1's pseudo-episode gate for one post-processing method."""
    if pseudo_layers.ndim == 3:
        pseudo_layers = pseudo_layers.unsqueeze(0)  # [k, 13, H, W]
    k = pseudo_layers.shape[0]
    votes = 0
    for i in range(k):
        pfused = upsample_map(fuse_layers(pseudo_layers[i], fusion, fusion_params))
        base = apply_threshold(
            pfused, threshold_method, threshold_params, s_mask_aug_mean
        )
        # Algorithm 1 uses one CRF iteration for the pseudo episode; the same
        # early-stopping spirit is used for guided/morphological variants.
        pp = apply_postproc(
            pfused,
            s_img[i].unsqueeze(0),
            postproc,
            postproc_params,
            threshold_method,
            threshold_params,
            s_mask_aug_mean,
            iterations_override=1 if postproc == "crf" else None,
        )
        if mask_iou(pp, s_mask[i].unsqueeze(0)) > mask_iou(base, s_mask[i].unsqueeze(0)):
            votes += 1
    return votes > k / 3.0


def apply_config(
    layers: torch.Tensor,
    q_img: torch.Tensor,
    s_mask: torch.Tensor,
    s_mask_aug_mean: float,
    pseudo_layers: torch.Tensor,
    s_img: torch.Tensor,
    cfg: Dict[str, Any],
) -> torch.Tensor:
    """Apply one experimental configuration to one episode."""
    if q_img.ndim == 3:
        q_img = q_img.unsqueeze(0)
    fusion = cfg["fusion"]
    fusion_params = cfg.get("fusion_params") or {}
    threshold = cfg["threshold"]
    threshold_params = cfg.get("threshold_params") or {}
    postproc = cfg["postproc"]
    postproc_params = cfg.get("postproc_params") or {}
    support_ratio = float(s_mask.mean().item())

    if layers.ndim == 3:
        layers = layers.unsqueeze(0)
    fused = upsample_map(fuse_layers(layers, fusion, fusion_params))

    if postproc == "none":
        return apply_threshold(fused, threshold, threshold_params, support_ratio)

    if postproc.endswith("_dynamic"):
        method = postproc[: -len("_dynamic")]
        good = dynamic_choice(
            pseudo_layers,
            s_img,
            s_mask,
            s_mask_aug_mean,
            fusion,
            fusion_params,
            threshold,
            threshold_params,
            method,
            postproc_params,
        )
        if good:
            return apply_postproc(
                fused,
                q_img,
                method,
                postproc_params,
                threshold,
                threshold_params,
                support_ratio,
            )
        return apply_threshold(fused, threshold, threshold_params, support_ratio)

    return apply_postproc(
        fused,
        q_img,
        postproc,
        postproc_params,
        threshold,
        threshold_params,
        support_ratio,
    )


# ---------------------------------------------------------------------------
# Experiment configuration lists
# ---------------------------------------------------------------------------
def _cfg(
    name: str,
    group: str,
    fusion: str = "mean",
    fusion_params: Optional[Dict[str, Any]] = None,
    threshold: str = "pred_mean",
    threshold_params: Optional[Dict[str, Any]] = None,
    postproc: str = "none",
    postproc_params: Optional[Dict[str, Any]] = None,
    note: str = "",
) -> Dict[str, Any]:
    return {
        "name": name,
        "group": group,
        "fusion": fusion,
        "fusion_params": fusion_params or {},
        "threshold": threshold,
        "threshold_params": threshold_params or {},
        "postproc": postproc,
        "postproc_params": postproc_params or {},
        "note": note,
    }


def group_a_configs() -> List[Dict[str, Any]]:
    cfgs = [
        _cfg("A0_mean", "A", note="baseline fusion (13-layer mean)"),
        _cfg("A1_low", "A", fusion="low", note="only Low (layers 1-4)"),
        _cfg("A2_mid", "A", fusion="mid", note="only Mid (layers 5-10)"),
        _cfg("A3_high", "A", fusion="high", note="only High (layers 11-13)"),
        _cfg("A4_low_mid", "A", fusion="low_mid", note="Low + Mid"),
        _cfg("A4_mid_high", "A", fusion="mid_high", note="Mid + High"),
        _cfg("A4_low_high", "A", fusion="low_high", note="Low + High"),
        _cfg("A6_adaptive_otsu", "A", fusion="adaptive_otsu",
             fusion_params={"temperature": 10.0},
             note="per-episode stage weights from Otsu separability"),
        _cfg("A6_adaptive_contrast", "A", fusion="adaptive_contrast",
             fusion_params={"temperature": 10.0},
             note="per-episode stage weights from map contrast"),
        _cfg("A6_adaptive_entropy", "A", fusion="adaptive_entropy",
             fusion_params={"temperature": 10.0},
             note="per-episode stage weights from histogram entropy"),
        _cfg("A7_norm_minmax", "A", fusion="norm_minmax_mean",
             note="per-layer min-max normalisation before mean"),
        _cfg("A7_norm_zscore", "A", fusion="norm_zscore_mean",
             note="per-layer z-score normalisation before mean"),
        _cfg("A8_max", "A", fusion="max", note="layer-wise max"),
        _cfg("A8_geomean", "A", fusion="geomean", note="layer-wise geometric mean"),
        _cfg("A8_median", "A", fusion="median", note="layer-wise median"),
    ]
    for wl in (0.5, 1.0, 2.0):
        for wm in (0.5, 1.0, 2.0):
            for wh in (0.5, 1.0, 2.0):
                cfgs.append(
                    _cfg(
                        f"A5_w{wl:g}_{wm:g}_{wh:g}",
                        "A",
                        fusion="weighted_stage",
                        fusion_params={"weights": (wl, wm, wh)},
                        note=f"stage weights L/M/H={wl:g}/{wm:g}/{wh:g}",
                    )
                )
    return cfgs


def group_b_configs() -> List[Dict[str, Any]]:
    cfgs = [
        _cfg("B0_pred_mean", "B", threshold="pred_mean", note="max(mean, Otsu) baseline"),
        _cfg("B1_mean", "B", threshold="mean", note="mean threshold"),
        _cfg("B2_otsus", "B", threshold="otsus", note="pure Otsu"),
        _cfg("B3_multi_otsu", "B", threshold="multi_otsu",
             threshold_params={"classes": 3}, note="3-class Otsu, highest class"),
        _cfg("B4_quantile_support", "B", threshold="quantile_support",
             note="quantile at support foreground ratio"),
        _cfg("B5_adaptive_local", "B", threshold="adaptive_local",
             threshold_params={"block_size": 31, "C": 0.0},
             note="local Gaussian adaptive threshold"),
    ]
    for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
        cfgs.append(
            _cfg(
                f"B6_lambda{lam:g}",
                "B",
                threshold="lambda",
                threshold_params={"lambda": lam},
                note=f"lambda*Otsu + (1-lambda)*mean, lambda={lam:g}",
            )
        )
    return cfgs


def group_c_configs() -> List[Dict[str, Any]]:
    morph_full_ops = ("open", "close", "remove_small", "fill_holes")
    return [
        _cfg("C1_off", "C", postproc="none", note="no post-processing"),
        _cfg("C0_crf_dynamic", "C", postproc="crf_dynamic",
             postproc_params={"iterations": 5},
             note="DenseCRF with Algorithm 1 gate (baseline)"),
        _cfg("C2_crf_always", "C", postproc="crf",
             postproc_params={"iterations": 5}, note="DenseCRF always on"),
        _cfg("C3_guided", "C", postproc="guided",
             postproc_params={"radius": 8, "eps": 1e-3},
             note="guided filter before thresholding"),
        _cfg("C4_morph_open", "C", postproc="morph",
             postproc_params={"ops": ["open"], "kernel": 3}, note="morphological opening"),
        _cfg("C4_morph_close", "C", postproc="morph",
             postproc_params={"ops": ["close"], "kernel": 3}, note="morphological closing"),
        _cfg("C4_morph_open_close", "C", postproc="morph",
             postproc_params={"ops": ["open", "close"], "kernel": 3},
             note="opening then closing"),
        _cfg("C4_morph_remove_small", "C", postproc="morph",
             postproc_params={"ops": ["remove_small"], "min_area_frac": 0.005},
             note="remove components < 0.5% of image"),
        _cfg("C4_morph_fill_holes", "C", postproc="morph",
             postproc_params={"ops": ["fill_holes"]}, note="fill holes"),
        _cfg("C4_morph_full", "C", postproc="morph",
             postproc_params={"ops": list(morph_full_ops), "kernel": 3,
                              "min_area_frac": 0.005},
             note="open + close + remove small + fill holes"),
        _cfg("C5_guided_morph", "C", postproc="guided_morph",
             postproc_params={"radius": 8, "eps": 1e-3, "ops": list(morph_full_ops),
                              "kernel": 3, "min_area_frac": 0.005},
             note="guided filter then threshold then morphology"),
        _cfg("C6_guided_dynamic", "C", postproc="guided_dynamic",
             postproc_params={"radius": 8, "eps": 1e-3},
             note="guided filter with Algorithm 1 gate"),
        _cfg("C6_morph_full_dynamic", "C", postproc="morph_dynamic",
             postproc_params={"ops": list(morph_full_ops), "kernel": 3,
                              "min_area_frac": 0.005},
             note="morphology with Algorithm 1 gate"),
    ]


def configs_for_group(group: str) -> List[Dict[str, Any]]:
    group = group.upper()
    if group == "A":
        return group_a_configs()
    if group == "B":
        return group_b_configs()
    if group == "C":
        return group_c_configs()
    raise ValueError(f"Unknown group {group!r}")


def is_crf_config(cfg: Dict[str, Any]) -> bool:
    return cfg["postproc"] in {"crf", "crf_dynamic"}


# ---------------------------------------------------------------------------
# Metric accumulation
# ---------------------------------------------------------------------------
class MetricAccumulator:
    """Accumulate mIoU / FB-IoU / foreground ratio exactly like eval/logger.py."""

    def __init__(self, nclass: int):
        self.nclass = int(nclass)
        self.inter = {}
        self.union = {}
        self.count = {}
        self.fg_pred = {}
        self.fg_gt = {}
        self.seconds = {}
        self.per_class = {}

    def _ensure(self, name: str) -> None:
        if name not in self.inter:
            self.inter[name] = np.zeros((self.nclass, 2), dtype=np.float64)
            self.union[name] = np.zeros((self.nclass, 2), dtype=np.float64)
            self.count[name] = np.zeros(self.nclass, dtype=np.int64)
            self.fg_pred[name] = 0.0
            self.fg_gt[name] = 0.0
            self.seconds[name] = 0.0
            self.per_class[name] = np.zeros(self.nclass, dtype=np.float64)

    def update(self, name: str, pred: torch.Tensor, gt: torch.Tensor, class_id: int, seconds: float = 0.0) -> None:
        self._ensure(name)
        if pred.ndim == 3:
            pred = pred[0]
        if gt.ndim == 3:
            gt = gt[0]
        p = (pred.detach().cpu().numpy() > 0.5).astype(np.uint8)
        g = (gt.detach().cpu().numpy() > 0.5).astype(np.uint8)
        inter_fg = float(np.logical_and(p == 1, g == 1).sum())
        union_fg = float(np.logical_or(p == 1, g == 1).sum())
        inter_bg = float(np.logical_and(p == 0, g == 0).sum())
        union_bg = float(np.logical_or(p == 0, g == 0).sum())
        cid = int(class_id)
        if 0 <= cid < self.nclass:
            self.inter[name][cid, 0] += inter_bg
            self.inter[name][cid, 1] += inter_fg
            self.union[name][cid, 0] += union_bg
            self.union[name][cid, 1] += union_fg
            self.count[name][cid] += 1
            if union_fg > 0:
                self.per_class[name][cid] += inter_fg / union_fg
        self.fg_pred[name] += float(p.mean())
        self.fg_gt[name] += float(g.mean())
        self.seconds[name] += float(seconds)

    def summarize(self, class_ids: Sequence[int]) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        class_ids = [int(c) for c in class_ids]
        for name in self.inter:
            inter = self.inter[name][class_ids]
            union = self.union[name][class_ids]
            n = self.count[name][class_ids].sum()
            if n == 0:
                continue
            iou_fg = inter[:, 1] / np.maximum(union[:, 1], 1.0)
            miou = float(iou_fg.mean() * 100.0) if len(iou_fg) else 0.0
            fb_fg = inter[:, 1].sum() / max(union[:, 1].sum(), 1.0)
            fb_bg = inter[:, 0].sum() / max(union[:, 0].sum(), 1.0)
            fb = float((fb_bg + fb_fg) * 0.5 * 100.0)
            out[name] = {
                "mIoU": miou,
                "FB-IoU": fb,
                "FG_pred": float(self.fg_pred[name] / n * 100.0),
                "FG_gt": float(self.fg_gt[name] / n * 100.0),
                "n_episodes": int(n),
                "seconds": float(self.seconds[name]),
                "per_class_iou": {
                    str(c): float(iou_fg[i] * 100.0) for i, c in enumerate(class_ids)
                },
            }
        return out


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------
def class_id_for_index(dataset: Any, idx: int) -> int:
    if getattr(dataset, "benchmark", "") == "fss":
        return dataset.categories.index(dataset.img_metadata[idx].split("/")[-2]) + 760
    return int(idx % len(dataset.class_ids))


def nclass_for_dataset(dataset: Any) -> int:
    return int(max(dataset.class_ids)) + 1


def class_ids_for_dataset(dataset: Any) -> List[int]:
    return list(dataset.class_ids)


def first_index_per_class(dataset: Any) -> Dict[int, int]:
    first: Dict[int, int] = {}
    for idx in range(len(dataset)):
        cid = class_id_for_index(dataset, idx)
        if cid not in first:
            first[cid] = idx
    return first


def select_indices(n: int, max_episodes: Optional[int]) -> List[int]:
    if max_episodes is None or max_episodes <= 0 or max_episodes >= n:
        return list(range(n))
    step = n / float(max_episodes)
    idxs = sorted({min(n - 1, int(round(i * step))) for i in range(max_episodes)})
    return idxs


def select_subset(indices: Sequence[int], max_items: Optional[int]) -> List[int]:
    """Stratified subset of an explicit list of episode indices."""
    indices = list(indices)
    if max_items is None or max_items <= 0 or max_items >= len(indices):
        return indices
    n = len(indices)
    step = n / float(max_items)
    positions = sorted({min(n - 1, int(round(i * step))) for i in range(max_items)})
    return [indices[p] for p in positions]
