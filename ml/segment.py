"""Estimate the damaged region inside a building crop.

A trained U-Net checkpoint is used when `models/unet_damage.pt` exists.
Otherwise the mask comes from pre/post change, or from local texture when only
the post-disaster image is available. xBD does not ship pixel damage masks, so
the fallback is an estimate for ranking, not a surveyed damage outline.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
import torch
from PIL import Image

try:
    from .preprocess import open_rgb, register_pair, to_gray
    from .unet import build_unet
except ImportError:
    from preprocess import open_rgb, register_pair, to_gray
    from unet import build_unet

MASK_SIZE = 256


def _box_mean(values: np.ndarray, radius: int) -> np.ndarray:
    padded = np.pad(values, radius, mode="reflect")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)
    k = radius * 2 + 1
    total = integral[k:, k:] - integral[:-k, k:] - integral[k:, :-k] + integral[:-k, :-k]
    return total / float(k * k)


def texture_mask(image: Image.Image) -> np.ndarray:
    """Mark high local-variance pixels. Debris and broken roofs score high."""
    gray = to_gray(open_rgb(image, MASK_SIZE))
    mean = _box_mean(gray, 3)
    mean_sq = _box_mean(gray * gray, 3)
    variance = np.clip(mean_sq - mean * mean, 0, None)
    cutoff = float(np.quantile(variance, 0.72))
    return variance >= cutoff


def change_mask(pre: Image.Image, post: Image.Image) -> np.ndarray:
    diff = np.abs(to_gray(pre) - to_gray(post))
    cutoff = max(0.12, float(diff.mean() + diff.std()))
    return diff >= cutoff


def unet_mask(model: torch.nn.Module, image: Image.Image, device: torch.device) -> np.ndarray:
    array = np.asarray(open_rgb(image, MASK_SIZE), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std
    with torch.inference_mode():
        probs = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
    return probs >= 0.5


def load_unet(checkpoint: Path, device: torch.device) -> torch.nn.Module | None:
    if not checkpoint.is_file():
        return None
    model = build_unet().to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state)
    model.eval()
    return model


def estimate_damage_mask(
    post: Image.Image,
    pre: Image.Image | None = None,
    unet: torch.nn.Module | None = None,
    device: torch.device | None = None,
) -> dict:
    shift = (0, 0)
    aligned_pre = None
    if unet is not None and device is not None:
        mask = unet_mask(unet, post, device)
        method = "unet"
    elif pre is not None:
        aligned_pre, shift = register_pair(pre, post, MASK_SIZE)
        post_r = open_rgb(post, MASK_SIZE)
        mask = change_mask(aligned_pre, post_r)
        method = "pre-post-change"
    else:
        mask = texture_mask(post)
        method = "texture"
    area = float(mask.mean()) if mask.size else 0.0
    return {
        "mask": mask,
        "damage_area_ratio": round(area, 4),
        "method": method,
        "registration_shift": {"dy": int(shift[0]), "dx": int(shift[1])},
        "aligned_pre": aligned_pre,
    }


def mask_png_base64(mask: np.ndarray) -> str:
    image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")
