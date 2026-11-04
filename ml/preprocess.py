"""Resize and align a pre/post disaster image pair.

Registration uses phase correlation and refuses large shifts, which are usually
a failed match rather than a real camera offset.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def open_rgb(image: Image.Image, size: int = 256) -> Image.Image:
    return image.convert("RGB").resize((size, size), Image.BILINEAR)


def to_gray(image: Image.Image) -> np.ndarray:
    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    return gray


def _wrap_shift(peak: int, length: int) -> int:
    if peak > length // 2:
        return peak - length
    return peak


def translation(reference: np.ndarray, moving: np.ndarray) -> tuple[int, int]:
    """Return (dy, dx) that shifts `moving` onto `reference`."""
    ref_fft = np.fft.fft2(reference)
    mov_fft = np.fft.fft2(moving)
    cross = ref_fft * np.conj(mov_fft)
    cross /= np.abs(cross) + 1e-8
    correlation = np.fft.ifft2(cross).real
    peak_y, peak_x = np.unravel_index(int(np.argmax(correlation)), correlation.shape)
    dy = _wrap_shift(int(peak_y), reference.shape[0])
    dx = _wrap_shift(int(peak_x), reference.shape[1])
    limit = max(2, reference.shape[0] // 8)
    if abs(dy) > limit or abs(dx) > limit:
        return 0, 0
    return dy, dx


def shift_image(image: Image.Image, dy: int, dx: int) -> Image.Image:
    if dy == 0 and dx == 0:
        return image
    array = np.asarray(image)
    shifted = np.roll(np.roll(array, dy, axis=0), dx, axis=1)
    return Image.fromarray(shifted)


def register_pair(pre: Image.Image, post: Image.Image, size: int = 256) -> tuple[Image.Image, tuple[int, int]]:
    """Align the pre-disaster image to the post-disaster image."""
    pre_r = open_rgb(pre, size)
    post_r = open_rgb(post, size)
    dy, dx = translation(to_gray(post_r), to_gray(pre_r))
    return shift_image(pre_r, dy, dx), (dy, dx)
