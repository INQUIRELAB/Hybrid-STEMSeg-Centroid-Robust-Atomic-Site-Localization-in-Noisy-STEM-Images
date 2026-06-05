"""Build centroid-aware training targets from atom coordinates or binary masks."""

from __future__ import annotations

import numpy as np


def gaussian_heatmap(
    h: int,
    w: int,
    centers_yx: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """(H, W) heatmap as max of 2D Gaussians at each (y, x)."""
    hm = np.zeros((h, w), dtype=np.float32)
    if centers_yx is None or len(centers_yx) == 0:
        return hm
    yy, xx = np.ogrid[:h, :w]
    sig2 = 2.0 * float(sigma) ** 2
    for y0, x0 in centers_yx:
        cy, cx = float(y0), float(x0)
        g = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / sig2)
        hm = np.maximum(hm, g)
    return hm


def build_offset_targets(
    h: int,
    w: int,
    centers_yx: np.ndarray,
    sigma: float,
    supervise_radius: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
  Returns offset (2, H, W) as (dy, dx) to nearest center, and mask (H, W) where loss applies.
  Supervision on pixels within ``supervise_radius`` of a center (default 2*sigma).
    """
    off = np.zeros((2, h, w), dtype=np.float32)
    off_mask = np.zeros((h, w), dtype=np.float32)
    if centers_yx is None or len(centers_yx) == 0:
        return off, off_mask
    rad = float(supervise_radius if supervise_radius is not None else max(2.0 * sigma, 4.0))
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    for cy, cx in centers_yx:
        cy, cx = float(cy), float(cx)
        dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
        region = dist2 <= rad * rad
        off[0, region] = cy - yy[region]
        off[1, region] = cx - xx[region]
        off_mask[region] = 1.0
    return off, off_mask


def pack_centroid_set(
    centers_yx: np.ndarray,
    h: int,
    w: int,
    max_atoms: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalized (y, x) in [0,1] and validity mask, shape (max_atoms, 2) and (max_atoms,)."""
    coords = np.zeros((max_atoms, 2), dtype=np.float32)
    valid = np.zeros((max_atoms,), dtype=np.float32)
    if centers_yx is None:
        return coords, valid
    n = min(len(centers_yx), max_atoms)
    for i in range(n):
        cy, cx = float(centers_yx[i, 0]), float(centers_yx[i, 1])
        coords[i, 0] = cy / max(h - 1, 1)
        coords[i, 1] = cx / max(w - 1, 1)
        valid[i] = 1.0
    return coords, valid


def centers_from_patch_com(
    xy_com: np.ndarray,
    top: int,
    left: int,
    patch_h: int,
    patch_w: int,
) -> np.ndarray:
    """Full-image (y,x) COM/atms -> patch pixel coordinates (N, 2)."""
    if xy_com is None or len(xy_com) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    y_all, x_all = xy_com[:, 0], xy_com[:, 1]
    inside = (
        (y_all >= top)
        & (y_all < top + patch_h)
        & (x_all >= left)
        & (x_all < left + patch_w)
    )
    if not np.any(inside):
        return np.zeros((0, 2), dtype=np.float32)
    yp = y_all[inside] - top
    xp = x_all[inside] - left
    return np.stack([yp, xp], axis=1).astype(np.float32)


def build_centroid_targets_from_centers(
    centers_yx: np.ndarray,
    h: int,
    w: int,
    sigma: float,
    max_atoms: int,
) -> dict[str, np.ndarray]:
    hm = gaussian_heatmap(h, w, centers_yx, sigma)
    off, off_mask = build_offset_targets(h, w, centers_yx, sigma)
    coords, valid = pack_centroid_set(centers_yx, h, w, max_atoms)
    return {
        "heatmap": hm[np.newaxis, ...],
        "offset": off,
        "offset_mask": off_mask[np.newaxis, ...],
        "centroids": coords,
        "centroid_valid": valid,
    }


def build_centroid_targets_from_mask(
    mask_hw: np.ndarray,
    sigma: float,
    max_atoms: int,
) -> dict[str, np.ndarray]:
    """Fallback when shard has no stored coordinates: peaks from binary mask."""
    from skimage.measure import label, regionprops

    m = (mask_hw > 0.5).astype(np.uint8)
    lbl = label(m)
    cents = np.array([p.centroid for p in regionprops(lbl)], dtype=np.float32)
    if len(cents) == 0:
        return build_centroid_targets_from_centers(cents, m.shape[0], m.shape[1], sigma, max_atoms)
    return build_centroid_targets_from_centers(cents, m.shape[0], m.shape[1], sigma, max_atoms)
