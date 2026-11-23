"""Renovation visualisation.

Deterministic, geometry-preserving renderer: each component's mask is filled
with a procedurally generated material texture at real-world scale (tiles
drawn at their true size using the calibrated ft-per-pixel), then multiplied
by the photo's own luminance so shadows, edges and depth cues of the original
house survive. Openings (windows/doors) are never painted.
"""
from __future__ import annotations

import hashlib

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .regions import Region

KIND_COLORS = {
    "wall": (230, 57, 70),
    "parapet": (244, 162, 97),
    "pillar": (42, 157, 143),
    "balcony": (69, 123, 157),
    "roof_edge": (233, 196, 106),
    "gate": (131, 56, 236),
    "railing": (255, 0, 110),
    "window": (58, 134, 255),
    "door": (251, 86, 7),
}


def hex_to_rgb01(h: str) -> np.ndarray:
    h = (h or "#cccccc").lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float32) / 255.0


def _rng(key: str) -> np.random.Generator:
    return np.random.default_rng(int(hashlib.md5(key.encode()).hexdigest()[:8], 16))


def _noise(h: int, w: int, rng: np.random.Generator, sigma: float) -> np.ndarray:
    n = rng.standard_normal((h, w)).astype(np.float32)
    if sigma > 0:
        n = cv2.GaussianBlur(n, (0, 0), sigma)
        s = n.std()
        if s > 0:
            n /= s
    return n


def _grid(h, w, cell_w, cell_h, stagger=False):
    """Return (cell_index_map, joint_mask) for a running-bond or stack grid."""
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    row = np.floor(ys / cell_h)
    offset = (row % 2) * (cell_w / 2.0) if stagger else 0.0
    xs2 = xs + offset
    col = np.floor(xs2 / cell_w)
    fx = xs2 - col * cell_w
    fy = ys - row * cell_h
    return row, col, fx, fy


def make_texture(pattern: str, color: str, h: int, w: int, px_per_ft: float, key: str, material: dict | None = None):
    """Return (rgb float32 HxWx3 in 0..1, alpha float32 HxW, use_shading bool)."""
    material = material or {}
    rng = _rng(key)
    base = hex_to_rgb01(color)
    rgb = np.ones((h, w, 3), np.float32) * base
    alpha = np.ones((h, w), np.float32)
    ppf = max(px_per_ft, 0.5)

    if pattern == "paint":
        rgb *= (1 + 0.015 * _noise(h, w, rng, 1.0))[..., None]
        return np.clip(rgb, 0, 1), alpha, True

    if pattern == "texture":
        fine = _noise(h, w, rng, 0.8)
        coarse = _noise(h, w, rng, max(ppf * 0.3, 2))
        rgb *= (1 + 0.07 * fine + 0.04 * coarse)[..., None]
        return np.clip(rgb, 0, 1), alpha, True

    if pattern in ("tile", "brick", "acp"):
        if pattern == "tile":
            cw = float(material.get("tile_w_ft", 2.0)) * ppf
            ch = float(material.get("tile_h_ft", 1.0)) * ppf
            stagger, joint_frac, var = False, 0.04, 0.05
            joint_color = np.array([0.80, 0.79, 0.76], np.float32)
        elif pattern == "brick":
            cw, ch = 0.75 * ppf, 0.25 * ppf
            stagger, joint_frac, var = True, 0.12, 0.13
            joint_color = np.array([0.78, 0.75, 0.70], np.float32)
        else:  # acp cassettes
            cw, ch = 4.0 * ppf, 2.0 * ppf
            stagger, joint_frac, var = False, 0.02, 0.02
            joint_color = base * 0.45
        cw, ch = max(cw, 4.0), max(ch, 3.0)
        row, col, fx, fy = _grid(h, w, cw, ch, stagger)
        joint = max(1.0, joint_frac * min(cw, ch))
        jmask = (fx < joint) | (fy < joint)
        # Per-cell tone variation.
        cell_id = (row * 10007 + col).astype(np.int64)
        uniq, inv = np.unique(cell_id, return_inverse=True)
        tones = 1 + var * rng.standard_normal(len(uniq)).astype(np.float32)
        tone = tones[inv].reshape(h, w)
        rgb *= tone[..., None]
        if pattern == "acp":
            rgb *= (1 + 0.06 * (fy / ch - 0.5))[..., None]  # subtle sheen
        if pattern == "brick":
            rgb *= (1 + 0.05 * _noise(h, w, rng, 0.7))[..., None]
        rgb[jmask] = joint_color
        return np.clip(rgb, 0, 1), alpha, True

    if pattern == "stone":
        # Random ashlar: rows of varying height, blocks of varying width.
        out = np.zeros((h, w, 3), np.float32)
        y = 0.0
        while y < h:
            rh = rng.uniform(0.35, 0.9) * ppf
            x = -rng.uniform(0, 1.5) * ppf
            y0, y1 = int(y), int(min(y + rh, h))
            while x < w:
                bw = rng.uniform(0.7, 2.0) * ppf
                x0, x1 = int(max(x, 0)), int(min(x + bw, w))
                # Mostly a brightness change per stone, with a slight warm/cool shift.
                warm = 0.035 * rng.standard_normal()
                tint = base * (1 + 0.13 * rng.standard_normal()) * np.array([1 + warm, 1.0, 1 - warm], np.float32)
                out[y0:y1, x0:x1] = tint
                jw = max(1, int(0.04 * ppf))
                out[y0:y1, x0:min(x0 + jw, w)] = base * 0.45
                x += bw
            jh = max(1, int(0.04 * ppf))
            out[y0:min(y0 + jh, h), :] = base * 0.45
            y += rh
        out *= (1 + 0.08 * _noise(h, w, rng, 0.9))[..., None]
        return np.clip(out, 0, 1), alpha, True

    if pattern == "wood":
        plank = max(float(material.get("panel_w_ft", 0.5)) * ppf, 3.0)
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        row = np.floor(ys / plank)
        uniq, inv = np.unique(row.astype(np.int64), return_inverse=True)
        tones = 1 + 0.08 * rng.standard_normal(len(uniq)).astype(np.float32)
        tone = tones[inv].reshape(h, w)
        grain = np.sin(xs / max(ppf * 0.15, 1.5) + 3 * _noise(h, w, rng, max(ppf * 0.2, 2))) * 0.05
        rgb *= (tone + grain)[..., None]
        rgb[(ys - row * plank) < max(1.0, plank * 0.06)] = base * 0.4
        return np.clip(rgb, 0, 1), alpha, True

    if pattern == "glass":
        # Frosted-clear look: mostly opaque tint so the old grill underneath
        # does not dominate, with a soft vertical reflection gradient.
        alpha[:] = 0.78
        xs = np.linspace(0, 1, w, dtype=np.float32)
        rgb *= (0.92 + 0.12 * np.abs(np.sin(xs * np.pi * 3)))[None, :, None]
        panel = max(3.0 * ppf, 6.0)
        seams = (np.arange(w) % panel) < max(1.0, 0.03 * ppf)
        rgb[:, seams] = np.array([0.55, 0.60, 0.63], np.float32)
        rail = max(2, int(0.15 * ppf))
        rgb[:rail] = np.array([0.78, 0.80, 0.82], np.float32)
        alpha[:rail] = 1.0
        return np.clip(rgb, 0, 1), alpha, False

    if pattern == "metal":
        alpha[:] = 0.0
        spacing = max(0.35 * ppf, 5.0)
        bar = max(1.5, 0.06 * ppf)
        xs = np.arange(w, dtype=np.float32)
        bars = (xs % spacing) < bar
        alpha[:, bars] = 1.0
        rail = max(2, int(0.15 * ppf))
        alpha[:rail] = 1.0
        alpha[-max(1, rail // 2):] = 1.0
        shade = 1 + 0.15 * np.cos((xs % spacing) / max(bar, 1) * np.pi)
        rgb *= shade[None, :, None]
        return np.clip(rgb, 0, 1), alpha, False

    raise ValueError(f"Unknown texture pattern {pattern!r}")


def render_design(
    image_rgb: np.ndarray,
    regions: list[Region],
    masks: dict[str, np.ndarray],
    assignments: dict[str, dict],
    catalog: dict[str, dict],
    ft_per_px: float,
    shading_strength: float = 0.9,
) -> np.ndarray:
    img = image_rgb.astype(np.float32) / 255.0
    out = img.copy()
    lum = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    px_per_ft = 1.0 / max(ft_per_px, 1e-6)

    # Paint low-priority surfaces first so that feathered edges of foreground
    # elements (pillars, railings) land on top.
    from .regions import PRIORITY
    ordered = sorted([r for r in regions if r.is_valid and not r.is_opening], key=lambda r: PRIORITY.get(r.kind, 0))
    for r in ordered:
        choice = assignments.get(r.id) or {}
        mid = choice.get("material")
        if not mid or mid not in catalog or r.id not in masks:
            continue
        m = masks[r.id]
        if not m.any():
            continue
        ys, xs = np.where(m)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        sub_m = m[y0:y1, x0:x1].astype(np.float32)
        mat = catalog[mid]
        color = choice.get("color") or mat.get("default_color", "#cccccc")
        tex, a, use_shading = make_texture(mat["pattern"], color, y1 - y0, x1 - x0, px_per_ft, f"{r.id}:{mid}", mat)

        sub_lum = lum[y0:y1, x0:x1]
        if use_shading:
            med = float(np.median(sub_lum[sub_m > 0])) or 0.5
            shading = np.clip(sub_lum / max(med, 1e-3), 0.35, 1.6)
            shading = 1 + shading_strength * (shading - 1)
            tex = np.clip(tex * shading[..., None], 0, 1)
        feather = cv2.GaussianBlur(sub_m, (0, 0), 1.2) * sub_m if min(sub_m.shape) > 4 else sub_m
        a_total = (a * feather)[..., None]
        out[y0:y1, x0:x1] = out[y0:y1, x0:x1] * (1 - a_total) + tex * a_total
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def _font(size: int):
    for name in ("DejaVuSans.ttf", "arial.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_overlay(image_rgb: np.ndarray, regions: list[Region], masks: dict[str, np.ndarray] | None = None, show_ids: bool = True) -> np.ndarray:
    """Colour-coded region map for the review step."""
    h, w = image_rgb.shape[:2]
    base = image_rgb.astype(np.float32)
    if masks:
        tint = base.copy()
        for r in regions:
            if r.id in masks:
                tint[masks[r.id]] = 0.55 * tint[masks[r.id]] + 0.45 * np.array(KIND_COLORS.get(r.kind, (200, 200, 200)), np.float32)
        base = tint
    pil = Image.fromarray(base.clip(0, 255).astype(np.uint8))
    d = ImageDraw.Draw(pil)
    font = _font(max(12, w // 70))
    for i, r in enumerate(regions):
        if not r.is_valid:
            continue
        col = KIND_COLORS.get(r.kind, (200, 200, 200))
        x0, y0, x1, y1 = r.px_box(w, h)
        d.rectangle([x0, y0, x1, y1], outline=col, width=max(2, w // 400))
        if show_ids:
            tag = r.label
            tb = d.textbbox((x0 + 3, y0 + 2), tag, font=font)
            d.rectangle([tb[0] - 2, tb[1] - 1, tb[2] + 2, tb[3] + 1], fill=col)
            d.text((x0 + 3, y0 + 2), tag, fill=(255, 255, 255), font=font)
    return np.array(pil)
