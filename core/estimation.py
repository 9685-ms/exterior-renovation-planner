"""Surface area estimation from a single photo.

How it works
------------
1. Calibrate a scale (feet per pixel) from something of known size:
   - a measurement the user types in (most reliable),
   - a detected door (standard Indian door leaf height ~7 ft),
   - the number of floors (typical floor-to-floor height ~10 ft),
   - a detected window (typical height ~4 ft, least reliable).
2. Count the pixels in each component's exclusive mask and convert:
   area_sqft = pixels * (ft_per_px ** 2).
   Openings (windows, doors) are carved out of walls by the mask builder.
3. Apply a per-component surface factor for faces the photo cannot show,
   e.g. a pillar has ~3 exposed faces, a parapet is finished on both sides.
4. Railings are measured as running length (box width * ft_per_px).

The photo is assumed to be roughly front-on. Perspective, lens distortion and
hidden facades are the main error sources; the UI lets the user correct the
scale and add a facade multiplier.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field

import numpy as np

from .regions import KIND_LABELS, Region

DEFAULT_REFERENCES = {
    "door_height_ft": 7.0,
    "window_height_ft": 4.0,
    "floor_height_ft": 10.0,
}

# Multiplier from the visible face to the area that actually gets finished.
DEFAULT_SURFACE_FACTORS = {
    "wall": 1.0,
    "parapet": 2.0,   # outer + inner face
    "pillar": 3.0,    # engaged column: front + two sides
    "balcony": 1.0,   # front fascia / slab edge as seen
    "roof_edge": 1.5, # chajja front + soffit
    "gate": 2.0,      # both sides
    "railing": 1.0,
}

DEFAULT_RAILING_HEIGHT_FT = 3.5


@dataclass
class ScaleCandidate:
    method: str
    ft_per_px: float
    detail: str


@dataclass
class ScaleResult:
    ft_per_px: float
    method: str
    candidates: list[ScaleCandidate] = field(default_factory=list)
    warning: str | None = None


def scale_candidates(
    regions: list[Region],
    width: int,
    height: int,
    known_width_ft: float | None = None,
    known_width_region_id: str | None = None,
    floors: int | None = None,
    refs: dict | None = None,
) -> list[ScaleCandidate]:
    refs = {**DEFAULT_REFERENCES, **(refs or {})}
    out: list[ScaleCandidate] = []
    valid = [r for r in regions if r.is_valid]

    # 1. User measurement of a specific component's width.
    if known_width_ft and known_width_ft > 0:
        target = None
        if known_width_region_id:
            target = next((r for r in valid if r.id == known_width_region_id), None)
        if target is None:
            walls = [r for r in valid if r.kind == "wall"]
            target = max(walls, key=lambda r: r.px_size(width, height)[0], default=None)
        if target is not None:
            wpx, _ = target.px_size(width, height)
            if wpx > 0:
                out.append(ScaleCandidate(
                    "user_measurement",
                    known_width_ft / wpx,
                    f"{target.label} measured as {known_width_ft:g} ft wide ({wpx} px)",
                ))

    # 2. Doors.
    door_scales = []
    for r in valid:
        if r.kind == "door":
            _, hpx = r.px_size(width, height)
            if hpx > 0:
                door_scales.append(refs["door_height_ft"] / hpx)
    if door_scales:
        out.append(ScaleCandidate(
            "door_reference",
            statistics.median(door_scales),
            f"{len(door_scales)} door(s) assumed {refs['door_height_ft']:g} ft tall",
        ))

    # 3. Floors.
    if floors and floors > 0:
        # Facade height = storeys only, so the parapet above the roof is excluded.
        storey = [r for r in valid if r.kind in ("wall", "pillar", "balcony", "roof_edge")]
        if storey:
            hpx = (max(r.y1 for r in storey) - min(r.y0 for r in storey)) * height
            if hpx > 0:
                out.append(ScaleCandidate(
                    "floor_count",
                    floors * refs["floor_height_ft"] / hpx,
                    f"{floors} floor(s) x {refs['floor_height_ft']:g} ft over {int(hpx)} px of facade height",
                ))

    # 4. Windows.
    win_scales = []
    for r in valid:
        if r.kind == "window":
            _, hpx = r.px_size(width, height)
            if hpx > 0:
                win_scales.append(refs["window_height_ft"] / hpx)
    if win_scales:
        out.append(ScaleCandidate(
            "window_reference",
            statistics.median(win_scales),
            f"{len(win_scales)} window(s) assumed {refs['window_height_ft']:g} ft tall",
        ))
    return out


METHOD_PRIORITY = ["user_measurement", "door_reference", "floor_count", "window_reference"]
METHOD_LABELS = {
    "user_measurement": "Your measurement",
    "door_reference": "Standard door height",
    "floor_count": "Number of floors",
    "window_reference": "Standard window height",
    "fallback": "Fallback assumption",
}


def choose_scale(candidates: list[ScaleCandidate], width: int, preferred: str = "auto") -> ScaleResult:
    by_method = {c.method: c for c in candidates}
    if preferred != "auto" and preferred in by_method:
        c = by_method[preferred]
        return ScaleResult(c.ft_per_px, c.method, candidates, _spread_warning(candidates))
    for m in METHOD_PRIORITY:
        if m in by_method:
            c = by_method[m]
            return ScaleResult(c.ft_per_px, c.method, candidates, _spread_warning(candidates))
    # Nothing to calibrate against: assume the photo spans ~40 ft.
    fallback = 40.0 / max(width, 1)
    return ScaleResult(
        fallback,
        "fallback",
        candidates,
        "No door, window, floor count or measurement available, so the photo width is assumed to be about 40 ft. "
        "Areas may be far off: add a door box or type in one measurement.",
    )


def _spread_warning(candidates: list[ScaleCandidate]) -> str | None:
    if len(candidates) < 2:
        return None
    vals = [c.ft_per_px for c in candidates]
    lo, hi = min(vals), max(vals)
    if lo > 0 and hi / lo > 1.35:
        return (
            f"Scale references disagree by {int((hi / lo - 1) * 100)}%. The photo may be taken at an angle, "
            "or a door/window is non-standard. Type in one real measurement for a better estimate."
        )
    return None


@dataclass
class SurfaceArea:
    region_id: str
    kind: str
    label: str
    visible_sqft: float
    surface_factor: float
    finish_sqft: float
    running_ft: float | None
    pixel_count: int


def estimate_areas(
    regions: list[Region],
    masks: dict[str, np.ndarray],
    ft_per_px: float,
    width: int,
    height: int,
    surface_factors: dict | None = None,
    facade_multiplier: float = 1.0,
    railing_height_ft: float = DEFAULT_RAILING_HEIGHT_FT,
) -> list[SurfaceArea]:
    factors = {**DEFAULT_SURFACE_FACTORS, **(surface_factors or {})}
    sqft_per_px = ft_per_px ** 2
    out: list[SurfaceArea] = []
    for r in regions:
        if not r.is_valid or r.is_opening or r.id not in masks:
            continue
        px = int(masks[r.id].sum())
        visible = px * sqft_per_px
        factor = float(factors.get(r.kind, 1.0))
        running = None
        if r.kind == "railing":
            wpx, _ = r.px_size(width, height)
            running = wpx * ft_per_px * facade_multiplier
            finish = running * railing_height_ft * factor
        else:
            finish = visible * factor * facade_multiplier
        out.append(SurfaceArea(
            region_id=r.id,
            kind=r.kind,
            label=r.label or KIND_LABELS.get(r.kind, r.kind),
            visible_sqft=round(visible, 1),
            surface_factor=factor,
            finish_sqft=round(finish, 1),
            running_ft=round(running, 1) if running is not None else None,
            pixel_count=px,
        ))
    return out


def opening_areas(regions: list[Region], masks: dict[str, np.ndarray], ft_per_px: float) -> float:
    """Total visible window + door area (deducted from walls), in sqft."""
    return round(sum(int(masks[r.id].sum()) for r in regions if r.is_opening and r.id in masks) * ft_per_px ** 2, 1)
