"""One call that turns (photo, regions, design, settings) into all outputs.

Kept free of Streamlit so it can be unit-tested and reused behind an API.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .catalog import apply_overrides, load_catalog
from .costing import Estimate, build_estimate
from .estimation import SurfaceArea, ScaleResult, choose_scale, estimate_areas, opening_areas, scale_candidates
from .regions import Region, build_masks
from .render import render_design

DEFAULT_SETTINGS = {
    "known_width_ft": 0.0,
    "known_width_region_id": None,
    "floors": 0,
    "scale_method": "auto",
    "facade_multiplier": 1.0,
    "surface_factors": {},
    "railing_height_ft": 3.5,
    "contingency_pct": 5.0,
    "gst_pct": 18.0,
    "refine_edges": False,
}


@dataclass
class Result:
    masks: dict[str, np.ndarray]
    scale: ScaleResult
    areas: list[SurfaceArea]
    openings_sqft: float
    estimate: Estimate
    rendered: np.ndarray
    catalog: dict


def run(image: np.ndarray, regions: list[Region], assignments: dict, settings: dict | None = None,
        rate_overrides: dict | None = None, masks: dict | None = None, render: bool = True) -> Result:
    s = {**DEFAULT_SETTINGS, **(settings or {})}
    h, w = image.shape[:2]
    catalog = apply_overrides(load_catalog(), rate_overrides or {})
    if masks is None:
        refine = ("wall", "parapet") if s["refine_edges"] else ()
        masks = build_masks(regions, w, h, image_rgb=image, refine_kinds=refine)
    cands = scale_candidates(regions, w, h, known_width_ft=s["known_width_ft"] or None,
                             known_width_region_id=s["known_width_region_id"], floors=int(s["floors"] or 0))
    scale = choose_scale(cands, w, s["scale_method"])
    areas = estimate_areas(regions, masks, scale.ft_per_px, w, h, s["surface_factors"],
                           float(s["facade_multiplier"]), float(s["railing_height_ft"]))
    est = build_estimate(areas, assignments, catalog, float(s["contingency_pct"]), float(s["gst_pct"]))
    rendered = render_design(image, regions, masks, assignments, catalog, scale.ft_per_px) if render else image
    return Result(masks, scale, areas, opening_areas(regions, masks, scale.ft_per_px), est, rendered, catalog)
