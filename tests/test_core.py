import json
import math

import numpy as np
import pytest

from core import pipeline
from core.catalog import apply_overrides, load_catalog
from core.costing import build_estimate, build_purchases, compute_line, inr
from core.detection import DetectionError, detect_components, parse_components
from core.estimation import SurfaceArea, choose_scale, scale_candidates
from core.presets import PRESETS, apply_preset
from core.project import dump_project, load_project
from core.quality import assess
from core.regions import Region, build_masks
from core.report import build_report
from core.sample import PX_PER_FT, sample_image, sample_regions


@pytest.fixture(scope="module")
def sample():
    return sample_image(), sample_regions()


def by_label(areas, label):
    return next(a for a in areas if a.label == label)


# ---------------------------------------------------------------- masks / areas
def test_masks_are_exclusive(sample):
    img, regs = sample
    masks = build_masks(regs, img.shape[1], img.shape[0])
    stack = np.stack(list(masks.values())).astype(int).sum(axis=0)
    assert stack.max() == 1, "a pixel was assigned to two components"


def test_scale_from_sample_references(sample):
    img, regs = sample
    cands = scale_candidates(regs, img.shape[1], img.shape[0], floors=2)
    assert {c.method for c in cands} == {"door_reference", "floor_count", "window_reference"}
    for c in cands:
        assert 1 / c.ft_per_px == pytest.approx(PX_PER_FT, rel=0.01)


def test_user_measurement_wins(sample):
    img, regs = sample
    cands = scale_candidates(regs, img.shape[1], img.shape[0], known_width_ft=74.0, floors=2)
    res = choose_scale(cands, img.shape[1])
    assert res.method == "user_measurement"
    # Largest wall is 740 px wide -> 10 px/ft, disagrees with doors (20 px/ft) -> warning.
    assert 1 / res.ft_per_px == pytest.approx(10.0)
    assert res.warning


def test_fallback_scale_warns():
    res = choose_scale([], 1000)
    assert res.method == "fallback" and res.warning


def test_area_matches_ground_truth(sample):
    img, regs = sample
    res = pipeline.run(img, regs, {}, {"floors": 2}, render=False)
    # Ground floor wall: 37 x 9.6 ft minus 2 windows (4x4), main door (3x7), 2 chajjas (5x0.6).
    assert by_label(res.areas, "Ground floor wall").visible_sqft == pytest.approx(37 * 9.6 - 32 - 21 - 6, abs=0.5)
    # First floor wall: 37 x 9.5 minus 2 windows, balcony door (3x7), railing in front of the wall (12x3 minus door overlap 3x3).
    assert by_label(res.areas, "First floor wall").visible_sqft == pytest.approx(37 * 9.5 - 32 - 21 - 27, abs=0.5)
    pillar = by_label(res.areas, "Left corner pillar")
    assert pillar.visible_sqft == pytest.approx(1.5 * 20, abs=0.2)
    assert pillar.finish_sqft == pytest.approx(3 * 30, abs=0.5)
    railing = by_label(res.areas, "Balcony railing")
    assert railing.running_ft == pytest.approx(12.0, abs=0.1)
    assert railing.finish_sqft == pytest.approx(12 * 3.5, abs=0.1)


def test_facade_multiplier_scales_areas(sample):
    img, regs = sample
    a1 = pipeline.run(img, regs, {}, {"floors": 2}, render=False).areas
    a3 = pipeline.run(img, regs, {}, {"floors": 2, "facade_multiplier": 3}, render=False).areas
    assert by_label(a3, "Ground floor wall").finish_sqft == pytest.approx(3 * by_label(a1, "Ground floor wall").finish_sqft, rel=0.01)


# ---------------------------------------------------------------- quantities / cost
def _area(sqft=100.0, running=None, kind="wall"):
    return SurfaceArea("r1", kind, "Wall", sqft, 1.0, sqft, running, 0)


def test_paint_quantity_and_primer():
    cat = load_catalog()
    lines = compute_line(_area(1000), cat["paint_emulsion"], "#ffffff")
    paint, primer = lines
    # 1000 sqft x 2 coats / 100 sqft per L x 1.05 = 21 L
    assert paint.quantity == 21.0
    assert paint.material_cost == 21 * 450
    assert paint.labour_cost == 1000 * 15
    assert primer.quantity == round(1000 / 120 * 1.05, 2)


def test_tile_boxes():
    cat = load_catalog()
    (li,) = compute_line(_area(100), cat["tile_vitrified_1x2"])
    # 100 sqft x 1.10 / 2 sqft = 55 tiles (exactly, no float noise) = 9.17 boxes of 6
    assert "~55 tiles" in li.quantity_note
    assert li.quantity == round(55 / 6, 2)
    (p,) = build_purchases([li])
    assert p.purchase_qty == 10 and p.cost == 10 * 840


def test_rounding_happens_once_per_material():
    """Two tiny chajjas in ACP must share one sheet, not buy a sheet each."""
    cat = load_catalog()
    small = [SurfaceArea(f"r{i}", "roof_edge", f"Chajja {i}", 3.0, 1.5, 4.5, None, 0) for i in range(2)]
    est = build_estimate(small, {a.region_id: {"material": "acp_panel"} for a in small}, cat, 0, 0)
    (p,) = est.purchases
    assert p.purchase_qty == 1
    assert est.material_total == 3200
    assert est.rounding_adjustment == pytest.approx(3200 - 2 * 4.5 * 1.08 / 32 * 3200, abs=0.05)


def test_railing_by_running_ft():
    cat = load_catalog()
    (li,) = compute_line(_area(42, running=12.0, kind="railing"), cat["glass_railing"])
    assert li.basis_unit == "running ft"
    assert li.quantity == round(12 * 1.03, 2)
    assert li.labour_cost == 12 * 400


def test_rate_override_changes_cost():
    cat = apply_overrides(load_catalog(), {"paint_emulsion": {"material_rate": 500, "bogus": 1}})
    assert cat["paint_emulsion"]["material_rate"] == 500
    assert "bogus" not in cat["paint_emulsion"]


def test_estimate_totals_add_up(sample):
    img, regs = sample
    design = apply_preset("Modern contemporary", regs, load_catalog())
    res = pipeline.run(img, regs, design, {"floors": 2, "contingency_pct": 10, "gst_pct": 18}, render=False)
    e = res.estimate
    assert e.subtotal == pytest.approx(sum(li.total for li in e.lines) + e.rounding_adjustment, abs=1)
    assert e.material_total == pytest.approx(sum(p.cost for p in e.purchases), abs=0.01)
    assert e.grand_total == pytest.approx(e.subtotal * 1.10 * 1.18, abs=2)
    assert sum(v["total"] for v in e.by_category.values()) == pytest.approx(e.subtotal, abs=1)
    assert not e.unassigned


def test_every_preset_is_valid(sample):
    _, regs = sample
    cat = load_catalog()
    for name in PRESETS:
        d = apply_preset(name, regs, cat)
        surfaces = [r for r in regs if not r.is_opening]
        assert len(d) == len(surfaces), name


def test_inr_grouping():
    assert inr(1234567) == "₹12,34,567"
    assert inr(999) == "₹999"
    assert inr(100000, "Rs. ") == "Rs. 1,00,000"


# ---------------------------------------------------------------- detection parsing
def test_parse_fenced_json_and_synonyms():
    text = """```json
{"components": [
 {"kind": "Column", "label": "left", "box_2d": [100, 50, 900, 80], "confidence": 0.9},
 {"kind": "main wall", "box_2d": [200, 100, 800, 900]},
 {"kind": "tree", "box_2d": [0, 0, 100, 100]},
 {"kind": "window", "box_2d": [300, 300, 300, 400]}
]}
```"""
    regs = parse_components(text)
    assert [r.kind for r in regs] == ["pillar", "wall"]
    assert regs[0].x0 == pytest.approx(0.05) and regs[0].y1 == pytest.approx(0.9)


def test_parse_bare_list_and_unit_coords():
    regs = parse_components('[{"kind": "door", "box_2d": [0.5, 0.4, 0.9, 0.5]}]')
    assert regs[0].kind == "door" and regs[0].y0 == pytest.approx(0.5)


def test_parse_rejects_garbage():
    with pytest.raises(DetectionError):
        parse_components("I cannot help with that.")
    with pytest.raises(DetectionError):
        parse_components('{"components": []}')


def test_detect_retries_once_on_bad_output(monkeypatch):
    import core.detection as det

    calls = []

    def fake(jpeg, key, model):
        calls.append(model)
        if len(calls) == 1:
            return "not json"
        return json.dumps({"components": [{"kind": "wall", "box_2d": [100, 100, 900, 900]}]})

    monkeypatch.setattr(det, "_call_gemini", fake)
    regs, info = detect_components(np.zeros((100, 100, 3), np.uint8), "gemini", "k", model="m")
    assert len(regs) == 1 and info["attempts"] == 2


def test_detect_surfaces_api_errors(monkeypatch):
    import core.detection as det

    def boom(*a):
        raise RuntimeError("401 invalid key")

    monkeypatch.setattr(det, "_call_openai", boom)
    with pytest.raises(DetectionError, match="401"):
        detect_components(np.zeros((50, 50, 3), np.uint8), "openai", "k")


# ---------------------------------------------------------------- quality gate
def test_quality_accepts_sample(sample):
    rep = assess(sample[0])
    assert rep.usable and rep.score > 50


def test_quality_rejects_blur_dark_and_tiny():
    rng = np.random.default_rng(0)
    flat = np.full((800, 1200, 3), 128, np.uint8)
    assert not assess(flat).usable  # no detail at all
    dark = (rng.random((800, 1200, 3)) * 40).astype(np.uint8)
    rep = assess(dark)
    assert not rep.usable and any("dark" in i for i in rep.issues)
    tiny = (rng.random((200, 300, 3)) * 255).astype(np.uint8)
    rep = assess(tiny)
    assert not rep.usable and any("Resolution" in i for i in rep.issues)


# ---------------------------------------------------------------- project + report
def test_project_roundtrip(sample):
    img, regs = sample
    designs = {"A": apply_preset("Classic warm", regs, load_catalog())}
    raw = dump_project("Test", img, regs, designs, "A", {"paint_emulsion": {"material_rate": 500}}, {"floors": 2})
    p = load_project(raw)
    assert p["name"] == "Test" and p["active_design"] == "A"
    assert [r.id for r in p["regions"]] == [r.id for r in regs]
    assert p["designs"] == designs
    assert p["image"].shape == img.shape
    with pytest.raises(ValueError):
        load_project(b'{"format": "other"}')


def test_report_pdf(sample):
    img, regs = sample
    cat = load_catalog()
    design = apply_preset("Earthy textured", regs, cat)
    res = pipeline.run(img, regs, design, {"floors": 2})
    pdf = build_report("Test house", "Design A", img, res.rendered, res.areas, res.estimate, design, res.catalog,
                       res.scale, res.openings_sqft, {"facade_multiplier": 1}, "rates note ₹")
    assert pdf[:4] == b"%PDF" and len(pdf) > 50_000


def test_render_changes_only_assigned_pixels(sample):
    img, regs = sample
    wall = next(r for r in regs if r.label == "Ground floor wall")
    res = pipeline.run(img, regs, {wall.id: {"material": "paint_emulsion", "color": "#1050A0"}}, {"floors": 2})
    changed = np.abs(res.rendered.astype(int) - img.astype(int)).sum(axis=2) > 6
    wall_mask = res.masks[wall.id]
    assert changed[wall_mask].mean() > 0.95
    # Outside the wall (allowing a 3 px feather band) nothing changes.
    import cv2
    grown = cv2.dilate(wall_mask.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    assert changed[~grown].mean() < 0.001
