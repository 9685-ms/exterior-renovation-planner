"""Downloadable PDF report for discussion with contractors."""
from __future__ import annotations

from datetime import datetime

import numpy as np
from fpdf import FPDF
from fpdf.fonts import FontFace
from PIL import Image

from .costing import Estimate, inr
from .estimation import METHOD_LABELS, ScaleResult, SurfaceArea

_REPLACE = {"₹": "Rs.", "–": "-", "—": "-", "×": "x", "’": "'", "‘": "'", "“": '"', "”": '"', "…": "...", "²": "2"}


def _t(s) -> str:
    s = str(s if s is not None else "")
    for a, b in _REPLACE.items():
        s = s.replace(a, b)
    return s.encode("latin-1", "replace").decode("latin-1")


def _money(x: float) -> str:
    return inr(x, "Rs. ")


class _PDF(FPDF):
    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120)
        self.cell(0, 8, _t(f"Advisory estimate, not a quotation. Page {self.page_no()}"), align="C")


def build_report(
    project_name: str,
    design_name: str,
    original: np.ndarray,
    redesigned: np.ndarray,
    areas: list[SurfaceArea],
    estimate: Estimate,
    assignments: dict,
    catalog: dict,
    scale: ScaleResult,
    openings_sqft: float,
    settings: dict,
    rates_note: str,
) -> bytes:
    pdf = _PDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, _t("Exterior Renovation Plan & Estimate"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(90)
    pdf.cell(0, 6, _t(f"Project: {project_name}   |   Design: {design_name}   |   {datetime.now():%d %b %Y}"),
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0)
    pdf.ln(2)

    # Images side by side.
    img_w = 90
    h_ratio = original.shape[0] / original.shape[1]
    y = pdf.get_y()
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(img_w, 6, "Original", new_x="RIGHT")
    pdf.set_x(pdf.l_margin + img_w + 5)
    pdf.cell(img_w, 6, "Redesigned", new_x="LMARGIN", new_y="NEXT")
    y = pdf.get_y()
    pdf.image(Image.fromarray(original), x=pdf.l_margin, y=y, w=img_w)
    pdf.image(Image.fromarray(redesigned), x=pdf.l_margin + img_w + 5, y=y, w=img_w)
    pdf.set_y(y + img_w * h_ratio + 4)

    # Summary.
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, _t(f"Estimated total: {_money(estimate.grand_total)}"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _t(
        f"Materials {_money(estimate.material_total)} + labour {_money(estimate.labour_total)} = "
        f"{_money(estimate.subtotal)}; contingency {estimate.contingency_pct:g}% {_money(estimate.contingency)}; "
        f"GST {estimate.gst_pct:g}% {_money(estimate.gst)}."
    ), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    head = FontFace(emphasis="BOLD", fill_color=(235, 235, 235))

    def section(title: str):
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 7, _t(title), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 8.5)

    section("1. Selected materials")
    with pdf.table(col_widths=(32, 50, 18, 40, 50), headings_style=head, line_height=4.5) as t:
        t.row(["Component", "Material", "Colour", "Durability", "Maintenance"])
        for a in areas:
            ch = assignments.get(a.region_id) or {}
            m = catalog.get(ch.get("material", ""))
            if not m:
                continue
            t.row([_t(a.label), _t(m["name"]), _t(ch.get("color") or m.get("default_color", "")),
                   _t(m.get("durability", "")), _t(m.get("maintenance", ""))])

    section("2. Surface areas")
    with pdf.table(col_widths=(50, 30, 25, 30, 30), headings_style=head, line_height=4.5,
                   text_align=("LEFT", "RIGHT", "RIGHT", "RIGHT", "RIGHT")) as t:
        t.row(["Component", "Visible (sqft)", "Factor", "To finish (sqft)", "Running ft"])
        for a in areas:
            t.row([_t(a.label), f"{a.visible_sqft:,.1f}", f"x{a.surface_factor:g}", f"{a.finish_sqft:,.1f}",
                   f"{a.running_ft:,.1f}" if a.running_ft is not None else "-"])
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _t(
        f"Window and door openings deducted: {openings_sqft:,.1f} sqft. Facade multiplier: "
        f"x{settings.get('facade_multiplier', 1):g}."), new_x="LMARGIN", new_y="NEXT")

    section("3. Quantities and cost")
    with pdf.table(col_widths=(30, 42, 22, 44, 22, 20, 22), headings_style=head, line_height=4.5,
                   text_align=("LEFT", "LEFT", "RIGHT", "LEFT", "RIGHT", "RIGHT", "RIGHT")) as t:
        t.row(["Component", "Material", "Qty", "How calculated", "Material", "Labour", "Total"])
        for li in estimate.lines:
            t.row([_t(li.component), _t(li.material), _t(f"{li.quantity:g} {li.purchase_unit}"), _t(li.quantity_note),
                   _money(li.material_cost), _money(li.labour_cost), _money(li.total)])

    section("4. Purchase list (rounded up to full packs)")
    with pdf.table(col_widths=(70, 30, 30, 30, 32), headings_style=head, line_height=4.5,
                   text_align=("LEFT", "RIGHT", "RIGHT", "RIGHT", "RIGHT")) as t:
        t.row(["Material", "Needed", "Buy", "Rate", "Cost"])
        for p in estimate.purchases:
            t.row([_t(p.material), _t(f"{p.exact_qty:g} {p.unit}"), _t(f"{p.purchase_qty:g} {p.unit}"),
                   _money(p.rate), _money(p.cost)])
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _t(f"Rounding to full packs adds {_money(estimate.rounding_adjustment)} to the component costs above."),
                   new_x="LMARGIN", new_y="NEXT")

    section("5. Totals by category")
    with pdf.table(col_widths=(60, 40, 40, 40), headings_style=head, line_height=5,
                   text_align=("LEFT", "RIGHT", "RIGHT", "RIGHT")) as t:
        t.row(["Category", "Material", "Labour", "Total"])
        for cat, v in estimate.by_category.items():
            t.row([_t(cat), _money(v["material"]), _money(v["labour"]), _money(v["total"])])
        t.row(["Subtotal", _money(estimate.material_total), _money(estimate.labour_total), _money(estimate.subtotal)])
        t.row([f"Contingency {estimate.contingency_pct:g}%", "", "", _money(estimate.contingency)])
        t.row([f"GST {estimate.gst_pct:g}%", "", "", _money(estimate.gst)])
        t.row(["Grand total", "", "", _money(estimate.grand_total)])

    section("6. How this was estimated")
    pdf.set_font("Helvetica", "", 9)
    lines = [
        f"Scale: {METHOD_LABELS.get(scale.method, scale.method)} -> {1 / scale.ft_per_px:.1f} px per ft.",
        *[f"  - {METHOD_LABELS.get(c.method, c.method)}: {c.detail} ({1 / c.ft_per_px:.1f} px/ft)" for c in scale.candidates],
        "Areas = pixels in each component's mask x (ft per px)^2, with windows and doors removed, then multiplied by a "
        "surface factor for faces the photo cannot show (pillars x3, parapets x2, gates x2, roof edges x1.5).",
        "Quantities use manufacturer-style coverage, tile/panel sizes and a wastage % per material; paint includes one primer coat. Each component keeps its exact quantity; rounding up to whole litres, boxes or sheets is done once per material in the purchase list.",
        "Cost = purchase quantity x material rate + finished area (or running ft) x labour rate, plus contingency and GST.",
        rates_note,
    ]
    if scale.warning:
        lines.insert(1, "Warning: " + scale.warning)
    if estimate.unassigned:
        lines.append("Not included (no material chosen): " + ", ".join(estimate.unassigned))
    for ln in lines:
        pdf.multi_cell(0, 4.5, _t(ln), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "I", 8.5)
    pdf.multi_cell(0, 4.2, _t(
        "Limitations: measurements come from a single photo and assume a roughly front-on view; sides and rear of the "
        "house are not seen unless a facade multiplier is set. Rates are indicative. Verify key dimensions on site "
        "and get supplier quotes before ordering. This document is advisory and not legally binding."),
        new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
