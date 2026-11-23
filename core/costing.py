"""Material quantities and cost estimate.

Quantity rules per component (all include the material's wastage %):
- paint:    litres = area * coats / coverage_per_litre_per_coat, plus a primer
            coat: litres = area / primer_coverage.
- coverage: units  = area * coats / coverage_per_unit (e.g. texture in kg).
- tile:     pieces = area / tile_area, boxes = pieces / pieces_per_box.
- panel:    sheets or planks = area / panel_area.
- area:     sqft of cladding (sold per sqft).
- length:   running ft (railings).

Component lines keep the exact (fractional) quantity so small parts are not
charged a whole box or sheet each. Rounding up to whole purchase units
(1 L, 1 kg, 1 box, 1 sheet, 1 sqft, 0.5 running ft) happens once per material
in the purchase list, and the difference is shown as "rounding to full packs".

Cost = purchase quantity * material rate + labour basis * labour rate.
Labour basis is the finished area (sqft) or running length (railings).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .estimation import SurfaceArea

PURCHASE_STEP = {"paint": 1.0, "coverage": 1.0, "tile": 1.0, "panel": 1.0, "area": 1.0, "length": 0.5}


def _ceil_to(x: float, step: float) -> float:
    """Round up to a multiple of step, ignoring float noise (100 * 1.1 / 2 must be 55, not 56)."""
    if x <= 0:
        return 0.0
    return math.ceil(round(x / step, 6)) * step


@dataclass
class LineItem:
    region_id: str
    component: str
    material_id: str
    material: str
    category: str
    color: str | None
    basis_qty: float          # sqft finished or running ft
    basis_unit: str
    quantity: float           # exact quantity in purchase units incl. wastage
    purchase_unit: str
    quantity_note: str
    material_cost: float      # quantity * rate (before rounding to packs)
    labour_cost: float
    unit_rate: float = 0.0
    purchase_step: float = 1.0

    @property
    def total(self) -> float:
        return self.material_cost + self.labour_cost

    def to_dict(self) -> dict:
        d = asdict(self)
        d["total"] = self.total
        return d


@dataclass
class PurchaseItem:
    material_id: str
    material: str
    category: str
    exact_qty: float
    purchase_qty: float
    unit: str
    rate: float

    @property
    def cost(self) -> float:
        return round(self.purchase_qty * self.rate, 2)


def compute_line(area: SurfaceArea, material: dict, color: str | None = None) -> list[LineItem]:
    method = material["quantity_method"]
    wpct = float(material.get("wastage_pct", 0))
    wastage = 1 + wpct / 100.0
    coats = float(material.get("coats", 1) or 1)
    rate = float(material["material_rate"])
    labour_rate = float(material["labour_rate"])
    sqft = float(area.finish_sqft)
    items: list[LineItem] = []

    if method == "length":
        basis, basis_unit = float(area.running_ft or 0.0), "running ft"
    else:
        basis, basis_unit = sqft, "sqft"

    if method == "paint":
        cov = float(material["coverage_sqft_per_unit"])
        qty = sqft * coats / cov * wastage
        note = f"{sqft:,.0f} sqft x {coats:g} coats / {cov:g} sqft per L + {wpct:g}% wastage"
    elif method == "coverage":
        cov = float(material["coverage_sqft_per_unit"])
        qty = sqft * coats / cov * wastage
        note = f"{sqft:,.0f} sqft / {cov:g} sqft per {material['purchase_unit']} + {wpct:g}% wastage"
    elif method == "tile":
        tile_area = float(material["tile_w_ft"]) * float(material["tile_h_ft"])
        pieces = sqft * wastage / tile_area
        per_box = int(material.get("pieces_per_box", 1))
        qty = pieces / per_box
        note = f"~{math.ceil(round(pieces, 6)):,} tiles ({tile_area:g} sqft each, {per_box}/box) incl. {wpct:g}% wastage"
    elif method == "panel":
        panel_area = float(material["panel_w_ft"]) * float(material["panel_h_ft"])
        qty = sqft * wastage / panel_area
        note = f"{sqft:,.0f} sqft / {panel_area:g} sqft per {material['purchase_unit']} + {wpct:g}% wastage"
    elif method == "area":
        qty = sqft * wastage
        note = f"{sqft:,.0f} sqft + {wpct:g}% wastage"
    elif method == "length":
        qty = basis * wastage
        note = f"{basis:,.1f} running ft + {wpct:g}% wastage"
    else:
        raise ValueError(f"Unknown quantity method {method!r}")

    items.append(LineItem(
        region_id=area.region_id, component=area.label, material_id=material["id"], material=material["name"],
        category=material["category"], color=color, basis_qty=round(basis, 1), basis_unit=basis_unit,
        quantity=round(qty, 2), purchase_unit=material["purchase_unit"], quantity_note=note,
        material_cost=round(qty * rate, 2), labour_cost=round(basis * labour_rate, 2),
        unit_rate=rate, purchase_step=PURCHASE_STEP[method],
    ))

    # Primer coat under paint / texture systems. Its labour is in the rate above.
    if method in ("paint", "coverage") and material.get("primer_coverage_sqft_per_litre"):
        pcov = float(material["primer_coverage_sqft_per_litre"])
        prate = float(material.get("primer_rate", 0))
        pqty = sqft / pcov * wastage
        items.append(LineItem(
            region_id=area.region_id, component=area.label, material_id=material["id"] + "__primer",
            material=f"Primer for {material['name'].lower()}", category=material["category"], color=None,
            basis_qty=round(sqft, 1), basis_unit="sqft", quantity=round(pqty, 2), purchase_unit="litre",
            quantity_note=f"{sqft:,.0f} sqft / {pcov:g} sqft per L, 1 coat", material_cost=round(pqty * prate, 2),
            labour_cost=0.0, unit_rate=prate, purchase_step=1.0,
        ))
    return items


@dataclass
class Estimate:
    lines: list[LineItem]
    purchases: list[PurchaseItem]
    rounding_adjustment: float
    by_category: dict[str, dict[str, float]]
    material_total: float
    labour_total: float
    subtotal: float
    contingency_pct: float
    contingency: float
    gst_pct: float
    gst: float
    grand_total: float
    unassigned: list[str]


def build_purchases(lines: list[LineItem]) -> list[PurchaseItem]:
    grouped: dict[str, PurchaseItem] = {}
    for li in lines:
        p = grouped.get(li.material_id)
        if p is None:
            p = grouped[li.material_id] = PurchaseItem(li.material_id, li.material, li.category, 0.0, 0.0,
                                                       li.purchase_unit, li.unit_rate)
        p.exact_qty += li.quantity
    steps = {li.material_id: li.purchase_step for li in lines}
    for mid, p in grouped.items():
        p.exact_qty = round(p.exact_qty, 2)
        p.purchase_qty = _ceil_to(p.exact_qty, steps[mid])
    return list(grouped.values())


def build_estimate(
    areas: list[SurfaceArea],
    assignments: dict[str, dict],
    catalog: dict[str, dict],
    contingency_pct: float = 5.0,
    gst_pct: float = 18.0,
) -> Estimate:
    """assignments: region_id -> {"material": id, "color": "#rrggbb" | None}."""
    lines: list[LineItem] = []
    unassigned: list[str] = []
    for a in areas:
        choice = assignments.get(a.region_id) or {}
        mid = choice.get("material")
        if not mid or mid not in catalog:
            unassigned.append(a.label)
            continue
        lines.extend(compute_line(a, catalog[mid], choice.get("color")))

    purchases = build_purchases(lines)
    by_cat: dict[str, dict[str, float]] = {}
    for li in lines:
        c = by_cat.setdefault(li.category, {"material": 0.0, "labour": 0.0, "total": 0.0})
        c["labour"] += li.labour_cost
    for p in purchases:
        by_cat[p.category]["material"] += p.cost
    for c in by_cat.values():
        c["total"] = c["material"] + c["labour"]

    mat = round(sum(p.cost for p in purchases), 2)
    rounding = round(mat - sum(li.material_cost for li in lines), 2)
    lab = round(sum(li.labour_cost for li in lines), 2)
    sub = round(mat + lab, 2)
    cont = round(sub * contingency_pct / 100.0, 2)
    gst = round((sub + cont) * gst_pct / 100.0, 2)
    return Estimate(
        lines=lines,
        purchases=purchases,
        rounding_adjustment=rounding,
        by_category={k: {kk: round(vv, 2) for kk, vv in v.items()} for k, v in by_cat.items()},
        material_total=mat,
        labour_total=lab,
        subtotal=sub,
        contingency_pct=contingency_pct,
        contingency=cont,
        gst_pct=gst_pct,
        gst=gst,
        grand_total=round(sub + cont + gst, 2),
        unassigned=unassigned,
    )


def inr(x: float, symbol: str = "₹") -> str:
    """Format a number in the Indian digit grouping, e.g. 12,34,567."""
    neg = x < 0
    n = int(round(abs(x)))
    s = str(n)
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        s = ",".join(groups + [tail])
    return f"{'-' if neg else ''}{symbol}{s}"
