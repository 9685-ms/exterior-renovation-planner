"""Material catalogue loading and lookup."""
from __future__ import annotations

import copy
import json
from functools import lru_cache
from pathlib import Path

CATALOG_PATH = Path(__file__).resolve().parent.parent / "data" / "materials.json"

# Fields a user may override from the UI ("modify material rates").
EDITABLE_FIELDS = ["material_rate", "labour_rate", "wastage_pct", "coverage_sqft_per_unit", "coats"]


@lru_cache(maxsize=1)
def _load_raw() -> dict:
    with open(CATALOG_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def load_catalog() -> dict[str, dict]:
    """Return a fresh, mutable copy of the catalogue keyed by material id."""
    raw = _load_raw()
    return {m["id"]: copy.deepcopy(m) for m in raw["materials"]}


def rates_note() -> str:
    return _load_raw().get("rates_note", "")


def materials_for(kind: str, catalog: dict[str, dict]) -> list[dict]:
    return [m for m in catalog.values() if kind in m["applies_to"]]


def apply_overrides(catalog: dict[str, dict], overrides: dict[str, dict]) -> dict[str, dict]:
    """Return a catalogue with user rate overrides applied (numbers only)."""
    merged = copy.deepcopy(catalog)
    for mid, fields in (overrides or {}).items():
        if mid not in merged:
            continue
        for key, value in fields.items():
            if key in EDITABLE_FIELDS and value is not None:
                try:
                    merged[mid][key] = float(value)
                except (TypeError, ValueError):
                    continue
    return merged
