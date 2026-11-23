"""Ready-made design styles so users can compare combinations quickly."""
from __future__ import annotations

from .regions import Region

# kind -> (material id, colour or None for the material default)
PRESETS: dict[str, dict[str, tuple[str, str | None]]] = {
    "Classic warm": {
        "wall": ("paint_emulsion", "#E8D8B8"),
        "parapet": ("paint_emulsion", "#B9764F"),
        "pillar": ("stone_cladding", None),
        "balcony": ("paint_emulsion", "#B9764F"),
        "roof_edge": ("paint_emulsion", "#F4EFE6"),
        "railing": ("ss_railing", None),
        "gate": ("enamel_metal", "#2F3A40"),
    },
    "Modern contemporary": {
        "wall": ("paint_elastomeric", "#E7E6E1"),
        "parapet": ("acp_panel", "#3E4A52"),
        "pillar": ("tile_vitrified_1x2", "#5E5A57"),
        "balcony": ("wpc_panel", None),
        "roof_edge": ("acp_panel", "#3E4A52"),
        "railing": ("glass_railing", None),
        "gate": ("enamel_metal", "#1F1F1F"),
    },
    "Earthy textured": {
        "wall": ("texture_finish", "#D6B98C"),
        "parapet": ("brick_cladding", None),
        "pillar": ("brick_cladding", None),
        "balcony": ("paint_emulsion", "#7A4E33"),
        "roof_edge": ("paint_emulsion", "#7A4E33"),
        "railing": ("ms_railing", None),
        "gate": ("enamel_metal", "#3B2A20"),
    },
    "Budget refresh": {
        "wall": ("paint_emulsion", "#DDE3E6"),
        "parapet": ("paint_emulsion", "#6C8A9B"),
        "pillar": ("paint_emulsion", "#6C8A9B"),
        "balcony": ("paint_emulsion", "#6C8A9B"),
        "roof_edge": ("paint_emulsion", "#FFFFFF"),
        "railing": ("enamel_metal", "#222222"),
        "gate": ("enamel_metal", "#222222"),
    },
}


def apply_preset(name: str, regions: list[Region], catalog: dict) -> dict[str, dict]:
    spec = PRESETS[name]
    out: dict[str, dict] = {}
    for r in regions:
        if r.is_opening or r.kind not in spec:
            continue
        mid, color = spec[r.kind]
        if mid in catalog and r.kind in catalog[mid]["applies_to"]:
            out[r.id] = {"material": mid, "color": color or catalog[mid]["default_color"]}
    return out
