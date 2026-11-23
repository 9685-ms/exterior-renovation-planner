"""Building component regions and pixel masks.

A region is an axis-aligned box in normalised image coordinates (0..1).
Boxes come from the vision model, the bundled sample, or the user, and the
user can edit every box in the UI. Masks are derived from boxes with a
priority order so that every pixel belongs to at most one component and no
area is counted twice (e.g. a window carved out of a wall, a pillar standing
in front of a wall).
"""
from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field

import numpy as np

# Component kinds the system understands.
SURFACE_KINDS = ["wall", "parapet", "pillar", "balcony", "roof_edge", "gate", "railing"]
OPENING_KINDS = ["window", "door"]
ALL_KINDS = SURFACE_KINDS + OPENING_KINDS

KIND_LABELS = {
    "wall": "Main wall",
    "parapet": "Parapet wall",
    "pillar": "Pillar / column",
    "balcony": "Balcony",
    "roof_edge": "Roof edge / chajja",
    "gate": "Gate",
    "railing": "Railing",
    "window": "Window",
    "door": "Door",
}

# Higher number wins when boxes overlap. Railings stand in front of whatever
# is behind them (including a balcony door); openings are never painted;
# foreground elements (pillars, balconies) sit in front of the wall.
PRIORITY = {
    "railing": 110,
    "window": 100,
    "door": 100,
    "gate": 80,
    "pillar": 70,
    "balcony": 60,
    "roof_edge": 50,
    "parapet": 40,
    "wall": 10,
}

# Synonyms a vision model may return.
KIND_SYNONYMS = {
    "column": "pillar",
    "columns": "pillar",
    "pillars": "pillar",
    "main_wall": "wall",
    "walls": "wall",
    "facade": "wall",
    "façade": "wall",
    "exterior_wall": "wall",
    "parapet_wall": "parapet",
    "roof": "roof_edge",
    "roof_edges": "roof_edge",
    "roofedge": "roof_edge",
    "chajja": "roof_edge",
    "sunshade": "roof_edge",
    "eave": "roof_edge",
    "eaves": "roof_edge",
    "fascia": "roof_edge",
    "gate_area": "gate",
    "gates": "gate",
    "railings": "railing",
    "balustrade": "railing",
    "windows": "window",
    "doors": "door",
    "main_door": "door",
    "entrance": "door",
    "balconies": "balcony",
}


def normalise_kind(raw: str) -> str | None:
    key = (raw or "").strip().lower().replace(" ", "_").replace("-", "_")
    key = KIND_SYNONYMS.get(key, key)
    return key if key in ALL_KINDS else None


@dataclass
class Region:
    kind: str
    x0: float
    y0: float
    x1: float
    y1: float
    label: str = ""
    confidence: float = 1.0
    source: str = "manual"
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    def __post_init__(self) -> None:
        # Clamp and order coordinates so downstream code can trust them.
        xs = sorted([min(max(float(self.x0), 0.0), 1.0), min(max(float(self.x1), 0.0), 1.0)])
        ys = sorted([min(max(float(self.y0), 0.0), 1.0), min(max(float(self.y1), 0.0), 1.0)])
        self.x0, self.x1 = xs
        self.y0, self.y1 = ys
        if not self.label:
            self.label = KIND_LABELS.get(self.kind, self.kind)

    @property
    def is_opening(self) -> bool:
        return self.kind in OPENING_KINDS

    @property
    def is_valid(self) -> bool:
        return self.kind in ALL_KINDS and (self.x1 - self.x0) > 0.005 and (self.y1 - self.y0) > 0.005

    def px_box(self, width: int, height: int) -> tuple[int, int, int, int]:
        return (
            int(round(self.x0 * width)),
            int(round(self.y0 * height)),
            int(round(self.x1 * width)),
            int(round(self.y1 * height)),
        )

    def px_size(self, width: int, height: int) -> tuple[int, int]:
        x0, y0, x1, y1 = self.px_box(width, height)
        return max(x1 - x0, 0), max(y1 - y0, 0)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Region":
        kind = normalise_kind(d.get("kind", ""))
        if kind is None:
            raise ValueError(f"Unknown component kind: {d.get('kind')!r}")
        return cls(
            kind=kind,
            x0=d["x0"],
            y0=d["y0"],
            x1=d["x1"],
            y1=d["y1"],
            label=d.get("label", "") or "",
            confidence=float(d.get("confidence", 1.0) or 1.0),
            source=d.get("source", "manual") or "manual",
            id=d.get("id") or uuid.uuid4().hex[:8],
        )


def box_mask(region: Region, width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    x0, y0, x1, y1 = region.px_box(width, height)
    mask[y0:y1, x0:x1] = True
    return mask


def refine_with_grabcut(image_rgb: np.ndarray, region: Region, iterations: int = 3) -> np.ndarray:
    """Tighten a box to the actual surface with OpenCV GrabCut.

    Falls back to the plain box when GrabCut fails or returns an implausibly
    small mask (it can latch onto a single texture patch on real photos).
    """
    import cv2

    h, w = image_rgb.shape[:2]
    base = box_mask(region, w, h)
    x0, y0, x1, y1 = region.px_box(w, h)
    if (x1 - x0) < 20 or (y1 - y0) < 20:
        return base
    # Work on a downscaled copy for speed.
    scale = min(1.0, 640 / max(w, h))
    small = cv2.resize(image_rgb, (max(int(w * scale), 1), max(int(h * scale), 1)), interpolation=cv2.INTER_AREA)
    rect = (int(x0 * scale), int(y0 * scale), max(int((x1 - x0) * scale), 2), max(int((y1 - y0) * scale), 2))
    gc_mask = np.zeros(small.shape[:2], np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(cv2.cvtColor(small, cv2.COLOR_RGB2BGR), gc_mask, rect, bgd, fgd, iterations, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return base
    fg = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
    fg = cv2.resize(fg, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool) & base
    if fg.sum() < 0.6 * base.sum():
        return base
    return fg


def build_masks(
    regions: list[Region],
    width: int,
    height: int,
    image_rgb: np.ndarray | None = None,
    refine_kinds: tuple[str, ...] = (),
) -> dict[str, np.ndarray]:
    """Return one exclusive boolean mask per region id.

    Pixels are assigned to the highest-priority region that covers them; ties
    go to the smaller box (a more specific annotation).
    """
    valid = [r for r in regions if r.is_valid]
    owner = np.full((height, width), -1, dtype=np.int32)
    best_priority = np.full((height, width), -1, dtype=np.int32)
    best_size = np.full((height, width), np.iinfo(np.int32).max, dtype=np.int64)

    raw: list[np.ndarray] = []
    for r in valid:
        if image_rgb is not None and r.kind in refine_kinds:
            raw.append(refine_with_grabcut(image_rgb, r))
        else:
            raw.append(box_mask(r, width, height))

    for idx, (r, m) in enumerate(zip(valid, raw)):
        pr = PRIORITY.get(r.kind, 0)
        size = int(m.sum())
        better = m & ((pr > best_priority) | ((pr == best_priority) & (size < best_size)))
        owner[better] = idx
        best_priority[better] = pr
        best_size[better] = size

    return {r.id: owner == idx for idx, r in enumerate(valid)}
