"""A synthetic two-storey house with known dimensions.

Used for the no-API-key demo and for tests: because the true size of every
component is known (20 px = 1 ft at 1200x800), the area estimator can be
checked against ground truth.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .regions import Region

W, H = 1200, 800
PX_PER_FT = 20.0

# (kind, label, x0, y0, x1, y1) in pixels.
LAYOUT = [
    ("parapet", "Roof parapet", 200, 200, 1000, 240),
    ("wall", "First floor wall", 230, 240, 970, 430),
    ("wall", "Ground floor wall", 230, 448, 970, 640),
    ("roof_edge", "Floor slab band", 200, 430, 1000, 448),
    ("pillar", "Left corner pillar", 200, 240, 230, 640),
    ("pillar", "Right corner pillar", 970, 240, 1000, 640),
    ("balcony", "Balcony fascia", 480, 430, 720, 448),
    ("railing", "Balcony railing", 480, 370, 720, 430),
    ("window", "First floor window L", 300, 300, 380, 380),
    ("window", "First floor window R", 780, 300, 860, 380),
    ("door", "Balcony door", 570, 290, 630, 430),
    ("roof_edge", "Chajja L", 290, 488, 390, 500),
    ("roof_edge", "Chajja R", 770, 488, 870, 500),
    ("window", "Ground floor window L", 300, 500, 380, 580),
    ("window", "Ground floor window R", 780, 500, 860, 580),
    ("door", "Main door", 560, 500, 620, 640),
    ("gate", "Compound gate", 1030, 560, 1160, 640),
]


def sample_regions() -> list[Region]:
    return [
        Region(kind=k, label=l, x0=x0 / W, y0=y0 / H, x1=x1 / W, y1=y1 / H, confidence=1.0, source="sample", id=f"s{i:02d}")
        for i, (k, l, x0, y0, x1, y1) in enumerate(LAYOUT)
    ]


def _plaster(h: int, w: int, color, rng, light_from_top=True):
    base = np.ones((h, w, 3), np.float32) * np.array(color, np.float32)
    noise = rng.normal(0, 4.5, (h, w, 1)).astype(np.float32)
    grad = np.linspace(1.04, 0.93, h, dtype=np.float32)[:, None, None] if light_from_top else 1.0
    return np.clip(base * grad + noise, 0, 255)


def sample_image() -> np.ndarray:
    rng = np.random.default_rng(7)
    img = np.zeros((H, W, 3), np.float32)
    # Sky and ground.
    sky = np.linspace(0, 1, 640, dtype=np.float32)[:, None]
    img[:640] = (np.array([150, 190, 225]) * (1 - sky) + np.array([215, 228, 238]) * sky)[:, None, :]
    img[640:] = np.array([120, 116, 108]) + rng.normal(0, 6, (H - 640, W, 1))
    img[640:700, 120:1200] = np.array([150, 146, 138]) + rng.normal(0, 5, (60, 1080, 1))  # driveway

    def fill(x0, y0, x1, y1, color, light=True):
        img[y0:y1, x0:x1] = _plaster(y1 - y0, x1 - x0, color, rng, light)

    fill(200, 200, 1000, 240, (205, 196, 178))           # parapet
    img[196:204, 196:1004] = (180, 172, 160)              # coping
    fill(230, 240, 970, 430, (222, 210, 186))             # first floor wall
    fill(230, 448, 970, 640, (222, 210, 186))             # ground floor wall
    fill(200, 430, 1000, 448, (196, 188, 172), False)     # slab band
    fill(200, 240, 230, 640, (198, 186, 165))             # pillars
    fill(970, 240, 1000, 640, (198, 186, 165))
    img[448:462, 230:970] *= 0.82                         # slab shadow
    img[640:652, 200:1000] *= 0.75                        # plinth shadow
    # Compound wall and gate.
    fill(1000, 580, 1030, 640, (190, 180, 165))
    fill(1160, 580, 1200, 640, (190, 180, 165))
    img[560:640, 1030:1160] = (225, 222, 215)

    pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(pil)

    def window(x0, y0, x1, y1):
        d.rectangle([x0, y0, x1, y1], fill=(245, 245, 240))
        d.rectangle([x0 + 5, y0 + 5, x1 - 5, y1 - 5], fill=(70, 92, 110))
        d.line([(x0 + (x1 - x0) // 2, y0 + 5), (x0 + (x1 - x0) // 2, y1 - 5)], fill=(245, 245, 240), width=4)
        d.rectangle([x0 + 8, y0 + 8, x0 + 20, y1 - 30], fill=(105, 128, 146))  # reflection

    for x0, y0, x1, y1 in [(300, 300, 380, 380), (780, 300, 860, 380), (300, 500, 380, 580), (780, 500, 860, 580)]:
        window(x0, y0, x1, y1)
    for x0, y0, x1, y1 in [(290, 488, 390, 500), (770, 488, 870, 500)]:
        d.rectangle([x0, y0, x1, y1], fill=(190, 182, 168))
        d.rectangle([x0 + 10, y1, x1 - 10, y1 + 6], fill=(170, 160, 145))
    # Doors.
    d.rectangle([560, 500, 620, 640], fill=(110, 70, 42))
    d.rectangle([566, 508, 614, 566], outline=(85, 52, 30), width=3)
    d.rectangle([566, 574, 614, 634], outline=(85, 52, 30), width=3)
    d.ellipse([606, 568, 612, 574], fill=(210, 180, 90))
    d.rectangle([570, 290, 630, 430], fill=(245, 245, 240))
    d.rectangle([576, 296, 624, 430], fill=(78, 98, 114))
    # Balcony railing.
    d.rectangle([480, 370, 720, 376], fill=(40, 40, 42))
    for x in range(484, 720, 14):
        d.rectangle([x, 376, x + 3, 430], fill=(45, 45, 48))
    # Gate bars.
    d.rectangle([1030, 560, 1160, 566], fill=(55, 58, 62))
    for x in range(1034, 1160, 10):
        d.rectangle([x, 566, x + 4, 640], fill=(60, 63, 68))
    d.line([(1095, 560), (1095, 640)], fill=(40, 42, 45), width=4)
    pil = pil.filter(ImageFilter.GaussianBlur(0.6))
    return np.array(pil)
