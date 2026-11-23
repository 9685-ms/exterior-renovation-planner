"""Upload handling and image-quality gate.

Checks a photo before any AI call so the user gets fast, specific guidance
instead of a poor result: resolution, blur (variance of the Laplacian),
exposure, and orientation. When several photos are uploaded the clearest
usable one is suggested as the working view.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageOps

MAX_SIDE = 1280          # working resolution; keeps rendering and API calls fast
MIN_SHORT_SIDE = 400     # below this, components are too small to map reliably
BLUR_REJECT = 25.0       # Laplacian variance (on a 800px copy) under this = too blurry
BLUR_WARN = 80.0
DARK_REJECT, DARK_WARN = 35.0, 60.0
BRIGHT_REJECT, BRIGHT_WARN = 235.0, 210.0


@dataclass
class QualityReport:
    usable: bool
    score: float                 # 0..100, used to rank multiple uploads
    width: int
    height: int
    sharpness: float
    brightness: float
    issues: list[str] = field(default_factory=list)
    tips: list[str] = field(default_factory=list)


def load_image(data: bytes) -> np.ndarray:
    """Decode bytes, apply EXIF rotation, convert to RGB and downscale."""
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    return np.array(img)


def assess(image_rgb: np.ndarray, original_size: tuple[int, int] | None = None) -> QualityReport:
    h, w = image_rgb.shape[:2]
    ow, oh = original_size or (w, h)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    scale = 800 / max(h, w)
    g = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else gray
    sharp = float(cv2.Laplacian(g, cv2.CV_64F).var())
    bright = float(gray.mean())

    issues, tips = [], []
    usable = True
    if min(ow, oh) < MIN_SHORT_SIDE:
        usable = False
        issues.append(f"Resolution too low ({ow}x{oh}px).")
        tips.append(f"Use a photo at least {MIN_SHORT_SIDE}px on the short side; a normal phone photo is fine.")
    if sharp < BLUR_REJECT:
        usable = False
        issues.append("Image is too blurry to find edges of walls and windows.")
        tips.append("Hold the phone steady, tap to focus on the house, avoid zoomed-in shots.")
    elif sharp < BLUR_WARN:
        issues.append("Image is slightly soft; detection and area estimates may be less precise.")
    if bright < DARK_REJECT:
        usable = False
        issues.append("Image is too dark.")
        tips.append("Take the photo in daylight, ideally on an overcast day or with the sun behind you.")
    elif bright < DARK_WARN:
        issues.append("Image is quite dark; colours in the preview may look muddy.")
    if bright > BRIGHT_REJECT:
        usable = False
        issues.append("Image is over-exposed (washed out).")
        tips.append("Avoid shooting into the sun; step to the side so the facade is evenly lit.")
    elif bright > BRIGHT_WARN:
        issues.append("Image is bright; some surface detail may be lost.")
    if h > w * 1.6:
        tips.append("A landscape photo that shows the whole front of the house usually works best.")
    tips.append("Stand straight in front of the facade so the walls look rectangular, not slanted.")

    sharp_score = min(sharp / 300.0, 1.0) * 50
    expo_score = max(0.0, 1 - abs(bright - 128) / 128) * 30
    res_score = min(min(ow, oh) / 1080, 1.0) * 20
    score = round(sharp_score + expo_score + res_score, 1) if usable else 0.0
    return QualityReport(usable, score, ow, oh, round(sharp, 1), round(bright, 1), issues, tips)


def original_size(data: bytes) -> tuple[int, int]:
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(data)))
    return img.size
