"""Optional photoreal pass with a Gemini image-editing model.

The deterministic renderer (render.py) is the source of truth: it keeps the
building geometry exactly, so the preview always matches the measured areas.
This optional step sends the original photo plus that preview to an image
model and asks it to make the preview look photographic without moving
anything. Generative models can still alter geometry, so the UI labels the
result as an artist's impression and the report keeps the deterministic
preview.
"""
from __future__ import annotations

import io
import os

import numpy as np
from PIL import Image

from .detection import DetectionError


def build_prompt(material_lines: list[str]) -> str:
    spec = "\n".join(f"- {line}" for line in material_lines) or "- (no materials selected)"
    return (
        "Image 1 is a photo of a house exterior. Image 2 is a flat mock-up of the same house with new finishes.\n"
        "Produce a photorealistic version of image 2: keep the exact camera angle, building outline, windows, "
        "doors, pillars, balconies and surroundings from image 1, and apply these finishes with realistic lighting, "
        "shadows and material texture:\n"
        f"{spec}\n"
        "Do not add or remove any building element, do not change the sky, trees or street."
    )


def photoreal_render(original_rgb: np.ndarray, preview_rgb: np.ndarray, material_lines: list[str],
                     api_key: str, model: str | None = None) -> np.ndarray:
    from google import genai
    from google.genai import types

    model = model or os.getenv("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image")

    def _png(arr: np.ndarray) -> bytes:
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        return buf.getvalue()

    client = genai.Client(api_key=api_key)
    try:
        resp = client.models.generate_content(
            model=model,
            contents=[
                types.Part.from_bytes(data=_png(original_rgb), mime_type="image/png"),
                types.Part.from_bytes(data=_png(preview_rgb), mime_type="image/png"),
                build_prompt(material_lines),
            ],
            config=types.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"]),
        )
    except Exception as exc:
        raise DetectionError(f"Image model request failed: {exc}") from exc
    for cand in resp.candidates or []:
        for part in (cand.content.parts if cand.content else []) or []:
            if getattr(part, "inline_data", None) and part.inline_data.data:
                img = Image.open(io.BytesIO(part.inline_data.data)).convert("RGB")
                img = img.resize((original_rgb.shape[1], original_rgb.shape[0]), Image.LANCZOS)
                return np.array(img)
    raise DetectionError("The image model did not return an image.")
