"""Exterior structure identification with a vision LLM.

The model returns bounding boxes for each building component in a strict
JSON shape. The response is validated with Pydantic, synonyms are mapped to
the fixed component vocabulary, and bad boxes are dropped. Users then review
and correct every box in the UI, so the model only has to be roughly right.

Providers: Google Gemini (recommended: Gemini models are trained to return
`box_2d` coordinates) and OpenAI (works, boxes are usually looser).
Without an API key the app still works with the bundled sample or with
boxes the user adds by hand.
"""
from __future__ import annotations

import io
import json
import logging
import os
import re
import time

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field, ValidationError, field_validator

from .regions import Region, normalise_kind

log = logging.getLogger(__name__)

PROMPT = """You are mapping the exterior of a low-rise residential building for a renovation estimate.
Detect every visible instance of these components on the house facade:
- wall: large plain plastered/painted wall surfaces (one box per distinct wall plane or floor band)
- window: each window opening including its frame
- door: each door opening (main door, balcony door)
- balcony: the balcony slab edge / fascia and balcony parapet (not the railing)
- railing: each railing or balustrade run (metal, glass or grill)
- pillar: each pillar or column
- parapet: parapet wall along the roof line
- roof_edge: roof edges, chajjas / sunshades above windows, sloped roof eaves
- gate: the compound gate and gate area if visible
Ignore trees, cars, people, sky, road and neighbouring buildings.

Return ONLY JSON of the form:
{"components": [{"kind": "<one of the kinds above>", "label": "<short description, e.g. 'Ground floor wall'>",
"box_2d": [ymin, xmin, ymax, xmax], "confidence": 0.0-1.0}]}
Coordinates are integers normalised to 0-1000 relative to image height (y) and width (x).
Boxes must be tight. Prefer several smaller accurate boxes over one large loose box."""


class _Component(BaseModel):
    kind: str
    label: str = ""
    box_2d: list[float] = Field(min_length=4, max_length=4)
    confidence: float = 0.7

    @field_validator("confidence", mode="before")
    @classmethod
    def _conf(cls, v):
        try:
            return min(max(float(v), 0.0), 1.0)
        except (TypeError, ValueError):
            return 0.5


class _Response(BaseModel):
    components: list[_Component]


class DetectionError(RuntimeError):
    pass


def _extract_json(text: str) -> dict:
    text = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1:
            # Some models return a bare list.
            start, end = text.find("["), text.rfind("]")
            if start == -1:
                raise DetectionError("Model did not return JSON.")
        data = json.loads(text[start:end + 1])
    if isinstance(data, list):
        data = {"components": data}
    return data


def parse_components(text: str) -> list[Region]:
    """Turn raw model output into validated Regions (0-1 coordinates)."""
    try:
        resp = _Response.model_validate(_extract_json(text))
    except (ValidationError, json.JSONDecodeError) as exc:
        raise DetectionError(f"Could not read the model's answer: {exc}") from exc
    regions: list[Region] = []
    for c in resp.components:
        kind = normalise_kind(c.kind)
        if kind is None:
            log.info("Dropping unknown kind %r", c.kind)
            continue
        ymin, xmin, ymax, xmax = c.box_2d
        # Accept 0-1 floats as well as 0-1000 ints.
        div = 1.0 if max(abs(v) for v in c.box_2d) <= 1.0 else 1000.0
        r = Region(kind=kind, x0=xmin / div, y0=ymin / div, x1=xmax / div, y1=ymax / div,
                   label=c.label.strip()[:60], confidence=c.confidence, source="ai")
        if r.is_valid:
            regions.append(r)
    if not regions:
        raise DetectionError("No building components were detected. Try a clearer, front-on photo or add boxes manually.")
    return regions


def _jpeg_bytes(image_rgb: np.ndarray, max_side: int = 1024) -> bytes:
    img = Image.fromarray(image_rgb)
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return buf.getvalue()


def available_providers(keys: dict[str, str | None]) -> list[str]:
    out = []
    if keys.get("GEMINI_API_KEY"):
        out.append("gemini")
    if keys.get("OPENAI_API_KEY"):
        out.append("openai")
    return out


def _call_gemini(image_jpeg: bytes, api_key: str, model: str) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(
        model=model,
        contents=[types.Part.from_bytes(data=image_jpeg, mime_type="image/jpeg"), PROMPT],
        config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.2),
    )
    return resp.text or ""


def _call_openai(image_jpeg: bytes, api_key: str, model: str) -> str:
    import base64

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    b64 = base64.b64encode(image_jpeg).decode()
    resp = client.chat.completions.create(
        model=model,
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"}},
            ],
        }],
    )
    return resp.choices[0].message.content or ""


def detect_components(image_rgb: np.ndarray, provider: str, api_key: str, model: str | None = None,
                      retries: int = 1) -> tuple[list[Region], dict]:
    """Call the provider and return (regions, run_info)."""
    model = model or (os.getenv("GEMINI_MODEL", "gemini-2.5-flash") if provider == "gemini"
                      else os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))
    call = {"gemini": _call_gemini, "openai": _call_openai}.get(provider)
    if call is None:
        raise DetectionError(f"Unknown provider {provider!r}")
    jpeg = _jpeg_bytes(image_rgb)
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        t0 = time.perf_counter()
        try:
            text = call(jpeg, api_key, model)
            regions = parse_components(text)
            return regions, {"provider": provider, "model": model, "seconds": round(time.perf_counter() - t0, 1),
                             "count": len(regions), "attempts": attempt + 1}
        except DetectionError as exc:
            last_exc = exc  # malformed output: worth one retry
        except Exception as exc:  # network / auth / quota: surface immediately
            raise DetectionError(f"{provider} request failed: {exc}") from exc
    raise DetectionError(str(last_exc))
