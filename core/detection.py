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
- wall: the main plastered/painted wall surface of each floor. Use ONE box per floor spanning the full
  width of that floor's wall; do not split a wall around windows or doors (openings are subtracted later)
- window: each window opening including its frame
- door: each door opening (main door, balcony door)
- balcony: the projecting balcony slab edge / fascia below a railing (a thin horizontal strip), not the railing
- railing: each railing or balustrade run (metal, glass or grill)
- pillar: each pillar or column, including narrow full-height corner columns at the left and right ends
  of the facade that look slightly different from the wall
- parapet: the low solid wall running along the top of the flat roof (the topmost horizontal band of the building)
- roof_edge: horizontal floor-slab bands between storeys, chajjas / sunshades above windows, sloped roof eaves
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


def friendly_error(provider: str, exc: Exception) -> str:
    """Turn raw SDK errors into a short message a homeowner can act on."""
    msg = str(exc)
    low = msg.lower()
    if "429" in msg or "resource_exhausted" in low or "quota" in low:
        if "limit: 0" in low:
            return (f"{provider}: this API key has no quota for this model on the free tier. "
                    "Use a key with billing enabled for this feature.")
        return f"{provider}: rate limit or quota reached. Wait a minute and try again."
    if any(c in msg for c in ("401", "403")) or "api key" in low or "permission" in low:
        return f"{provider}: the API key was rejected. Check the key in the sidebar or the app secrets."
    if "404" in msg or "not found" in low:
        return f"{provider}: model not available for this key. Set GEMINI_MODEL / OPENAI_MODEL to a model you can use."
    return f"{provider} request failed: {msg[:300]}"


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

    # JSON mode and image output are only enabled on v1beta for this SDK version.
    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(api_version="v1beta"))
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


TRANSIENT = ("503", "500", "unavailable", "overloaded", "high demand", "deadline", "timeout", "timed out")


def _is_transient(exc: Exception) -> bool:
    low = str(exc).lower()
    return any(t in low for t in TRANSIENT)


def _model_chain(provider: str, model: str | None) -> list[str]:
    if model:
        return [model]
    if provider == "gemini":
        primary = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
        fallbacks = os.getenv("GEMINI_FALLBACK_MODELS", "gemini-2.5-flash")
    else:
        primary = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        fallbacks = os.getenv("OPENAI_FALLBACK_MODELS", "")
    chain = [primary] + [m.strip() for m in fallbacks.split(",") if m.strip()]
    return list(dict.fromkeys(chain))


def detect_components(image_rgb: np.ndarray, provider: str, api_key: str, model: str | None = None,
                      retries: int = 1) -> tuple[list[Region], dict]:
    """Call the provider and return (regions, run_info).

    Malformed model output is retried once on the same model. Transient
    server errors (overloaded / 5xx) move on to the next model in the chain.
    Auth and quota errors stop immediately with a readable message.
    """
    call = {"gemini": _call_gemini, "openai": _call_openai}.get(provider)
    if call is None:
        raise DetectionError(f"Unknown provider {provider!r}")
    jpeg = _jpeg_bytes(image_rgb)
    last_msg = "No model was tried."
    tried: list[str] = []
    for m in _model_chain(provider, model):
        tried.append(m)
        for attempt in range(retries + 1):
            t0 = time.perf_counter()
            try:
                text = call(jpeg, api_key, m)
                regions = parse_components(text)
                return regions, {"provider": provider, "model": m, "seconds": round(time.perf_counter() - t0, 1),
                                 "count": len(regions), "attempts": attempt + 1, "models_tried": tried}
            except DetectionError as exc:
                last_msg = str(exc)  # malformed output: retry same model
                continue
            except Exception as exc:
                if _is_transient(exc):
                    log.warning("%s %s transient error, trying next model: %s", provider, m, exc)
                    last_msg = f"{provider} ({m}) is temporarily overloaded. Please try again in a minute."
                    break
                raise DetectionError(friendly_error(provider, exc)) from exc
    raise DetectionError(last_msg)
