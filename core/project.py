"""Save / load a renovation project as a single JSON file.

The file carries the working photo, the reviewed regions, every design
option, rate overrides and estimation settings, so a user can close the
browser and continue editing later, or hand the file to a contractor.
"""
from __future__ import annotations

import base64
import io
import json
from datetime import datetime, timezone

import numpy as np
from PIL import Image

from .regions import Region

FORMAT_VERSION = 1
MAX_PROJECT_BYTES = 15 * 1024 * 1024


def image_to_b64(image_rgb: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(image_rgb).save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode()


def b64_to_image(data: str) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB"))


def dump_project(name: str, image_rgb: np.ndarray, regions: list[Region], designs: dict, active_design: str,
                 rate_overrides: dict, settings: dict) -> bytes:
    payload = {
        "format": "renovation-project",
        "version": FORMAT_VERSION,
        "name": name,
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "image_jpeg_b64": image_to_b64(image_rgb),
        "regions": [r.to_dict() for r in regions],
        "designs": designs,
        "active_design": active_design,
        "rate_overrides": rate_overrides,
        "settings": settings,
    }
    return json.dumps(payload, indent=1).encode("utf-8")


def load_project(data: bytes) -> dict:
    if len(data) > MAX_PROJECT_BYTES:
        raise ValueError("Project file is too large.")
    payload = json.loads(data.decode("utf-8"))
    if payload.get("format") != "renovation-project":
        raise ValueError("This is not a renovation project file.")
    if int(payload.get("version", 0)) > FORMAT_VERSION:
        raise ValueError("Project was saved by a newer version of the app.")
    regions = [Region.from_dict(r) for r in payload.get("regions", [])]
    designs = payload.get("designs") or {"Design A": {}}
    # Only keep string keys / dict values in designs.
    clean = {str(k): {str(rid): dict(v) for rid, v in (d or {}).items() if isinstance(v, dict)} for k, d in designs.items()}
    return {
        "name": str(payload.get("name") or "My house"),
        "image": b64_to_image(payload["image_jpeg_b64"]),
        "regions": regions,
        "designs": clean,
        "active_design": payload.get("active_design") if payload.get("active_design") in clean else next(iter(clean)),
        "rate_overrides": payload.get("rate_overrides") or {},
        "settings": payload.get("settings") or {},
    }
