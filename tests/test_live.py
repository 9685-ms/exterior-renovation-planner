"""Live API tests. Opt-in: set RUN_LIVE_TESTS=1 and a GEMINI_API_KEY (e.g. in .env)."""
import os

import pytest
from dotenv import load_dotenv
from streamlit.testing.v1 import AppTest

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
pytestmark = pytest.mark.skipif(
    not (os.getenv("RUN_LIVE_TESTS") == "1" and os.getenv("GEMINI_API_KEY")),
    reason="live tests disabled (set RUN_LIVE_TESTS=1 and GEMINI_API_KEY)",
)


def test_live_detection_on_sample_via_ui():
    at = AppTest.from_file("../app.py", default_timeout=180)
    at.run()
    next(b for b in at.button if b.label == "Load the sample house").click().run()
    next(b for b in at.button if "Detect components with AI" in b.label).click().run()
    assert not at.exception, at.exception
    assert not at.error, [e.value for e in at.error]
    regs = at.session_state["regions"]
    kinds = {r.kind for r in regs}
    assert {"wall", "window", "door"} <= kinds
    assert at.session_state["detect_info"]["count"] >= 8
    scale = {m.label: m.value for m in at.metric}["Scale used"]
    px_per_ft = float(scale.split()[0])
    assert 17 <= px_per_ft <= 23, scale  # true scale is 20 px/ft
