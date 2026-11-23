"""End-to-end UI test using Streamlit's headless AppTest runner."""
from streamlit.testing.v1 import AppTest

TIMEOUT = 60


def _start():
    at = AppTest.from_file("../app.py", default_timeout=TIMEOUT)
    at.run()
    assert not at.exception
    return at


def test_empty_state_prompts_upload():
    at = _start()
    assert any("Start in" in i.value for i in at.info)


def test_sample_flow_end_to_end():
    at = _start()
    next(b for b in at.button if b.label == "Load the sample house").click().run()
    assert not at.exception, at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Grand total"].startswith("₹")
    total_a = metrics["Grand total"]

    # Switch to Design B (Modern preset) -> different total.
    design_box = next(s for s in at.selectbox if s.label == "Design option")
    design_box.select("Design B").run()
    assert not at.exception
    total_b = {m.label: m.value for m in at.metric}["Grand total"]
    assert total_b != total_a

    # Change the scale with a measurement -> scale method switches.
    at.number_input(key="set_known_width").set_value(74.0).run()
    assert not at.exception
    assert {m.label: m.value for m in at.metric}["Scale used"] == "10.0 px/ft"

    # Build the PDF.
    next(b for b in at.button if b.label == "Prepare PDF report").click().run()
    assert not at.exception
    assert at.session_state["report_pdf"][:4] == b"%PDF"


def test_change_material_and_rate():
    at = _start()
    next(b for b in at.button if b.label == "Load the sample house").click().run()
    before = {m.label: m.value for m in at.metric}["Grand total"]
    # Ground floor wall -> stone cladding
    wall_box = next(s for s in at.selectbox if s.label == "Ground floor wall")
    wall_box.select("stone_cladding").run()
    assert not at.exception
    after = {m.label: m.value for m in at.metric}["Grand total"]
    assert after != before
    # Apply a preset.
    next(s for s in at.selectbox if s.label == "Style preset").select("Budget refresh").run()
    next(b for b in at.button if b.label == "Apply preset to this design").click().run()
    assert not at.exception
    assert {m.label: m.value for m in at.metric}["Grand total"] != after
