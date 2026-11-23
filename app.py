"""Streamlit UI: AI exterior renovation planner and cost estimator.

Flow: 1 Upload photo -> 2 Detect & review components -> 3 Apply materials
and compare designs -> 4 Measure & estimate cost -> 5 Download report /
save project.
"""
from __future__ import annotations

import hashlib
import io
import os
import uuid

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from PIL import Image, ImageDraw

from core import pipeline
from core.ai_render import photoreal_render
from core.catalog import EDITABLE_FIELDS, load_catalog, materials_for, rates_note
from core.costing import inr
from core.detection import DetectionError, available_providers, detect_components
from core.estimation import DEFAULT_SURFACE_FACTORS, METHOD_LABELS
from core.presets import PRESETS, apply_preset
from core.project import dump_project, load_project
from core.quality import assess, load_image, original_size
from core.regions import ALL_KINDS, KIND_LABELS, SURFACE_KINDS, Region, build_masks
from core.render import draw_overlay
from core.report import build_report
from core.sample import sample_image, sample_regions

load_dotenv()
st.set_page_config(page_title="Exterior Renovation Planner", page_icon="🏠", layout="wide")

CATALOG = load_catalog()
AI_CALL_LIMIT = int(os.getenv("AI_CALLS_PER_SESSION", "12"))


def secret(name: str) -> str | None:
    val = os.getenv(name)
    if val:
        return val
    try:
        return st.secrets.get(name)  # Streamlit Cloud / HF secrets
    except Exception:
        return None


# --------------------------------------------------------------------------- state
def init_state() -> None:
    ss = st.session_state
    ss.setdefault("project_name", "My house")
    ss.setdefault("image", None)
    ss.setdefault("image_source", None)
    ss.setdefault("regions", [])
    ss.setdefault("regions_version", 0)
    ss.setdefault("regions_base", [])
    ss.setdefault("designs", {"Design A": {}})
    ss.setdefault("active_design", "Design A")
    ss.setdefault("design_rev", 0)
    ss.setdefault("rate_overrides", {})
    ss.setdefault("rates_version", 0)
    ss.setdefault("ai_calls", 0)
    ss.setdefault("detect_info", None)
    ss.setdefault("photoreal", {})
    ss.setdefault("loaded_project_sig", None)
    defaults = {
        "set_floors": 0, "set_known_width": 0.0, "set_known_region": "(largest wall)", "set_scale_method": "auto",
        "set_facade_mult": 1.0, "set_railing_h": 3.5, "set_contingency": 5.0, "set_gst": 18.0, "set_refine": False,
    }
    for k, v in defaults.items():
        ss.setdefault(k, v)
    for kind, f in DEFAULT_SURFACE_FACTORS.items():
        ss.setdefault(f"sf_{kind}", float(f))


def set_regions(regions: list[Region], source: str) -> None:
    ss = st.session_state
    ss.regions = regions
    ss.regions_base = regions
    ss.regions_version += 1
    ss.image_source = source


def reset_designs_with_default(regions: list[Region]) -> None:
    ss = st.session_state
    ss.designs = {
        "Design A": apply_preset("Classic warm", regions, CATALOG),
        "Design B": apply_preset("Modern contemporary", regions, CATALOG),
    }
    ss.active_design = "Design A"
    ss.design_rev += 1
    ss.photoreal = {}


def settings_from_state() -> dict:
    ss = st.session_state
    region_id = None
    if ss.set_known_region != "(largest wall)":
        region_id = ss.set_known_region.split(" | ")[0]
    return {
        "known_width_ft": float(ss.set_known_width or 0),
        "known_width_region_id": region_id,
        "floors": int(ss.set_floors or 0),
        "scale_method": ss.set_scale_method,
        "facade_multiplier": float(ss.set_facade_mult),
        "surface_factors": {k: float(ss[f"sf_{k}"]) for k in DEFAULT_SURFACE_FACTORS},
        "railing_height_ft": float(ss.set_railing_h),
        "contingency_pct": float(ss.set_contingency),
        "gst_pct": float(ss.set_gst),
        "refine_edges": bool(ss.set_refine),
    }


def apply_loaded_project(p: dict) -> None:
    ss = st.session_state
    ss.project_name = p["name"]
    ss.image = p["image"]
    set_regions(p["regions"], "project file")
    ss.designs = p["designs"]
    ss.active_design = p["active_design"]
    ss.design_rev += 1
    ss.rate_overrides = p["rate_overrides"]
    ss.rates_version += 1
    ss.photoreal = {}
    s = p["settings"]
    mapping = {
        "known_width_ft": "set_known_width", "floors": "set_floors", "scale_method": "set_scale_method",
        "facade_multiplier": "set_facade_mult", "railing_height_ft": "set_railing_h",
        "contingency_pct": "set_contingency", "gst_pct": "set_gst", "refine_edges": "set_refine",
    }
    for src, dst in mapping.items():
        if src in s and s[src] is not None:
            ss[dst] = type(ss[dst])(s[src])
    for kind, val in (s.get("surface_factors") or {}).items():
        if f"sf_{kind}" in ss:
            ss[f"sf_{kind}"] = float(val)
    rid = s.get("known_width_region_id")
    match = next((r for r in p["regions"] if r.id == rid), None)
    ss.set_known_region = f"{match.id} | {match.label}" if match else "(largest wall)"


@st.cache_data(show_spinner=False, max_entries=32)
def cached_masks(image: np.ndarray, regions_key: tuple, refine: bool) -> dict:
    regions = [Region(kind=k, x0=a, y0=b, x1=c, y1=d, id=i) for (i, k, a, b, c, d) in regions_key]
    h, w = image.shape[:2]
    return build_masks(regions, w, h, image_rgb=image, refine_kinds=("wall", "parapet") if refine else ())


def regions_key(regions: list[Region]) -> tuple:
    return tuple((r.id, r.kind, round(r.x0, 4), round(r.y0, 4), round(r.x1, 4), round(r.y1, 4)) for r in regions)


def add_grid(img: np.ndarray) -> np.ndarray:
    pil = Image.fromarray(img)
    d = ImageDraw.Draw(pil)
    w, h = pil.size
    for i in range(1, 10):
        x, y = int(w * i / 10), int(h * i / 10)
        d.line([(x, 0), (x, h)], fill=(255, 255, 255), width=1)
        d.line([(0, y), (w, y)], fill=(255, 255, 255), width=1)
        d.text((x + 2, 2), f"{i * 10}", fill=(255, 255, 0))
        d.text((2, y + 2), f"{i * 10}", fill=(255, 255, 0))
    return np.array(pil)


init_state()
ss = st.session_state

# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("🏠 Renovation Planner")
    st.text_input("Project name", key="project_name")

    st.subheader("Open a saved project")
    proj_file = st.file_uploader("Project file (.json)", type=["json"], key="proj_upload", label_visibility="collapsed")
    if proj_file is not None:
        sig = hashlib.md5(proj_file.getvalue()).hexdigest()
        if sig != ss.loaded_project_sig:
            try:
                apply_loaded_project(load_project(proj_file.getvalue()))
                ss.loaded_project_sig = sig
                st.success("Project loaded.")
                st.rerun()
            except Exception as exc:
                st.error(f"Could not open project: {exc}")

    st.subheader("AI settings")
    access_code = secret("APP_ACCESS_CODE")
    server_keys = {"GEMINI_API_KEY": secret("GEMINI_API_KEY"), "OPENAI_API_KEY": secret("OPENAI_API_KEY")}
    server_ok = True
    if access_code and any(server_keys.values()):
        entered = st.text_input("Access code for the hosted AI key", type="password",
                                help="Ask the app owner for the code, or paste your own API key below.")
        server_ok = entered == access_code
        if entered and not server_ok:
            st.warning("Access code not recognised.")
    with st.expander("Use your own API key (optional)"):
        st.caption("Kept only in this browser session; never saved to the project file.")
        user_gemini = st.text_input("Gemini API key", type="password", key="user_gemini")
        user_openai = st.text_input("OpenAI API key", type="password", key="user_openai")
    keys = {
        "GEMINI_API_KEY": user_gemini or (server_keys["GEMINI_API_KEY"] if server_ok else None),
        "OPENAI_API_KEY": user_openai or (server_keys["OPENAI_API_KEY"] if server_ok else None),
    }
    providers = available_providers(keys)
    if providers:
        provider = st.selectbox("Vision provider", providers,
                                format_func=lambda p: {"gemini": "Google Gemini (recommended)", "openai": "OpenAI"}[p])
        uses_own_key = bool(user_gemini if provider == "gemini" else user_openai)
        st.caption(f"AI calls this session: {ss.ai_calls}" + ("" if uses_own_key else f" / {AI_CALL_LIMIT}"))
    else:
        provider = None
        uses_own_key = False
        st.info("No AI key available. You can still use the sample house, or mark components by hand.")

    st.divider()
    st.caption("Estimates are advisory, not a quotation. Rates are indicative and editable.")


def can_call_ai() -> bool:
    return provider is not None and (uses_own_key or ss.ai_calls < AI_CALL_LIMIT)


# --------------------------------------------------------------------------- tabs
st.title("Exterior Renovation Planner & Cost Estimator")
st.caption("See your own house in new finishes, then get quantities and a transparent cost breakdown before work starts.")
tab_upload, tab_review, tab_design, tab_estimate, tab_report = st.tabs(
    ["1 · Upload", "2 · Review components", "3 · Design & compare", "4 · Measure & cost", "5 · Report & save"]
)

# ---- 1. Upload -------------------------------------------------------------
with tab_upload:
    c1, c2 = st.columns([3, 2])
    with c1:
        files = st.file_uploader(
            "Upload one or more photos of the house exterior (JPG/PNG/WebP)",
            type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True,
        )
    with c2:
        st.markdown("**No photo handy?**")
        if st.button("Load the sample house", width="stretch"):
            ss.image = sample_image()
            regs = sample_regions()
            set_regions(regs, "sample")
            reset_designs_with_default(regs)
            ss.set_floors = 2
            ss.detect_info = None
            st.rerun()
        st.caption("The sample comes with its components already marked, so every step works without an API key.")

    if files:
        reports = []
        for f in files:
            data = f.getvalue()
            try:
                img = load_image(data)
                rep = assess(img, original_size(data))
            except Exception as exc:
                st.error(f"{f.name}: could not read image ({exc}).")
                continue
            reports.append((f, img, rep))
        if reports:
            reports.sort(key=lambda t: t[2].score, reverse=True)
            usable = [t for t in reports if t[2].usable]
            st.markdown("#### Photo check")
            cols = st.columns(min(len(reports), 4))
            for i, (f, img, rep) in enumerate(reports):
                with cols[i % len(cols)]:
                    st.image(img, caption=f.name, width="stretch")
                    if rep.usable:
                        st.success(f"Usable · quality {rep.score:.0f}/100")
                    else:
                        st.error("Not usable")
                    for issue in rep.issues:
                        st.write("• " + issue)
                    if not rep.usable:
                        for tip in rep.tips:
                            st.caption("Tip: " + tip)
            if usable:
                names = [t[0].name for t in usable]
                choice = st.selectbox("Working view (clearest photo pre-selected)", names)
                f, img, rep = next(t for t in usable if t[0].name == choice)
                sig = hashlib.md5(img.tobytes()).hexdigest()
                if st.button("Use this photo", type="primary"):
                    ss.image = img
                    set_regions([], f"upload:{choice}:{sig}")
                    ss.designs = {"Design A": {}}
                    ss.active_design = "Design A"
                    ss.design_rev += 1
                    ss.photoreal = {}
                    ss.detect_info = None
                    st.success("Photo selected. Go to **2 · Review components**.")
            else:
                st.warning("None of the photos are usable yet. Follow the tips above and try again.")

    if ss.image is not None:
        st.markdown("#### Current working photo")
        st.image(ss.image, width="stretch")

if ss.image is None:
    for t in (tab_review, tab_design, tab_estimate, tab_report):
        with t:
            st.info("Start in **1 · Upload**: add a photo or load the sample house.")
    st.stop()

image: np.ndarray = ss.image
H, W = image.shape[:2]

# ---- 2. Review components --------------------------------------------------
with tab_review:
    st.markdown("The system maps the facade into components. Check every box and correct it if needed: "
                "areas and costs are only as good as this map.")
    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("🔍 Detect components with AI", type="primary", disabled=not can_call_ai(), width="stretch"):
            with st.spinner("Asking the vision model to map the facade..."):
                try:
                    regs, info = detect_components(image, provider, keys[f"{provider.upper()}_API_KEY"])
                    ss.ai_calls += 1
                    set_regions(regs, f"ai:{provider}")
                    reset_designs_with_default(regs)
                    ss.detect_info = info
                    st.rerun()
                except DetectionError as exc:
                    ss.ai_calls += 1
                    st.error(str(exc))
        if provider is None:
            st.caption("Needs a Gemini or OpenAI key (sidebar).")
        elif not can_call_ai():
            st.caption("Session AI limit reached. Add your own key in the sidebar.")
    with b2:
        if st.button("Start from a blank map", width="stretch"):
            set_regions([], ss.image_source or "manual")
            st.rerun()
    with b3:
        show_grid = st.toggle("Show 10% grid", value=True, help="Helps you read box coordinates off the photo.")

    if ss.detect_info:
        i = ss.detect_info
        st.caption(f"Detected {i['count']} components with {i['provider']} / {i['model']} in {i['seconds']}s.")

    st.markdown("**Components** (coordinates are % of image width/height; add rows with ➕, delete with the row checkbox)")
    base_df = pd.DataFrame([
        {"id": r.id, "kind": r.kind, "label": r.label,
         "left %": round(r.x0 * 100, 1), "top %": round(r.y0 * 100, 1),
         "right %": round(r.x1 * 100, 1), "bottom %": round(r.y1 * 100, 1),
         "source": r.source}
        for r in ss.regions_base
    ], columns=["id", "kind", "label", "left %", "top %", "right %", "bottom %", "source"])
    edited = st.data_editor(
        base_df,
        key=f"regions_editor_{ss.regions_version}",
        num_rows="dynamic",
        hide_index=True,
        column_config={
            "id": st.column_config.TextColumn("id", disabled=True, width="small"),
            "kind": st.column_config.SelectboxColumn("Component", options=ALL_KINDS, required=True),
            "label": st.column_config.TextColumn("Label"),
            "left %": st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=0.5, required=True),
            "top %": st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=0.5, required=True),
            "right %": st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=0.5, required=True),
            "bottom %": st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=0.5, required=True),
            "source": st.column_config.TextColumn("Source", disabled=True, width="small"),
        },
    )
    regions: list[Region] = []
    skipped = 0
    for idx, row in edited.iterrows():
        try:
            if pd.isna(row["kind"]) or any(pd.isna(row[c]) for c in ("left %", "top %", "right %", "bottom %")):
                skipped += 1
                continue
            rid = row["id"] if isinstance(row["id"], str) and row["id"] else None
            if rid is None:
                # Stable id for a newly added row, derived from its position in the editor.
                rid = "m" + hashlib.md5(f"{ss.regions_version}:{idx}".encode()).hexdigest()[:7]
            r = Region(kind=row["kind"], label=row["label"] if isinstance(row["label"], str) else "",
                       x0=row["left %"] / 100, y0=row["top %"] / 100, x1=row["right %"] / 100, y1=row["bottom %"] / 100,
                       source=row["source"] if isinstance(row["source"], str) else "manual", id=rid)
            if r.is_valid:
                regions.append(r)
            else:
                skipped += 1
        except Exception:
            skipped += 1
    ss.regions = regions
    if skipped:
        st.caption(f"{skipped} incomplete row(s) ignored until all fields are filled in.")

    masks = cached_masks(image, regions_key(regions), bool(ss.set_refine))
    overlay = draw_overlay(image, regions, masks)
    st.image(add_grid(overlay) if show_grid else overlay, width="stretch",
             caption="Shaded areas are what will be measured and finished. Windows and doors are cut out of walls.")
    counts = pd.Series([r.kind for r in regions]).value_counts() if regions else pd.Series(dtype=int)
    if regions:
        st.caption(" · ".join(f"{KIND_LABELS[k]}: {n}" for k, n in counts.items()))
    if not any(r.kind == "door" for r in regions):
        st.warning("No door marked. A door is the best size reference; without it, set the number of floors or a "
                   "measurement in **4 · Measure & cost**.")

# ---- 3. Design inputs ------------------------------------------------------
surface_regions = [r for r in regions if not r.is_opening]
with tab_design:
    top = st.columns([2, 2, 3])
    names = list(ss.designs.keys())
    if ss.active_design not in names:
        ss.active_design = names[0]
    with top[0]:
        active = st.selectbox("Design option", names, index=names.index(ss.active_design))
        if active != ss.active_design:
            ss.active_design = active
            st.rerun()
    with top[1]:
        new_name = st.text_input("New design name", value=f"Design {chr(65 + len(names))}")
        if st.button("➕ Copy current as new design", width="stretch"):
            nm = new_name.strip() or f"Design {uuid.uuid4().hex[:4]}"
            if nm in ss.designs:
                st.warning("A design with that name already exists.")
            else:
                ss.designs[nm] = {k: dict(v) for k, v in ss.designs[ss.active_design].items()}
                ss.active_design = nm
                ss.design_rev += 1
                st.rerun()
    with top[2]:
        preset = st.selectbox("Style preset", list(PRESETS.keys()))
        pc1, pc2 = st.columns(2)
        if pc1.button("Apply preset to this design", width="stretch"):
            ss.designs[ss.active_design] = apply_preset(preset, regions, CATALOG)
            ss.design_rev += 1
            ss.photoreal.pop(ss.active_design, None)
            st.rerun()
        if pc2.button("🗑 Delete this design", width="stretch", disabled=len(names) < 2):
            ss.designs.pop(ss.active_design)
            ss.photoreal.pop(ss.active_design, None)
            ss.active_design = next(iter(ss.designs))
            ss.design_rev += 1
            st.rerun()

    design = ss.designs.setdefault(ss.active_design, {})
    controls_col, preview_col = st.columns([2, 3])
    with controls_col:
        st.markdown(f"**Materials for {ss.active_design}**")
        if not surface_regions:
            st.info("Mark at least one wall, pillar, railing or other surface in step 2.")
        for r in surface_regions:
            options = ["(keep as is)"] + [m["id"] for m in materials_for(r.kind, CATALOG)]
            kmat = f"mat_{ss.design_rev}_{ss.active_design}_{r.id}"
            cur = design.get(r.id) or {}
            if kmat not in ss:
                ss[kmat] = cur.get("material") if cur.get("material") in options else "(keep as is)"
            c1, c2 = st.columns([5, 1])
            with c1:
                mid = st.selectbox(
                    r.label, options, key=kmat,
                    format_func=lambda m: m if m == "(keep as is)" else f"{CATALOG[m]['name']} · {CATALOG[m]['category']}",
                )
            if mid == "(keep as is)":
                design.pop(r.id, None)
                continue
            # One colour widget per material, so switching material starts from that material's own default.
            kcol = f"col_{ss.design_rev}_{ss.active_design}_{r.id}_{mid}"
            if kcol not in ss:
                ss[kcol] = cur.get("color") if cur.get("material") == mid and cur.get("color") else CATALOG[mid]["default_color"]
            with c2:
                col = st.color_picker("Colour", key=kcol, label_visibility="hidden")
            design[r.id] = {"material": mid, "color": col}
        with st.expander("About these materials (durability & maintenance)"):
            used = sorted({v["material"] for v in design.values() if v.get("material") in CATALOG})
            for mid in used or list(CATALOG):
                m = CATALOG[mid]
                st.markdown(f"**{m['name']}** ({m['category']}) — best for: {m['best_for']}. "
                            f"Durability: {m['durability']}. Maintenance: {m['maintenance']}.")
        st.toggle("Refine wall edges (GrabCut)", key="set_refine",
                  help="Tightens wall boxes to the actual wall using colour similarity. Useful when a box includes sky or trees.")

# ---- 4. Estimate inputs ----------------------------------------------------
with tab_estimate:
    st.markdown("#### Scale calibration")
    st.caption("Areas come from pixels, so the app needs to know how many pixels make a foot. "
               "It uses every reference it can find; one real measurement beats all assumptions.")
    s1, s2, s3, s4 = st.columns(4)
    s1.number_input("Number of floors visible", min_value=0, max_value=6, step=1, key="set_floors",
                    help="0 = don't use. Assumes ~10 ft floor-to-floor.")
    s2.number_input("A width you measured (ft)", min_value=0.0, max_value=500.0, step=1.0, key="set_known_width",
                    help="e.g. the front wall is 32 ft wide. 0 = not measured.")
    region_opts = ["(largest wall)"] + [f"{r.id} | {r.label}" for r in surface_regions]
    if ss.set_known_region not in region_opts:
        ss.set_known_region = "(largest wall)"
    s3.selectbox("...which component is that width?", region_opts, key="set_known_region")
    s4.selectbox("Scale method", ["auto", "user_measurement", "door_reference", "floor_count", "window_reference"],
                 key="set_scale_method", format_func=lambda m: "Auto (best available)" if m == "auto" else METHOD_LABELS[m])

    with st.expander("Advanced: facades, surface factors, taxes"):
        a1, a2, a3, a4 = st.columns(4)
        a1.number_input("Facade multiplier", min_value=0.5, max_value=4.0, step=0.5, key="set_facade_mult",
                        help="1 = only the photographed front. Use ~2.5-3 to approximate all four sides of a similar house.")
        a2.number_input("Railing height (ft)", min_value=2.0, max_value=6.0, step=0.5, key="set_railing_h")
        a3.number_input("Contingency %", min_value=0.0, max_value=30.0, step=1.0, key="set_contingency")
        a4.number_input("GST %", min_value=0.0, max_value=28.0, step=1.0, key="set_gst")
        st.caption("Surface factor = finished area ÷ area visible in the photo.")
        fcols = st.columns(len(DEFAULT_SURFACE_FACTORS))
        for col, kind in zip(fcols, DEFAULT_SURFACE_FACTORS):
            col.number_input(KIND_LABELS[kind], min_value=0.0, max_value=6.0, step=0.5, key=f"sf_{kind}")

    st.markdown("#### Material rates (edit to match your local quotes)")
    rate_rows = []
    for mid, m in CATALOG.items():
        o = ss.rate_overrides.get(mid, {})
        rate_rows.append({
            "id": mid, "material": m["name"], "per": m["purchase_unit"],
            "material_rate": float(o.get("material_rate", m["material_rate"])),
            "labour_rate": float(o.get("labour_rate", m["labour_rate"])),
            "labour per": "running ft" if m["labour_basis"] == "running_ft" else "sqft",
            "wastage_pct": float(o.get("wastage_pct", m["wastage_pct"])),
            "coverage_sqft_per_unit": float(o.get("coverage_sqft_per_unit", m.get("coverage_sqft_per_unit") or 0)),
            "coats": float(o.get("coats", m.get("coats", 1))),
        })
    used_ids = {v["material"] for v in design.values()}
    only_used = st.toggle("Show only materials in this design", value=True)
    rate_df = pd.DataFrame(rate_rows)
    if only_used and used_ids:
        rate_df = rate_df[rate_df["id"].isin(used_ids)].reset_index(drop=True)
    edited_rates = st.data_editor(
        rate_df,
        key=f"rates_editor_{ss.rates_version}_{only_used}_{'-'.join(sorted(used_ids)) if only_used else 'all'}",
        hide_index=True, width="stretch", num_rows="fixed",
        disabled=["id", "material", "per", "labour per"],
        column_config={
            "id": None,
            "material_rate": st.column_config.NumberColumn("Material rate (₹ per unit)", min_value=0.0, format="%.0f"),
            "labour_rate": st.column_config.NumberColumn("Labour rate (₹)", min_value=0.0, format="%.0f"),
            "wastage_pct": st.column_config.NumberColumn("Wastage %", min_value=0.0, max_value=50.0),
            "coverage_sqft_per_unit": st.column_config.NumberColumn("Coverage (sqft/unit/coat)", min_value=0.0,
                                                                    help="Paint and texture only; 0 for others."),
            "coats": st.column_config.NumberColumn("Coats", min_value=1.0, max_value=4.0, step=1.0),
        },
    )
    for _, row in edited_rates.iterrows():
        mid = row["id"]
        base = CATALOG[mid]
        diff = {}
        for f in EDITABLE_FIELDS:
            if f not in row or pd.isna(row[f]):
                continue
            base_val = float(base.get(f) or 0)
            if f == "coverage_sqft_per_unit" and not base.get(f):
                continue
            if float(row[f]) != base_val:
                diff[f] = float(row[f])
        if diff:
            ss.rate_overrides[mid] = diff
        else:
            ss.rate_overrides.pop(mid, None)
    if st.button("Reset all rates to defaults"):
        ss.rate_overrides = {}
        ss.rates_version += 1
        st.rerun()
    st.caption(rates_note())

# --------------------------------------------------------------------------- compute
settings = settings_from_state()
masks = cached_masks(image, regions_key(regions), settings["refine_edges"])
result = pipeline.run(image, regions, design, settings, ss.rate_overrides, masks=masks)

# ---- 3. Design outputs -----------------------------------------------------
with tab_design:
    with preview_col:
        mode = st.radio("View", ["Side by side", "Before / after slider", "All designs"], horizontal=True,
                        label_visibility="collapsed")
        if mode == "Side by side":
            c1, c2 = st.columns(2)
            c1.image(image, caption="Original", width="stretch")
            c2.image(result.rendered, caption=f"{ss.active_design} (geometry-preserving render)", width="stretch")
        elif mode == "Before / after slider":
            pct = st.slider("Reveal redesign", 0, 100, 50, format="%d%%")
            cut = int(W * pct / 100)
            combo = image.copy()
            combo[:, :cut] = result.rendered[:, :cut]
            if 0 < cut < W:
                combo[:, max(cut - 1, 0):cut + 1] = 255
            st.image(combo, caption="Left of the line: redesigned · right: original", width="stretch")
        else:
            cols = st.columns(2)
            for i, (nm, d) in enumerate(ss.designs.items()):
                r_i = result if nm == ss.active_design else pipeline.run(image, regions, d, settings, ss.rate_overrides, masks=masks)
                with cols[i % 2]:
                    st.image(r_i.rendered, width="stretch",
                             caption=f"{nm} · est. {inr(r_i.estimate.grand_total)}")
        st.metric("Estimated total for this design", inr(result.estimate.grand_total),
                  help="Materials + labour + contingency + GST. Details in 4 · Measure & cost.")

        with st.expander("✨ Photoreal impression (optional, uses Gemini image model)"):
            st.caption("Sends the original photo and this render to an image-editing model for realistic lighting. "
                       "Generative models can shift details, so the report and measurements always use the "
                       "geometry-preserving render above.")
            gem_key = keys.get("GEMINI_API_KEY")
            can_photo = bool(gem_key) and (bool(user_gemini) or ss.ai_calls < AI_CALL_LIMIT)
            if st.button("Generate photoreal impression", disabled=not can_photo):
                lines = []
                for r in surface_regions:
                    ch = design.get(r.id)
                    if ch:
                        lines.append(f"{r.label}: {CATALOG[ch['material']]['name']} in colour {ch.get('color')}")
                with st.spinner("Rendering (can take 10-30 s)..."):
                    try:
                        ss.photoreal[ss.active_design] = photoreal_render(image, result.rendered, lines, gem_key)
                        ss.ai_calls += 1
                    except DetectionError as exc:
                        ss.ai_calls += 1
                        st.error(str(exc))
            if not gem_key:
                st.caption("Needs a Gemini key.")
            if ss.active_design in ss.photoreal:
                st.image(ss.photoreal[ss.active_design], caption="Artist's impression (AI-generated)", width="stretch")

# ---- 4. Estimate outputs ---------------------------------------------------
with tab_estimate:
    sc = result.scale
    st.markdown("#### Result")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Grand total", inr(result.estimate.grand_total))
    m2.metric("Materials", inr(result.estimate.material_total))
    m3.metric("Labour", inr(result.estimate.labour_total))
    m4.metric("Scale used", f"{1 / sc.ft_per_px:.1f} px/ft", METHOD_LABELS.get(sc.method, sc.method), delta_color="off")
    if sc.warning:
        st.warning(sc.warning)
    with st.expander("Scale references found", expanded=sc.method == "fallback"):
        if sc.candidates:
            st.dataframe(pd.DataFrame([{"method": METHOD_LABELS[c.method], "px per ft": round(1 / c.ft_per_px, 2),
                                        "basis": c.detail, "used": "✅" if c.method == sc.method else ""}
                                       for c in sc.candidates]), hide_index=True, width="stretch")
        else:
            st.write("None. Mark a door, set the number of floors, or enter one measurement.")

    st.markdown("#### Surface areas")
    area_df = pd.DataFrame([{
        "Component": a.label, "Type": KIND_LABELS[a.kind], "Visible (sqft)": a.visible_sqft,
        "Factor": a.surface_factor, "To finish (sqft)": a.finish_sqft,
        "Running ft": a.running_ft if a.running_ft is not None else None,
    } for a in result.areas])
    st.dataframe(area_df, hide_index=True, width="stretch")
    st.caption(f"Window and door openings deducted from walls: {result.openings_sqft:,.1f} sqft visible area.")

    st.markdown("#### Quantities & cost")
    if result.estimate.lines:
        line_df = pd.DataFrame([{
            "Component": li.component, "Material": li.material, "Quantity": li.quantity, "Unit": li.purchase_unit,
            "How calculated": li.quantity_note, "Material ₹": round(li.material_cost), "Labour ₹": round(li.labour_cost),
            "Total ₹": round(li.total),
        } for li in result.estimate.lines])
        st.dataframe(line_df, hide_index=True, width="stretch")
        st.markdown("**Purchase list** — quantities rounded up once per material to whole litres, boxes or sheets")
        st.dataframe(pd.DataFrame([{
            "Material": p.material, "Needed": f"{p.exact_qty:g} {p.unit}", "Buy": f"{p.purchase_qty:g} {p.unit}",
            "Rate ₹": round(p.rate), "Cost ₹": round(p.cost),
        } for p in result.estimate.purchases]), hide_index=True, width="stretch")
        st.caption(f"Rounding to full packs adds {inr(result.estimate.rounding_adjustment)} to the component costs above.")
        cat_df = pd.DataFrame([{"Category": k, "Material ₹": round(v["material"]), "Labour ₹": round(v["labour"]),
                                "Total ₹": round(v["total"])} for k, v in result.estimate.by_category.items()])
        c1, c2 = st.columns([3, 2])
        c1.dataframe(cat_df, hide_index=True, width="stretch")
        e = result.estimate
        c2.markdown(
            f"Subtotal: **{inr(e.subtotal)}**  \n"
            f"Contingency {e.contingency_pct:g}%: {inr(e.contingency)}  \n"
            f"GST {e.gst_pct:g}%: {inr(e.gst)}  \n"
            f"### Grand total: {inr(e.grand_total)}"
        )
    else:
        st.info("Choose materials in **3 · Design & compare** to see quantities and costs.")
    if result.estimate.unassigned:
        st.caption("Not costed (kept as is): " + ", ".join(result.estimate.unassigned))

# ---- 5. Report & save ------------------------------------------------------
with tab_report:
    st.markdown("Download a report to discuss with contractors, or save the project to continue editing later.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**PDF report** (original, redesign, materials, quantities, cost breakdown, method)")
        if st.button("Prepare PDF report", type="primary", disabled=not result.estimate.lines):
            with st.spinner("Building report..."):
                ss.report_pdf = build_report(
                    ss.project_name, ss.active_design, image, result.rendered, result.areas, result.estimate,
                    design, result.catalog, result.scale, result.openings_sqft, settings, rates_note(),
                )
                ss.report_for = (ss.active_design, result.estimate.grand_total)
        if ss.get("report_pdf") and ss.get("report_for") == (ss.active_design, result.estimate.grand_total):
            st.download_button("⬇ Download PDF", ss.report_pdf,
                               file_name=f"{ss.project_name.strip().replace(' ', '_') or 'house'}_{ss.active_design.replace(' ', '_')}_estimate.pdf",
                               mime="application/pdf")
        elif ss.get("report_pdf"):
            st.caption("The design changed since the last report. Prepare it again.")
    with c2:
        st.markdown("**Project file** (photo, components, all designs, rates, settings)")
        proj_bytes = dump_project(ss.project_name, image, regions, ss.designs, ss.active_design,
                                  ss.rate_overrides, settings)
        st.download_button("⬇ Save project (.json)", proj_bytes,
                           file_name=f"{ss.project_name.strip().replace(' ', '_') or 'house'}_project.json",
                           mime="application/json")
        st.caption("Re-open it from the sidebar at any time to keep editing.")
