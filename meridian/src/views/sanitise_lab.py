"""Sanitisation Lab: upload -> policy -> verified result -> (optional) price in the Document Gate.

Layout: left rail = policy; right = pipeline result. Sensitive text is never displayed: regions show category, rule,
location, action and verification only (the engine stores keyed hashes and shapes, not values)."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from ..sanitize import (APPROVED, BLOCKED, FAILED, REJECTED, REVIEW, Options, Policy, build_registry, builtin_rules, capability_matrix, process_document)
from ..sanitize.bridge import document_from_result
from ..sanitize.imaging import face_backend_status, ocr_available
from ..ui import theme as T

TYPES = ["pdf", "docx", "pptx", "png", "jpg", "jpeg", "tif", "tiff", "bmp"]
STATUS_TONE = {APPROVED: "ok", REVIEW: "high", BLOCKED: "crit", REJECTED: "crit", FAILED: "crit"}
STATUS_TEXT = {APPROVED: "Every finding was removed and re-checked in the output file.",
               REVIEW: "Held for a person: something could not be inspected or fully verified.",
               BLOCKED: "Not released: a removal failed verification. The file is quarantined.",
               REJECTED: "Rejected before processing (type, size or structure not acceptable).",
               FAILED: "Processing failed. Nothing was released."}


def _policy_rail() -> tuple[Policy, list[str]]:
    reg = build_registry(Policy())
    rules = [r for r in builtin_rules()]
    st.markdown("##### Policy")
    ocr_ok = ocr_available()
    ocr = st.toggle("Read text inside images (OCR)", value=ocr_ok, disabled=not ocr_ok, help="Needs the Tesseract program on this machine." if not ocr_ok else None)
    if not ocr_ok:
        T.callout("Tesseract is not installed, so image-borne text cannot be inspected. Files with images will be held for review.", "warn")
    fb = face_backend_status()
    faces = st.toggle("Blank face regions (local detector, no identification)", value=False, disabled=not fb["available"], help=fb["detail"])
    optional = [r for r in rules if not r.hard_block and r.id != "FACE.REGION"]
    off = st.multiselect("Rules to switch off", [r.id for r in optional], format_func=lambda i: f"{i} - {next(r.label for r in optional if r.id == i)}",
                         help="Mandatory rules (government IDs, card numbers, secrets) cannot be switched off.")
    with st.expander("Allow-list, deny-list, custom pattern"):
        allow = st.text_area("Never treat as sensitive (one per line)", height=70, help="Cannot override mandatory rules.")
        deny = st.text_area("Always remove (one per line)", height=70)
        cid = st.text_input("Custom rule name", "")
        cpat = st.text_input("Custom regular expression", "", help="Screened for catastrophic backtracking before use.")
    custom = [{"id": cid or "RULE", "pattern": cpat, "label": cid or "Custom"}] if cpat.strip() else []
    pol = Policy(ocr=ocr, faces=faces, rule_toggles={i: False for i in off}, custom_rules=custom,
                 allowlist=[x.strip() for x in allow.splitlines() if x.strip()], denylist=[x.strip() for x in deny.splitlines() if x.strip()])
    warns = list(build_registry(pol).warnings)
    return pol, warns


def _result_view(res: dict, out_bytes: bytes | None, out_name: str, vendors, get_state, POL, evaluate_document) -> None:
    status = res["release"]["status"]
    T.verdict(status.replace("_", " "), STATUS_TEXT.get(status, ""), STATUS_TONE.get(status, "high"))
    sa, ve, cov = res["sanitisation"], res["verification"], res["coverage"]
    T.kpi_row([("Findings", res["detected"]["regions"], f"{res['detected']['hard_block_regions']} mandatory", "accent"),
               ("Removed", f"{sa['applied']}/{sa['eligible']}", f"{sa['applied_ratio']:.0%} applied", "ok" if sa["applied_ratio"] == 1 else "high"),
               ("Verified", f"{ve['verified_ratio']:.0%}", f"checks: {ve['status']}", "ok" if ve["status"] == "passed" else "crit"),
               ("Inspected", f"{cov['payload_ratio']:.0%}", f"{cov['units_inspected']}/{cov['units_total']} text units", "accent")])
    for r in res["release"]["reasons"]:
        T.callout(r, STATUS_TONE.get(status, "info") if status != APPROVED else "ok")
    if status == APPROVED and out_bytes:
        st.download_button("Download sanitised file", out_bytes, out_name, type="primary")
    elif status != APPROVED:
        T.callout("No file is offered for download: it was not approved for release.", "warn")
    tabs = st.tabs(["Findings", "Capabilities", "Verification", "Price in Gate", "Details"])
    with tabs[0]:
        regs = res["regions"]
        if not regs:
            T.empty_state("Nothing sensitive found", "No rule matched. That is not a guarantee: only the listed payloads were inspected.")
        else:
            st.dataframe(pd.DataFrame([{"Category": r["category"], "Rule": r["rule_id"], "Where": r["unit_id"], "Method": r["source_method"], "Action": r["action"],
                                        "Confidence": round(r["confidence"], 2), "Verified": r["verification_status"], "Shape": r["text_shape"], "Mandatory": r["hard_block"]} for r in regs]),
                         hide_index=True, width="stretch")
            st.caption("Values are not displayed or stored. Shape shows only the character pattern.")
    with tabs[1]:
        st.dataframe(pd.DataFrame([{"Payload": p["payload"], "Outcome": p["outcome"], "Count": p["count"], "Can do at most": p["max_capability"], "Note": p["detail"]} for p in res["payloads"]]),
                     hide_index=True, width="stretch")
        if res["unsupported"]:
            T.callout("Unsupported content was reported, not silently dropped: " + ", ".join(str(u) if not isinstance(u, dict) else u.get("payload", "?") for u in res["unsupported"]), "warn")
    with tabs[2]:
        st.dataframe(pd.DataFrame([{"Check": c["name"], "Status": c["status"], "Detail": c["detail"]} for c in ve["checks"]]), hide_index=True, width="stretch")
    with tabs[3]:
        doc_b = document_from_result(res)
        doc_a = document_from_result(res, after_sanitisation=True)
        v = st.selectbox("Destination vendor", list(vendors), format_func=lambda x: f"{x} - {vendors[x]['vendor_name']}", key="san_v")
        tasks = list(POL["tasks"])
        task = st.selectbox("Task", tasks, format_func=lambda t: POL["tasks"][t]["label"], key="san_t")
        s = get_state(vendors[v])
        c1, c2 = st.columns(2)
        for col, name, d in ((c1, "As found", doc_b), (c2, "After sanitisation (verified)", doc_a)):
            ev = evaluate_document(s, "sanitised-file", d, task)
            with col:
                st.markdown(f"**{name}**")
                T.render(T.kv([("Decision", T.raw(T.badge(ev["decision"], ev["decision"]))), ("Cost", f"{ev['cost']:.1f}"), ("Budget after", f"{ev['balance_after']:,.0f}")]))
        st.caption("Pricing only. Nothing is written to the ledger from this tab.")
    with tabs[4]:
        st.json({k: res[k] for k in ("run_id", "engine_version", "rule_registry_version", "policy", "input", "timings", "cleanup", "warnings", "errors")}, expanded=False)


def render(vendors, get_state, POL, evaluate_document) -> None:
    T.page_header("Sanitisation Lab", "Remove sensitive content from PDF, Word, PowerPoint and image files, then prove it is gone. "
                  "A file is released only after the output has been re-opened and independently checked.", "Document sanitisation")
    left, right = st.columns([1, 2.2], gap="large")
    with left:
        pol, warns = _policy_rail()
        for w in warns:
            T.callout(w, "warn")
        up = st.file_uploader("File", type=TYPES, help="Content is validated by its bytes, not its extension.")
        go = st.button("Sanitise", type="primary", disabled=up is None)
    with right:
        if go and up is not None:
            with tempfile.TemporaryDirectory() as d:
                ext = Path(up.name).suffix.lower()
                src, out = Path(d) / f"input{ext}", Path(d) / f"output{ext}"
                src.write_bytes(up.getvalue())
                with st.spinner("Detecting, removing, re-opening and verifying…"):
                    res = process_document(src, out, pol, Options(persist_regions=False, doc_id=None))
                data = out.read_bytes() if out.exists() and res["release"]["released"] else None
            res["input"]["name"] = Path(up.name).name
            st.session_state["san_last"] = (res, data, f"sanitised_{Path(up.name).name}")
        last = st.session_state.get("san_last")
        if not last:
            T.empty_state("Drop a file to begin", "Supported: PDF (native, scanned, mixed), DOCX, PPTX, PNG, JPEG, TIFF, BMP. Everything runs on this machine.")
            with st.expander("What can be removed from each format"):
                st.dataframe(pd.DataFrame(capability_matrix()), hide_index=True, width="stretch")
        else:
            _result_view(*last, vendors, get_state, POL, evaluate_document)
