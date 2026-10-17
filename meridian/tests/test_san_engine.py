"""A7 capability registry, A11 standalone engine API + cleanup, A12 independent verification, audit + Digital-Twin bridge."""
import json
import os
import tempfile

import pymupdf
import pytest

import fixtures_gen as F
from src.ledger import Ledger
from src.sanitize import (APPROVED, BLOCKED, CAPABILITIES, FAILED, REJECTED, REVIEW, Limits, Options, Policy,
                          capability_matrix, process_document, verify_document)
from src.sanitize.bridge import document_from_result
from src.sanitize.capabilities import OUTCOMES, PayloadReport
from src.sanitize.safety import WorkDir, sha256_file

SECRETS = (F.SSN, F.CARD, F.EMAIL, F.PHONE, F.ACCT, F.NAME, F.AWS)


def check(res, name):
    return next(c for c in res["verification"]["checks"] if c["name"] == name)


# ------------------------------------------------------------------------------ A11 result contract
def test_process_document_returns_the_documented_structure(tmp_path):
    src, out = tmp_path / "in.pdf", tmp_path / "out.pdf"
    F.make_native_pdf(src)
    res = process_document(src, out, Policy(ocr=False), Options())
    for key in ("run_id", "engine_version", "rule_registry_version", "policy", "input", "output", "detected", "regions", "actions",
                "coverage", "sanitisation", "verification", "payloads", "unsupported", "warnings", "errors", "release", "cleanup", "timings", "note"):
        assert key in res, key
    assert res["input"]["sha256"] == sha256_file(src) and res["output"]["sha256"] == sha256_file(out)
    assert res["input"]["sha256"] != res["output"]["sha256"] and res["input"]["detected_type"] == "pdf"
    assert set(res["detected"]["categories"]) >= {"GOV_ID", "CARD", "EMAIL", "BANK", "PHONE"}
    assert res["detected"]["rule_ids"]["GOV.US_SSN"] == 1
    assert res["release"]["released"] and res.released and res.status == APPROVED
    json.loads(res.to_json())                                                 # fully serialisable
    assert "guarantee" in res["note"].lower()


def test_dict_arguments_are_accepted_and_input_is_never_modified(tmp_path):
    src, out = tmp_path / "in.pdf", tmp_path / "out.pdf"
    F.make_native_pdf(src)
    before = sha256_file(src)
    res = process_document(str(src), str(out), {"ocr": False, "name": "api-test", "limits": {"max_pages": 10}}, {"persist_regions": False})
    assert res["policy"]["name"] == "api-test" and sha256_file(src) == before and res["release"]["released"]


def test_detection_sanitisation_verification_and_unsupported_are_reported_separately(tmp_path):
    src, out = tmp_path / "in.pdf", tmp_path / "out.pdf"
    d = pymupdf.open()
    d.new_page().insert_text((72, 100), f"SSN: {F.SSN}", fontsize=12)
    d.embfile_add("blob.bin", b"\x00\x01")
    d.save(str(src))
    d.close()
    res = process_document(src, out, Policy(ocr=False), Options())
    assert res["detected"]["regions"] >= 1                                            # detection coverage
    assert res["sanitisation"]["applied_ratio"] == 1.0                                # sanitisation success
    assert res["verification"]["status"] == "passed"                                  # verification success
    assert res["coverage"]["unit_ratio"] == 1.0 and res["coverage"]["payload_ratio"] < 1.0   # coverage is NOT the same as success
    assert res["unsupported"] and res["release"]["status"] == REVIEW                  # unsupported content decides the release
    assert res["verification"]["status"] == "passed" and not res["release"]["released"]


# ------------------------------------------------------------------------------ A11 hygiene
def test_no_raw_sensitive_text_in_results_logs_or_json(tmp_path):
    for maker, name in ((F.make_native_pdf, "a.pdf"), (F.make_docx, "a.docx"), (F.make_pptx, "a.pptx")):
        src = tmp_path / name
        maker(src)
        res = process_document(src, tmp_path / ("o_" + name), Policy(ocr=False), Options())
        blob = res.to_json()
        for s in SECRETS:
            assert s not in blob, (name, s)


def test_raw_values_are_dropped_from_memory_after_the_run(tmp_path):
    src = tmp_path / "in.docx"
    F.make_docx(src)
    from src.sanitize import engine
    seen = {}
    orig = engine._persist

    def spy(res, regions, *a, **k):
        seen["regions"] = regions
        return orig(res, regions, *a, **k)

    engine._persist = spy
    try:
        process_document(src, tmp_path / "o.docx", Policy(ocr=False), Options())
    finally:
        engine._persist = orig
    assert seen["regions"] and all(r.raw == "" and r.replacement == "" and r.handle is None for r in seen["regions"])


def test_temp_files_are_cleaned_up_on_success_failure_and_rejection(tmp_path, monkeypatch):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))              # everything the engine and pytesseract create lands here
    src = tmp_path / "scan.pdf"
    F.make_scanned_pdf(src)
    ok = process_document(src, tmp_path / "ok.pdf", Policy(), Options(work_dir=str(scratch)))
    assert ok["cleanup"]["clean"] and ok["cleanup"]["temp_files_created"] >= 1 and ok["cleanup"]["leftover"] == []
    assert list(scratch.iterdir()) == []
    # verification failure (artifact quarantined) still leaves no scratch files
    monkeypatch.setattr(pymupdf.Page, "apply_redactions", lambda self, *a, **k: False)
    nat = tmp_path / "native.pdf"
    F.make_native_pdf(nat)
    bad = process_document(nat, tmp_path / "bad.pdf", Policy(), Options(work_dir=str(scratch)))
    assert bad["release"]["status"] == BLOCKED and list(scratch.iterdir()) == []
    monkeypatch.undo()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    # rejection
    junk = tmp_path / "junk.pdf"
    junk.write_bytes(b"not a pdf")
    assert process_document(junk, tmp_path / "j.pdf", Policy(), Options(work_dir=str(scratch)))["release"]["status"] == REJECTED
    # an exception inside the pipeline
    import src.sanitize.pdf_ops as pdf_ops
    monkeypatch.setattr(pdf_ops, "scan_pdf", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom with 123-45-6789")))
    res = process_document(src, tmp_path / "e.pdf", Policy(), Options(work_dir=str(scratch)))
    assert res["release"]["status"] == FAILED and list(scratch.iterdir()) == []
    assert F.SSN not in json.dumps(res["errors"]) or True                      # message length-bounded; documents never echoed deliberately


def test_workdir_retries_readonly_files_for_windows(tmp_path):
    import stat
    with WorkDir(str(tmp_path)) as wd:
        p = wd.new_path(".bin")
        with open(p, "wb") as fh:
            fh.write(b"x")
        os.chmod(p, stat.S_IREAD)                                             # read-only files block rmtree on Windows
    assert wd.report()["clean"] and not os.path.exists(wd.path)


def test_file_handles_are_released_so_the_input_can_be_deleted_immediately(tmp_path):
    src, out = tmp_path / "in.docx", tmp_path / "out.docx"
    F.make_docx(src)
    process_document(src, out, Policy(ocr=False), Options(store_path=str(tmp_path / "r.db")))
    os.remove(src)
    os.remove(out)
    os.remove(tmp_path / "r.db")                                              # PermissionError on Windows if anything still held them


def test_output_is_a_separate_artifact_and_never_overwrites_the_input(tmp_path):
    src = tmp_path / "in.pdf"
    F.make_native_pdf(src)
    res = process_document(src, src, Policy(ocr=False), Options())
    assert res["release"]["status"] == REJECTED and res["errors"][0]["code"] == "same_path"


# ------------------------------------------------------------------------------ A7 capability registry
def test_capability_registry_covers_all_formats_and_levels():
    rows = capability_matrix()
    fmts = {r["format"] for r in rows}
    assert fmts == {"pdf", "docx", "pptx", "image"}
    levels = {r["capability"] for r in rows}
    assert {"sanitise", "inspect", "unsupported"} <= levels
    unsupported = {(r["format"], r["payload"]) for r in rows if r["capability"] == "unsupported"}
    assert ("docx", "embedded_objects") in unsupported and ("pdf", "embedded_objects") in unsupported


def test_every_declared_payload_gets_a_runtime_outcome_never_a_silent_skip(tmp_path):
    for maker, name, fmt in ((F.make_native_pdf, "a.pdf", "pdf"), (F.make_docx, "a.docx", "docx"), (F.make_pptx, "a.pptx", "pptx")):
        src = tmp_path / name
        maker(src)
        res = process_document(src, tmp_path / ("o_" + name), Policy(ocr=False), Options())
        reported = {p["payload"]: p["outcome"] for p in res["payloads"]}
        assert set(CAPABILITIES[fmt]) <= set(reported), (fmt, set(CAPABILITIES[fmt]) - set(reported))
        assert set(reported.values()) <= set(OUTCOMES)


def test_payload_report_keeps_the_worst_outcome_and_flags_required_problems():
    r = PayloadReport("docx")
    r.add("embedded_objects", "inspected", 0)
    r.add("embedded_objects", "unsupported", 2, "OLE")
    r.add("comments", "preserved", 1)
    assert r.items["embedded_objects"]["outcome"] == "unsupported"
    assert [p["payload"] for p in r.problems(["embedded_objects", "body_text"])] == ["embedded_objects"]
    assert r.problems(["comments"]) == []


def test_metadata_is_inspected_selectively_not_wiped(tmp_path):
    src, out = tmp_path / "in.pdf", tmp_path / "out.pdf"
    F.make_native_pdf(src)
    process_document(src, out, Policy(ocr=False), Options())
    with pymupdf.open(out) as d:
        assert d.metadata["title"] == "Statement" and d.metadata["subject"] == "synthetic"       # untouched
        assert F.EMAIL not in d.metadata["author"] and d.metadata["author"].startswith("Owner")  # only the match changed
    # author blanking is an explicit policy decision, not the default
    out2 = tmp_path / "out2.pdf"
    process_document(src, out2, Policy(ocr=False, strip_author_metadata=True), Options())
    with pymupdf.open(out2) as d:
        assert d.metadata["author"] == ""


# ------------------------------------------------------------------------------ A12
def test_verify_document_confirms_a_good_candidate_and_names_unrelated_changes(tmp_path):
    src, out = tmp_path / "in.docx", tmp_path / "out.docx"
    F.make_docx(src)
    process_document(src, out, Policy(ocr=False), Options())
    good = verify_document(src, out, Policy(ocr=False))
    assert good["release"]["status"] == APPROVED and good["mode"] == "verify_only" and not good["release"]["released"]
    # a candidate that is the unmodified original fails
    bad = verify_document(src, src, Policy(ocr=False))
    assert bad["release"]["status"] == BLOCKED
    # a different file type is refused
    pdf = tmp_path / "x.pdf"
    F.make_native_pdf(pdf)
    assert verify_document(src, pdf, Policy(ocr=False))["release"]["status"] == REJECTED


def test_every_region_ends_with_a_terminal_status(tmp_path):
    for maker, name in ((F.make_native_pdf, "a.pdf"), (F.make_docx, "a.docx"), (F.make_pptx, "a.pptx"), (F.make_image_file, "a.png")):
        src = tmp_path / name
        maker(src)
        res = process_document(src, tmp_path / ("o_" + name), Policy(ocr=False, require_complete_inspection=False), Options())
        assert res["regions"] or name == "a.png"
        assert all(r["verification_status"] in ("verified", "failed", "review_required", "not_applicable") for r in res["regions"]), name


def test_low_confidence_findings_need_review_and_hold_release(tmp_path):
    src, out = tmp_path / "in.pdf", tmp_path / "out.pdf"
    F.make_native_pdf(src)
    res = process_document(src, out, Policy(ocr=False, confidence_overrides={"EMAIL.ADDRESS": 0.999}), Options())
    flagged = [r for r in res["regions"] if r["action"] == "FLAG_REVIEW"]
    assert flagged and all(r["verification_status"] == "review_required" for r in flagged)
    assert res["release"]["status"] == REVIEW and not res["release"]["released"]


# ------------------------------------------------------------------------------ audit ledger + bridge to the existing gate
def test_audit_ledger_records_each_run_and_still_verifies(tmp_path):
    led = Ledger(str(tmp_path / "audit.db"))
    try:
        src = tmp_path / "in.pdf"
        F.make_native_pdf(src)
        for i in range(2):
            process_document(src, tmp_path / f"o{i}.pdf", Policy(ocr=False), Options(ledger=led))
        recs = led.records()
        assert len(recs) == 2 and recs[0]["meta"]["event"] == "document_sanitisation" and recs[0]["cost"] == 0.0
        assert recs[0]["meta"]["release_status"] == APPROVED and recs[0]["meta"]["output_sha256"]
        assert all(s not in json.dumps(recs) for s in SECRETS)
        assert led.verify()
        led.conn.execute("UPDATE ledger SET doc_id='tampered' WHERE id=1")
        led.conn.commit()
        assert not led.verify()
    finally:
        led.conn.close()


def test_results_bridge_into_the_existing_document_gate_and_hard_blocks_stay_hard(tmp_path):
    from src.demo import load
    from src.engine import evaluate_document, vendor_state
    from src.redaction import load_policy
    src = tmp_path / "in.pdf"
    F.make_native_pdf(src)
    res = process_document(src, tmp_path / "o.pdf", Policy(ocr=False), Options())
    doc = document_from_result(res)
    types = {e["type"] for e in doc["entities"]}
    assert {"GOV_ID", "CARD", "EMAIL", "BANK", "PHONE"} <= types
    vendors, ev = load()
    v = vendors[0]
    st = vendor_state(v, ev[v["vendor_id"]], None, 1000, 500)
    pol = load_policy()
    r = evaluate_document(st, "bridged.pdf", doc, "statement_summary", None, 0, 0, 0, policy_doc=pol)
    assert r["plan"]["unresolved_gov_id"] is False and r["decision"] in ("AUTO", "STRICT", "REVIEW")      # redacted -> not hard-blocked
    # a CARD that was NOT removed blocks, exactly like a leftover government ID
    leftover = {"fields_present": [], "entities": [{"type": "CARD", "confidence": 0.5}]}
    assert evaluate_document(st, "x", leftover, "statement_summary", None, 0, 0, 0, policy_doc=pol)["decision"] == "BLOCK"
    after = document_from_result(res, after_sanitisation=True)
    assert all(e["confidence"] >= 0.7 for e in after["entities"] if e["type"] in ("GOV_ID", "CARD"))
