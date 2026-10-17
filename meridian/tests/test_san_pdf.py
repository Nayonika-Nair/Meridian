"""A4 - PDF sanitisation: native, scanned, mixed, rotated; genuine redaction; verification and release gating."""
import pymupdf
import pytest

import fixtures_gen as F
from src.sanitize import APPROVED, BLOCKED, REVIEW, Options, Policy, process_document, verify_document
from src.sanitize.imaging import ocr_available

needs_ocr = pytest.mark.skipif(not ocr_available(), reason="tesseract not installed")


def run(src, out, policy=None, options=None):
    return process_document(src, out, policy or Policy(), options or Options())


def text_of(path):
    with pymupdf.open(path) as d:
        return "\n".join(p.get_text() for p in d)


def check(res, name):
    return next(c for c in res["verification"]["checks"] if c["name"] == name)


def raw_bytes_contain(path, value: str) -> bool:
    """Search decoded streams and objects - not just the text layer."""
    with pymupdf.open(path) as d:
        for x in range(1, d.xref_length()):
            blobs = [d.xref_object(x, compressed=False).encode("latin-1", "ignore")]
            try:
                if d.xref_is_stream(x):
                    blobs.append(d.xref_stream(x) or b"")
            except Exception:
                pass
            if any(value.encode("latin-1") in b for b in blobs):
                return True
    return False


# ------------------------------------------------------------------------------ native text
def test_native_pdf_targets_are_removed_for_real_and_unrelated_text_survives(tmp_path):
    src, out = tmp_path / "in.pdf", tmp_path / "out.pdf"
    F.make_native_pdf(src)
    res = run(src, out)
    assert res["release"]["status"] == APPROVED and res["release"]["released"] and out.exists()
    t = text_of(out)
    for secret in (F.SSN, F.CARD, F.EMAIL, F.PHONE):
        assert secret not in t
    assert F.KEEP_PHRASE in t and "This footer line is ordinary text and must be preserved." in t
    assert "Name: Jane Doe" in t or "Name:" in t                                 # PERSON rule is label-driven: see rule docs
    assert not raw_bytes_contain(out, "123-45-6789")                            # not hiding in a content stream
    assert input_untouched(src)


def input_untouched(path):
    return text_of(path).count(F.SSN) == 1


def test_only_the_labelled_occurrence_is_redacted_not_every_equal_string(tmp_path):
    """`Invoice #98765432` shares digits with the labelled account number and must survive."""
    src, out = tmp_path / "in.pdf", tmp_path / "out.pdf"
    F.make_native_pdf(src)
    res = run(src, out, Policy(ocr=False))
    t = text_of(out)
    assert f"Invoice #{F.ACCT} is unrelated to any account" in t
    assert f"Account number: {F.ACCT}" not in t
    assert res["release"]["status"] == APPROVED


def test_replacement_text_is_placed_for_masked_values(tmp_path):
    src, out = tmp_path / "in.pdf", tmp_path / "out.pdf"
    F.make_native_pdf(src)
    run(src, out, Policy(ocr=False))
    t = text_of(out)
    assert "******32" in t                          # masked account number keeps only the allowed tail
    assert "j***@example.com" in t


def test_every_region_has_stable_id_coordinates_and_final_status(tmp_path):
    src, out = tmp_path / "in.pdf", tmp_path / "out.pdf"
    F.make_native_pdf(src)
    res = run(src, out, Policy(ocr=False))
    assert res["regions"]
    for r in res["regions"]:
        assert r["region_id"].startswith("R-") and r["rule_id"] and r["category"]
        assert r["verification_status"] in ("verified", "review_required", "not_applicable")
        if r["space"] == "pdf_unrot_tl":
            assert r["bbox"] and r["unit_id"].startswith("page:") and r["page_size"] == [595.0, 842.0]
    ssn = next(r for r in res["regions"] if r["rule_id"] == "GOV.US_SSN")
    assert ssn["action"] == "REDACT" and ssn["hard_block"] and ssn["source_method"] == "native_text"
    assert "GOV.TAX_ID_LABELLED" in ssn["also_matched"]


@pytest.mark.parametrize("rot", [90, 180, 270])
def test_rotated_native_pages_are_redacted_in_the_right_place(tmp_path, rot):
    src, out = tmp_path / "in.pdf", tmp_path / "out.pdf"
    F.make_native_pdf(src, rotation=rot)
    res = run(src, out, Policy(ocr=False))
    assert res["release"]["status"] == APPROVED
    t = text_of(out)
    assert F.SSN not in t and F.KEEP_PHRASE in t and "ordinary text and must be preserved" in t


# ------------------------------------------------------------------------------ scanned / mixed (OCR)
@needs_ocr
def test_scanned_pdf_is_ocr_mapped_and_pixels_under_the_region_are_destroyed(tmp_path):
    src, out = tmp_path / "scan.pdf", tmp_path / "out.pdf"
    F.make_scanned_pdf(src)
    assert F.SSN not in text_of(src)                                   # no text layer to begin with
    res = run(src, out)
    assert res["release"]["status"] == APPROVED
    ocr_regions = [r for r in res["regions"] if r["source_method"] == "ocr"]
    assert {r["category"] for r in ocr_regions} >= {"GOV_ID", "BANK"}
    for r in ocr_regions:
        assert r["chain"] and r["chain"][-1]["to"] == "pdf_unrot_tl"      # explicit chain back to PDF space
        assert 36 <= r["bbox"][0] <= 559 and 60 <= r["bbox"][1] <= 583
    assert check(res, "ocr_recheck")["status"] == "passed" and check(res, "visual_content_preserved")["status"] == "passed"
    # the embedded image itself no longer contains the SSN pixels: OCR the extracted image
    from PIL import Image
    import io
    import pytesseract
    with pymupdf.open(out) as d:
        xref = d[0].get_images()[0][0]
        img = Image.open(io.BytesIO(d.extract_image(xref)["image"]))
    assert F.SSN not in pytesseract.image_to_string(img.convert("L"))
    assert "Ordinary closing line" in pytesseract.image_to_string(img.convert("L"))


@needs_ocr
@pytest.mark.parametrize("rot", [90, 180, 270])
def test_rotated_scanned_pages_use_orientation_recovery_and_the_page_chain(tmp_path, rot):
    src, out = tmp_path / "scan.pdf", tmp_path / "out.pdf"
    F.make_scanned_pdf(tmp_path / "base.pdf")
    d = pymupdf.open(tmp_path / "base.pdf")
    d[0].set_rotation(rot)
    d.save(str(src))
    d.close()
    res = run(src, out)
    assert res["release"]["status"] == APPROVED, res["verification"]["checks"]
    assert any(r["category"] == "GOV_ID" for r in res["regions"])


@needs_ocr
def test_mixed_native_and_scanned_pages(tmp_path):
    src, out = tmp_path / "mixed.pdf", tmp_path / "out.pdf"
    F.make_mixed_pdf(src)
    res = run(src, out)
    assert res["release"]["status"] == APPROVED
    by_page = {}
    for r in res["regions"]:
        by_page.setdefault(r["unit_id"], set()).add((r["source_method"], r["category"]))
    assert ("native_text", "GOV_ID") in by_page["page:1"]
    assert ("ocr", "CARD") in by_page["page:2"]
    assert any(m.startswith("native_text") and c == "EMAIL" for m, c in by_page["page:3"])       # native, also seen by OCR
    assert ("ocr", "BANK") in by_page["page:3"]                                                 # only the image carries this one
    t = text_of(out)
    assert F.SSN not in t and F.EMAIL not in t and "Native header on a mixed page" in t
    assert res["coverage"]["images_ocr"] == 2 and res["coverage"]["unit_ratio"] == 1.0


@needs_ocr
def test_native_text_seen_twice_by_ocr_is_one_region_with_provenance(tmp_path):
    src, out = tmp_path / "in.pdf", tmp_path / "out.pdf"
    d = pymupdf.open()
    p = d.new_page()
    p.insert_text((72, 100), f"SSN: {F.SSN}", fontsize=14)
    p.insert_image(pymupdf.Rect(300, 200, 400, 250), stream=F.image_bytes(F.text_image(["logo"], size=(200, 100))))   # forces OCR on a native page
    d.save(str(src))
    d.close()
    res = run(src, out)
    ssn = [r for r in res["regions"] if r["rule_id"] == "GOV.US_SSN"]
    assert len(ssn) == 1 and ssn[0]["source_method"] in ("native_text", "native_text+ocr")


# ------------------------------------------------------------------------------ the verifier is not fooled by overlays
def test_visual_overlay_is_not_accepted_as_sanitisation(tmp_path):
    src, fake = tmp_path / "in.pdf", tmp_path / "fake.pdf"
    F.make_native_pdf(src)
    d = pymupdf.open(src)
    for hit in d[0].search_for(F.SSN):
        d[0].draw_rect(hit, color=(0, 0, 0), fill=(0, 0, 0))
    d.save(str(fake))
    d.close()
    res = verify_document(src, fake, Policy())
    assert res["release"]["status"] == BLOCKED
    assert check(res, "target_text_removed_native")["status"] == "failed"
    assert check(res, "hard_block_values_absent")["status"] == "failed"
    assert not res["release"]["released"]


def test_verification_failure_prevents_release_and_quarantines_the_artifact(tmp_path, monkeypatch):
    import src.sanitize.pdf_ops as pdf_ops
    src, out = tmp_path / "in.pdf", tmp_path / "out.pdf"
    F.make_native_pdf(src)
    monkeypatch.setattr(pymupdf.Page, "apply_redactions", lambda self, *a, **k: False)     # simulate a broken redaction
    res = run(src, out, Policy(ocr=False))
    assert res["release"]["status"] == BLOCKED and not res["release"]["released"]
    assert not out.exists() and (tmp_path / "out.pdf.quarantined").exists()
    assert res["output"]["released"] is False
    assert res["sanitisation"]["failed"] > 0
    assert check(res, "hard_block_values_absent")["status"] == "failed"


def test_policy_can_downgrade_failure_to_review_but_never_to_release(tmp_path, monkeypatch):
    src, out = tmp_path / "in.pdf", tmp_path / "out.pdf"
    F.make_native_pdf(src)
    monkeypatch.setattr(pymupdf.Page, "apply_redactions", lambda self, *a, **k: False)
    res = run(src, out, Policy(ocr=False, release_requires_verification=False))
    assert res["release"]["status"] == REVIEW and not res["release"]["released"]


# ------------------------------------------------------------------------------ non-page payloads
def test_annotations_form_fields_links_bookmarks_metadata_and_xmp(tmp_path):
    src, out = tmp_path / "in.pdf", tmp_path / "out.pdf"
    d = pymupdf.open()
    p = d.new_page()
    p.insert_text((72, 100), "Body text stays", fontsize=12)
    a = p.add_freetext_annot(pymupdf.Rect(72, 150, 400, 180), f"Contact {F.EMAIL}", fontsize=11)
    a.set_info(content=f"Contact {F.EMAIL}")
    a.update()
    w = pymupdf.Widget()
    w.field_name, w.field_type, w.rect, w.field_value = "ssn", pymupdf.PDF_WIDGET_TYPE_TEXT, pymupdf.Rect(72, 220, 300, 245), F.SSN
    p.add_widget(w)
    p.insert_link({"kind": pymupdf.LINK_URI, "from": pymupdf.Rect(72, 300, 200, 320), "uri": f"mailto:{F.EMAIL}"})
    d.set_toc([[1, f"Report for Name: {F.NAME}", 1]])
    d.set_metadata({"title": "t", "author": f"x {F.EMAIL}"})
    d.set_xml_metadata(f"<x:xmpmeta xmlns:x='adobe:ns:meta/'><rdf>{F.EMAIL}</rdf></x:xmpmeta>")
    d.save(str(src))
    d.close()
    res = run(src, out, Policy(ocr=False))
    assert res["release"]["status"] == APPROVED, res["verification"]["checks"]
    with pymupdf.open(out) as o:
        pg = o[0]
        assert "Body text stays" in pg.get_text()
        assert all(F.EMAIL not in (an.info.get("content") or "") for an in pg.annots())
        assert all((w.field_value or "") != F.SSN for w in pg.widgets())
        assert all(F.EMAIL not in (l.get("uri") or "") for l in pg.get_links())
        assert F.NAME not in str(o.get_toc())
        assert F.EMAIL not in str(o.metadata) and F.EMAIL not in (o.get_xml_metadata() or "")
    payloads = {x["payload"]: x["outcome"] for x in res["payloads"]}
    assert payloads["annotations"] in ("inspected", "sanitised") and payloads["form_fields"] == "inspected"


def test_white_hidden_text_is_still_text_and_is_removed(tmp_path):
    src, out = tmp_path / "in.pdf", tmp_path / "out.pdf"
    d = pymupdf.open()
    p = d.new_page()
    p.insert_text((72, 100), "Visible line", fontsize=12)
    p.insert_text((72, 140), f"SSN: {F.SSN}", fontsize=12, color=(1, 1, 1))        # white on white
    d.save(str(src))
    d.close()
    res = run(src, out, Policy(ocr=False))
    assert res["release"]["status"] == APPROVED and F.SSN not in text_of(out)
    assert any(x["payload"] == "hidden_text" and x["count"] > 0 for x in res["payloads"])


def test_embedded_file_attachments_are_reported_unsupported_and_force_review(tmp_path):
    src, out = tmp_path / "in.pdf", tmp_path / "out.pdf"
    d = pymupdf.open()
    d.new_page().insert_text((72, 100), "Cover page", fontsize=12)
    d.embfile_add("notes.txt", f"private: SSN {F.SSN}".encode())
    d.embfile_add("blob.bin", b"\x00\x01\x02binary")
    d.save(str(src))
    d.close()
    res = run(src, out, Policy(ocr=False))
    assert res["release"]["status"] == REVIEW and res["incomplete"]
    assert any(u["payload"] == "embedded_objects" and u["outcome"] == "unsupported" and u["count"] == 2 for u in res["unsupported"])
    assert any(r["unit_id"].startswith("attachment:") and r["verification_status"] == "review_required" for r in res["regions"])
    # requiring complete inspection is a policy choice, never silent
    lax = run(src, tmp_path / "out2.pdf", Policy(ocr=False, require_complete_inspection=False))
    assert any(r["unit_id"].startswith("attachment:") for r in lax["regions"]) and lax["release"]["status"] == REVIEW


def test_pdf_javascript_is_reported_not_silently_ignored(tmp_path):
    src, out = tmp_path / "in.pdf", tmp_path / "out.pdf"
    d = pymupdf.open()
    d.new_page().insert_text((72, 100), "hello", fontsize=12)
    d.save(str(src))
    d.close()
    d = pymupdf.open(src)
    xref = d.get_new_xref()
    d.update_object(xref, "<< /S /JavaScript /JS (app.alert\\('x'\\);) >>")
    d.xref_set_key(d.pdf_catalog(), "OpenAction", f"{xref} 0 R")
    d.save(str(tmp_path / "js.pdf"), garbage=0)
    d.close()
    src = tmp_path / "js.pdf"
    res = run(src, out, Policy(ocr=False))
    js = next(x for x in res["payloads"] if x["payload"] == "javascript_macros")
    assert js["outcome"] == "preserved" and js["count"] >= 1 and any("javascript" in w for w in res["warnings"])
