"""A10 - content-based MIME validation and processing limits; malformed / encrypted / unsupported inputs."""
import io
import os
import shutil
import tempfile
import zipfile

import pytest
from PIL import Image

import fixtures_gen as F
from src.sanitize import APPROVED, FAILED, REJECTED, REVIEW, Limits, Options, Policy, process_document
from src.sanitize.imaging import OcrTimeout
from src.sanitize.safety import InputRejected, detect_type, validate_input


def run(src, out, **pol):
    return process_document(src, out, Policy(**pol), Options())


def rejected_code(res):
    assert res["release"]["status"] == REJECTED, res["release"]
    return res["errors"][0]["code"]


# ------------------------------------------------------------------------------ content, not extension
def test_detect_type_uses_content_not_extension(tmp_path):
    pdf, docx, png = tmp_path / "a.txt", tmp_path / "b.bin", tmp_path / "c.pdf"
    F.make_native_pdf(pdf)
    F.make_docx(docx)
    F.make_image_file(png)
    assert detect_type(pdf)[0] == "pdf" and detect_type(docx)[0] == "docx" and detect_type(png)[0] == "png"


def test_extension_mismatch_is_rejected_by_default_and_processed_by_content_when_allowed(tmp_path):
    lie = tmp_path / "statement.pdf"
    F.make_docx(lie)                                       # a Word document pretending to be a PDF
    assert rejected_code(run(lie, tmp_path / "o.pdf")) == "extension_mismatch"
    res = process_document(lie, tmp_path / "o.docx", Policy(allow_extension_mismatch=True, ocr=False), Options())
    assert res["input"]["detected_type"] == "docx" and res["input"]["extension_mismatch"]
    assert any("extension_mismatch" in w for w in res["warnings"])
    # output extension must also match the detected content
    ok = tmp_path / "ok.docx"
    F.make_docx(ok)
    assert rejected_code(run(ok, tmp_path / "wrong.pdf")) == "output_extension_mismatch"


def test_text_file_renamed_pdf_is_unsupported(tmp_path):
    p = tmp_path / "x.pdf"
    p.write_text("just some text with SSN 123-45-6789")
    assert rejected_code(run(p, tmp_path / "o.pdf")) == "unsupported_type"


def test_input_and_output_must_differ(tmp_path):
    p = tmp_path / "x.pdf"
    F.make_native_pdf(p)
    assert rejected_code(run(p, p)) == "same_path"
    assert p.exists()


# ------------------------------------------------------------------------------ explicit handling of bad files
def test_empty_missing_and_corrupt_files(tmp_path):
    empty = tmp_path / "e.pdf"
    empty.write_bytes(b"")
    assert rejected_code(run(empty, tmp_path / "o.pdf")) == "empty"
    assert rejected_code(run(tmp_path / "nope.pdf", tmp_path / "o.pdf")) == "not_found"
    bad_zip = tmp_path / "c.docx"
    bad_zip.write_bytes(b"PK\x03\x04" + os.urandom(200))
    assert rejected_code(run(bad_zip, tmp_path / "o.docx")) in ("malformed", "unsupported_type")
    bad_pdf = tmp_path / "m.pdf"
    bad_pdf.write_bytes(b"%PDF-1.7\n1 0 obj\n<< garbage")
    code = rejected_code(run(bad_pdf, tmp_path / "o2.pdf"))
    assert code in ("malformed",)


def test_encrypted_pdf_is_rejected_explicitly(tmp_path):
    import pymupdf
    src = tmp_path / "plain.pdf"
    F.make_native_pdf(src)
    enc = tmp_path / "enc.pdf"
    d = pymupdf.open(src)
    d.save(str(enc), encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="o", user_pw="u")
    d.close()
    res = run(enc, tmp_path / "o.pdf")
    assert rejected_code(res) == "encrypted" and not (tmp_path / "o.pdf").exists()


def test_legacy_or_encrypted_office_container_is_rejected(tmp_path):
    p = tmp_path / "old.docx"
    p.write_bytes(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 600)
    assert rejected_code(run(p, tmp_path / "o.docx")) == "encrypted_or_legacy"


def test_spreadsheets_are_unsupported(tmp_path):
    p = tmp_path / "s.xlsx"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("xl/workbook.xml", "<workbook/>")
        z.writestr("[Content_Types].xml", "<Types/>")
    assert rejected_code(run(p, tmp_path / "o.xlsx")) == "unsupported_type"


# ------------------------------------------------------------------------------ configurable limits
def test_file_size_limit(tmp_path):
    p = tmp_path / "x.pdf"
    F.make_native_pdf(p)
    res = process_document(p, tmp_path / "o.pdf", Policy(limits=Limits(max_file_bytes=500)), Options())
    assert rejected_code(res) == "file_too_large"


def test_page_count_limit(tmp_path):
    import pymupdf
    p = tmp_path / "many.pdf"
    d = pymupdf.open()
    for _ in range(6):
        d.new_page().insert_text((72, 72), "page")
    d.save(str(p))
    d.close()
    assert rejected_code(process_document(p, tmp_path / "o.pdf", Policy(limits=Limits(max_pages=5)), Options())) == "too_many_pages"


def test_slide_count_limit(tmp_path):
    p = tmp_path / "deck.pptx"
    F.make_pptx(p)
    res = process_document(p, tmp_path / "o.pptx", Policy(limits=Limits(max_slides=1)), Options())
    assert rejected_code(res) == "too_many_slides"


def test_image_dimension_limit(tmp_path):
    p = tmp_path / "big.png"
    Image.new("RGB", (900, 900), "white").save(p)
    res = process_document(p, tmp_path / "o.png", Policy(limits=Limits(max_image_pixels=100_000)), Options())
    assert rejected_code(res) == "image_too_large"


def test_archive_expansion_limits(tmp_path):
    bomb = tmp_path / "bomb.docx"
    with zipfile.ZipFile(bomb, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("word/document.xml", "<w:document/>")
        z.writestr("word/media/huge.bin", b"\x00" * 8_000_000)
    res = process_document(bomb, tmp_path / "o.docx", Policy(limits=Limits(max_archive_ratio=50)), Options())
    assert rejected_code(res) == "archive_expansion"
    res = process_document(bomb, tmp_path / "o2.docx", Policy(limits=Limits(max_archive_uncompressed_bytes=1_000_000, max_archive_ratio=1e9)), Options())
    assert rejected_code(res) == "archive_expansion"


def test_archive_entry_count_and_unsafe_paths(tmp_path):
    many = tmp_path / "many.docx"
    with zipfile.ZipFile(many, "w") as z:
        z.writestr("word/document.xml", "<w:document/>")
        for i in range(30):
            z.writestr(f"x/{i}.txt", "a")
    assert rejected_code(process_document(many, tmp_path / "o.docx", Policy(limits=Limits(max_archive_entries=10)), Options())) == "archive_too_many_entries"
    evil = tmp_path / "evil.docx"
    with zipfile.ZipFile(evil, "w") as z:
        z.writestr("word/document.xml", "<w:document/>")
        z.writestr("../../escape.txt", "x")
    assert rejected_code(run(evil, tmp_path / "o2.docx")) == "archive_unsafe_path"


def test_embedded_object_count_limit(tmp_path):
    p = tmp_path / "ole.docx"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("word/document.xml", "<w:document/>")
        for i in range(5):
            z.writestr(f"word/embeddings/o{i}.bin", b"x")
    assert rejected_code(process_document(p, tmp_path / "o.docx", Policy(limits=Limits(max_embedded_objects=3)), Options())) == "too_many_embedded_objects"


def test_processing_duration_limit_fails_closed_and_releases_nothing(tmp_path):
    p = tmp_path / "x.pdf"
    F.make_native_pdf(p)
    res = process_document(p, tmp_path / "o.pdf", Policy(limits=Limits(max_duration_s=0.0)), Options())
    assert res["release"]["status"] == FAILED and res["errors"][0]["code"] == "duration_limit"
    assert not (tmp_path / "o.pdf").exists() and not (tmp_path / "o.pdf.quarantined").exists()


def test_ocr_timeout_is_reported_and_forces_review(tmp_path):
    p = tmp_path / "scan.pdf"
    F.make_scanned_pdf(p)

    def slow(img, timeout):
        raise OcrTimeout("Tesseract process timeout")

    res = process_document(p, tmp_path / "o.pdf", Policy(), Options(ocr_engine=slow))
    assert res["release"]["status"] == REVIEW and not res["release"]["released"]
    assert any(u["payload"] == "scanned_images" and u["outcome"] == "failed" for u in res["payloads"])
    assert res["coverage"]["images_ocr_failed"] >= 1 and res["incomplete"]


def test_ocr_budget_exhaustion_is_reported_not_silent(tmp_path):
    p = tmp_path / "scan.pdf"
    F.make_scanned_pdf(p)
    res = process_document(p, tmp_path / "o.pdf", Policy(limits=Limits(ocr_total_s=0.1)), Options(ocr_engine=lambda i, t: []))
    assert res["release"]["status"] == REVIEW
    assert any(u["payload"] == "scanned_images" for u in res["unsupported"])


def test_ocr_disabled_by_policy_marks_images_unsupported_not_clean(tmp_path):
    p = tmp_path / "scan.pdf"
    F.make_scanned_pdf(p)
    res = process_document(p, tmp_path / "o.pdf", Policy(ocr=False), Options())
    assert res["release"]["status"] == REVIEW
    assert any(u["payload"] == "scanned_images" and u["outcome"] == "unsupported" for u in res["payloads"])
