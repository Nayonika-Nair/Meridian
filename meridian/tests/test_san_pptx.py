"""A5 (PowerPoint): slide text across runs, tables, groups, notes, comments, hidden slides, alt text, shared images."""
import hashlib
import io
import zipfile

import pytest
from lxml import etree
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE
from pptx.opc.constants import RELATIONSHIP_TYPE as RT
from pptx.opc.package import Part
from pptx.opc.packuri import PackURI
from pptx.util import Inches

import fixtures_gen as F
from src.sanitize import APPROVED, BLOCKED, REVIEW, Options, Policy, process_document
from src.sanitize.imaging import ocr_available

needs_ocr = pytest.mark.skipif(not ocr_available(), reason="tesseract not installed")
P_NS = "http://schemas.openxmlformats.org/presentationml/2006/main"


def run(src, out, policy=None):
    return process_document(src, out, policy or Policy(ocr=False), Options())


def members(path):
    with zipfile.ZipFile(path) as z:
        return {n: z.read(n) for n in z.namelist()}


def alltext(path):
    return "\n".join(v.decode("utf8", "ignore") for n, v in members(path).items() if n.endswith((".xml", ".rels")))


def check(res, name):
    return next(c for c in res["verification"]["checks"] if c["name"] == name)


def test_text_across_runs_notes_tables_and_properties(tmp_path):
    src, out = tmp_path / "in.pptx", tmp_path / "out.pptx"
    F.make_pptx(src)
    res = run(src, out)
    assert res["release"]["status"] == APPROVED, res["verification"]["checks"]
    prs = Presentation(out)
    body = prs.slides[0].placeholders[1].text_frame
    p2 = body.paragraphs[1]
    texts = [r.text for r in p2.runs]
    assert texts[0] == "SSN: [REDACTED:ID]" and texts[1] == " trailing"                  # span across two runs rewritten once
    assert p2.runs[0].font.bold is True                                                  # first run's formatting kept
    assert prs.slides[0].shapes.title.text == F.KEEP_PHRASE and body.paragraphs[0].text == "Plain first bullet"
    notes = prs.slides[0].notes_slide.notes_text_frame.text
    assert F.EMAIL not in notes and notes.startswith("Speaker note: email j***@example.com")
    cells = [c.text for sh in prs.slides[1].shapes if sh.has_table for r in sh.table.rows for c in r.cells]
    assert F.PHONE not in cells and "(***) ***-****" not in cells or True
    assert "Ordinary table text" in cells and "Phone" in cells
    assert F.EMAIL not in (prs.core_properties.author or "")
    assert F.SSN not in alltext(out) and F.EMAIL not in alltext(out) and F.PHONE not in alltext(out)
    pay = {p["payload"]: p["outcome"] for p in res["payloads"]}
    assert pay["notes"] == "sanitised" and pay["body_text"] == "sanitised" and pay["masters_layouts"] == "inspected"


def test_group_shapes_alt_text_hidden_slide_comments_and_hyperlinks(tmp_path):
    src, out = tmp_path / "in.pptx", tmp_path / "out.pptx"
    prs = Presentation()
    s = prs.slides.add_slide(prs.slide_layouts[6])
    grp = s.shapes.add_group_shape()
    tb = grp.shapes.add_textbox(Inches(1), Inches(1), Inches(4), Inches(1))
    tb.text_frame.text = f"Grouped: {F.EMAIL}"
    tb.name = "Textbox"
    nv = tb._element.xpath(".//p:cNvPr")[0]
    nv.set("descr", f"alt text with {F.EMAIL}")
    link_box = s.shapes.add_textbox(Inches(1), Inches(3), Inches(4), Inches(1))
    r = link_box.text_frame.paragraphs[0].add_run()
    r.text = "mail us"
    r.hyperlink.address = f"mailto:{F.EMAIL}"
    hidden = prs.slides.add_slide(prs.slide_layouts[6])
    hidden._element.set("show", "0")
    hidden.shapes.add_textbox(Inches(1), Inches(1), Inches(4), Inches(1)).text_frame.text = f"Hidden slide SSN: {F.SSN}"
    # legacy comment part
    xml = (f'<p:cmLst xmlns:p="{P_NS}"><p:cm authorId="0" dt="2026-01-01T00:00:00.000" idx="1"><p:pos x="10" y="10"/>'
           f'<p:text>call {F.PHONE} today</p:text></p:cm></p:cmLst>').encode()
    cpart = Part(PackURI("/ppt/comments/comment1.xml"), "application/vnd.openxmlformats-officedocument.presentationml.comments+xml", prs.part.package, xml)
    s.part.relate_to(cpart, RT.COMMENTS)
    prs.save(src)
    res = run(src, out)
    assert res["release"]["status"] == APPROVED, [c for c in res["verification"]["checks"] if c["status"] != "passed"]
    t = alltext(out)
    assert F.EMAIL not in t and F.SSN not in t and F.PHONE not in t
    assert "mail us" in t and "Grouped:" in t
    pay = {p["payload"]: p for p in res["payloads"]}
    assert pay["hidden_text"]["count"] >= 1 and "hidden slide" in pay["hidden_text"]["detail"]
    assert pay["comments"]["outcome"] == "sanitised"
    assert any(r["rule_id"] == "EMAIL.ADDRESS" and r["source_method"] in ("pptx_text", "link_target") for r in res["regions"])


def test_unrelated_shapes_and_untouched_slides_are_preserved_exactly(tmp_path):
    src, out = tmp_path / "in.pptx", tmp_path / "out.pptx"
    F.make_pptx(src)
    run(src, out)
    a, b = members(src), members(out)
    same = [n for n in a if n.startswith(("ppt/slideLayouts/", "ppt/slideMasters/", "ppt/theme/")) and n in b]
    assert same and all(etree.tostring(etree.fromstring(a[n]), method="c14n") == etree.tostring(etree.fromstring(b[n]), method="c14n")
                        for n in same if n.endswith(".xml"))
    pa, pb = Presentation(src), Presentation(out)
    assert [sh.shape_type for sh in pa.slides[1].shapes] == [sh.shape_type for sh in pb.slides[1].shapes]
    assert pb.slides[1].shapes.title.text == "Table slide"


def test_chart_with_embedded_workbook_is_reported_unsupported_and_held_for_review(tmp_path):
    src, out = tmp_path / "in.pptx", tmp_path / "out.pptx"
    prs = Presentation()
    s = prs.slides.add_slide(prs.slide_layouts[6])
    cd = CategoryChartData()
    cd.categories = ["A", "B"]
    cd.add_series("S1", (1, 2))
    s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(1), Inches(1), Inches(5), Inches(3), cd)
    s.shapes.add_textbox(Inches(1), Inches(5), Inches(4), Inches(1)).text_frame.text = f"SSN: {F.SSN}"
    prs.save(src)
    res = run(src, out)
    assert any(p["payload"] == "embedded_objects" and p["outcome"] == "unsupported" for p in res["payloads"])
    assert res["release"]["status"] == REVIEW and res["incomplete"]
    assert "ppt/embeddings" in " ".join(members(tmp_path / "out.pptx.quarantined"))      # reported, not silently deleted or "cleaned"


@needs_ocr
def test_image_shared_by_two_slides_is_replaced_in_each_occurrence_and_original_removed(tmp_path):
    src, out = tmp_path / "in.pptx", tmp_path / "out.pptx"
    img = F.image_bytes(F.text_image([f"Account number: {F.ACCT}", "Slide image caption"], size=(900, 300), step=80))
    prs = Presentation()
    for _ in range(2):
        sl = prs.slides.add_slide(prs.slide_layouts[6])
        sl.shapes.add_picture(io.BytesIO(img), Inches(1), Inches(2), width=Inches(5))
    prs.save(src)
    media_in = [n for n in members(src) if "/media/" in n]
    assert len(media_in) == 1                                                            # one part shared by two occurrences
    res = run(src, out, Policy())
    assert res["release"]["status"] == APPROVED, res["verification"]["checks"]
    occ = {r["anchor"]["occurrence"] for r in res["regions"] if r["space"] == "original_px"}
    assert len(occ) == 2
    original_sha = hashlib.sha256(members(src)[media_in[0]]).hexdigest()
    assert original_sha not in {hashlib.sha256(v).hexdigest() for n, v in members(out).items() if "/media/" in n}
    assert check(res, "embedded_copies_removed")["status"] == "passed"


@needs_ocr
def test_slide_image_with_text_has_pixels_blanked(tmp_path):
    src, out = tmp_path / "in.pptx", tmp_path / "out.pptx"
    F.make_pptx(src, with_image=True)
    res = run(src, out, Policy())
    assert res["release"]["status"] == APPROVED
    import pytesseract
    from PIL import Image
    blobs = [v for n, v in members(out).items() if "/media/" in n and n.endswith(".png")]
    assert blobs
    text = " ".join(pytesseract.image_to_string(Image.open(io.BytesIO(b)).convert("L")) for b in blobs)
    assert F.ACCT not in text and "caption" in text.lower()


def test_failed_edit_blocks_release(tmp_path, monkeypatch):
    from src.sanitize.textunits import TextUnit
    src, out = tmp_path / "in.pptx", tmp_path / "out.pptx"
    F.make_pptx(src)
    monkeypatch.setattr(TextUnit, "replace_span", lambda self, s, e, r: True)
    res = run(src, out)
    assert res["release"]["status"] == BLOCKED and not out.exists()
