"""Synthetic fixtures for the sanitisation tests. Nothing here is real personal data.

All sensitive values are well-known documentation/test values (e.g. the Luhn-valid 4111... card number).
"""
from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SSN = "123-45-6789"
CARD = "4111 1111 1111 1111"
ACCT = "98765432"
EMAIL = "jane.doe@example.com"
PHONE = "(415) 555-0132"
NAME = "Jane Doe"
AWS = "AKIAIOSFODNN7EXAMPLE"
KEEP_PHRASE = "Quarterly revenue summary"      # unrelated content that must survive

TARGETS = {"ssn": SSN, "card": CARD, "acct": ACCT, "email": EMAIL, "phone": PHONE, "name": NAME, "aws": AWS}


def font(size: int = 30):
    return ImageFont.load_default(size=size)


def text_image(lines: list[str], size=(1000, 560), start=(60, 60), step=70, fsize=34) -> Image.Image:
    img = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(img)
    f = font(fsize)
    y = start[1]
    for ln in lines:
        d.text((start[0], y), ln, fill="black", font=f)
        y += step
    return img


def image_bytes(img: Image.Image, fmt="PNG") -> bytes:
    b = io.BytesIO()
    img.save(b, fmt)
    return b.getvalue()


# ------------------------------------------------------------------------------- PDFs
def make_native_pdf(path: str | Path, *, rotation: int = 0) -> None:
    import pymupdf
    doc = pymupdf.open()
    p = doc.new_page(width=595, height=842)
    y = 90
    for ln in (
        KEEP_PHRASE,
        f"Name: {NAME}",
        f"SSN: {SSN}",
        f"Account number: {ACCT}",
        f"Invoice #{ACCT} is unrelated to any account",
        f"Email: {EMAIL}   Phone: {PHONE}",
        f"Card: {CARD}",
        "This footer line is ordinary text and must be preserved.",
    ):
        p.insert_text((72, y), ln, fontsize=12)
        y += 26
    if rotation:
        p.set_rotation(rotation)
    doc.set_metadata({"title": "Statement", "author": f"Owner {EMAIL}", "subject": "synthetic"})
    doc.save(str(path))
    doc.close()


def make_scanned_pdf(path: str | Path, lines: list[str] | None = None) -> None:
    """A page that is ONE image (no text layer)."""
    import pymupdf
    img = text_image(lines or [KEEP_PHRASE, f"SSN: {SSN}", f"Account number: {ACCT}", "Ordinary closing line"])
    doc = pymupdf.open()
    p = doc.new_page(width=595, height=842)
    p.insert_image(pymupdf.Rect(36, 60, 559, 60 + 523 * 560 / 1000), stream=image_bytes(img))
    doc.save(str(path))
    doc.close()


def make_mixed_pdf(path: str | Path) -> None:
    """Page 1 native; page 2 scanned image; page 3 native text + an image carrying text."""
    import pymupdf
    doc = pymupdf.open()
    p1 = doc.new_page(width=595, height=842)
    p1.insert_text((72, 100), KEEP_PHRASE, fontsize=12)
    p1.insert_text((72, 130), f"SSN: {SSN}", fontsize=12)
    p2 = doc.new_page(width=595, height=842)
    p2.insert_image(pymupdf.Rect(36, 60, 559, 60 + 523 * 560 / 1000),
                    stream=image_bytes(text_image([KEEP_PHRASE, f"Card: {CARD}", "Closing scanned line"])))
    p3 = doc.new_page(width=595, height=842)
    p3.insert_text((72, 100), "Native header on a mixed page", fontsize=12)
    p3.insert_text((72, 130), f"Email: {EMAIL}", fontsize=12)
    p3.insert_image(pymupdf.Rect(36, 200, 559, 200 + 523 * 560 / 1000),
                    stream=image_bytes(text_image([f"Account number: {ACCT}", "Scanned footnote line"])))
    doc.save(str(path))
    doc.close()


def make_overlay_only_pdf(path: str | Path) -> None:
    """A deliberately WRONG 'redaction': black box drawn over live text. Verification must reject it."""
    import pymupdf
    doc = pymupdf.open()
    p = doc.new_page()
    p.insert_text((72, 100), f"SSN: {SSN}", fontsize=12)
    doc.save(str(path))
    doc.close()


# ------------------------------------------------------------------------------- DOCX
def make_docx(path: str | Path, *, with_image: bool = False, shared_image: bool = False) -> None:
    from docx import Document
    from docx.shared import Pt, RGBColor
    d = Document()
    d.core_properties.author = f"Owner {EMAIL}"
    d.core_properties.title = "Synthetic statement"
    d.add_paragraph(KEEP_PHRASE)
    # SSN split across three runs with different formatting
    p = d.add_paragraph()
    p.add_run("SSN: ")
    r1 = p.add_run("123-45")
    r1.bold = True
    r2 = p.add_run("-67")
    r2.italic = True
    r2.font.color.rgb = RGBColor(0xAA, 0x00, 0x00)
    r3 = p.add_run("89 end of line")
    r3.font.size = Pt(14)
    d.add_paragraph(f"Account number: {ACCT}")
    d.add_paragraph(f"Invoice #{ACCT} is not an account")          # same digits, must survive
    t = d.add_table(rows=2, cols=2)
    t.cell(0, 0).text = "Contact"
    t.cell(0, 1).text = f"Email: {EMAIL}"
    t.cell(1, 0).text = "Reference"
    t.cell(1, 1).text = f"Card: {CARD}"
    sec = d.sections[0]
    sec.header.paragraphs[0].text = f"Header Name: {NAME}"
    sec.footer.paragraphs[0].text = "Footer ordinary text"
    d.add_paragraph("Closing paragraph, ordinary text.")
    if with_image:
        img = image_bytes(text_image([f"SSN: {SSN}", "Scanned note in picture"], size=(900, 300), step=80))
        d.add_picture(io.BytesIO(img))
        if shared_image:
            d.add_paragraph("Second use of the same picture:")
            d.add_picture(io.BytesIO(img))
    d.save(str(path))


# ------------------------------------------------------------------------------- PPTX
def make_pptx(path: str | Path, *, with_image: bool = False) -> None:
    from pptx import Presentation
    from pptx.util import Inches, Pt
    prs = Presentation()
    prs.core_properties.author = f"Owner {EMAIL}"
    s1 = prs.slides.add_slide(prs.slide_layouts[1])
    s1.shapes.title.text = KEEP_PHRASE
    body = s1.placeholders[1].text_frame
    body.text = "Plain first bullet"
    p2 = body.add_paragraph()
    r = p2.add_run()
    r.text = "SSN: 123-45"
    r.font.bold = True
    r2 = p2.add_run()
    r2.text = "-6789 trailing"
    r2.font.size = Pt(18)
    s1.notes_slide.notes_text_frame.text = f"Speaker note: email {EMAIL} before the call"
    s2 = prs.slides.add_slide(prs.slide_layouts[5])
    s2.shapes.title.text = "Table slide"
    tbl = s2.shapes.add_table(2, 2, Inches(1), Inches(2), Inches(6), Inches(1.5)).table
    tbl.cell(0, 0).text = "Phone"
    tbl.cell(0, 1).text = PHONE
    tbl.cell(1, 0).text = "Note"
    tbl.cell(1, 1).text = "Ordinary table text"
    if with_image:
        img = image_bytes(text_image([f"Account number: {ACCT}", "Slide image caption"], size=(900, 300), step=80))
        s2.shapes.add_picture(io.BytesIO(img), Inches(1), Inches(4), width=Inches(5))
    prs.save(str(path))


# ------------------------------------------------------------------------------- images
def make_image_file(path: str | Path, fmt="PNG", *, exif_gps: bool = False) -> None:
    img = text_image([KEEP_PHRASE, f"SSN: {SSN}", f"Email: {EMAIL}", "Ordinary caption line"])
    kw = {}
    if exif_gps:
        ex = Image.Exif()
        ex[0x010E] = "Camera description with secrets"
        ex[0x0112] = 1
        ex.get_ifd(0x8825)[1] = "N"
        ex.get_ifd(0x8825)[2] = (37.0, 46.0, 30.0)
        kw["exif"] = ex.tobytes()
    img.save(str(path), fmt, **kw)
