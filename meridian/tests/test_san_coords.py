"""A1 - explicit coordinate transformation chains, with synthetic documents whose sensitive text sits at known places."""
import math
import re

import numpy as np
import pytest
from PIL import Image, ImageDraw

import fixtures_gen as F
from src.sanitize.coords import (IDENTITY, Step, TransformChain, crop_step, from_normalised, mat_inv, mat_mul,
                                 pdf_chain, pdf_page_chain, pil_rotate_matrix, rect_iou, rect_overlap_fraction,
                                 rect_pad, rect_union, render_to_points_chain, scale_step, to_normalised,
                                 transpose_step)
from src.sanitize.imaging import (ocr_available, ocr_words, prepare_for_ocr, span_to_boxes, words_to_text)

needs_ocr = pytest.mark.skipif(not ocr_available(), reason="tesseract not installed")


def red_box_bounds(img):
    a = np.asarray(img.convert("RGB")).astype(int)
    m = (a[:, :, 0] > 200) & (a[:, :, 1] < 90) & (a[:, :, 2] < 90)
    ys, xs = np.where(m)
    return (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)


def marker_image():
    img = Image.new("RGB", (800, 500), "white")
    ImageDraw.Draw(img).rectangle([600, 50, 699, 89], fill=(255, 0, 0))
    return img, (600, 50, 700, 90)


# ------------------------------------------------------------------------------ chain mechanics
def test_chain_rejects_discontinuous_steps():
    ch = TransformChain([Step("a", "x", "y", IDENTITY)])
    with pytest.raises(ValueError):
        ch.add(Step("b", "z", "w", IDENTITY))


def test_matrix_inverse_roundtrip():
    m = mat_mul((2, 0, 5, 0, 3, -7, 0, 0, 1), (0, -1, 10, 1, 0, 0, 0, 0, 1))
    i = mat_inv(m)
    p = TransformChain([Step("m", "a", "b", m)]).forward((12.5, 40.0))
    q = TransformChain([Step("i", "b", "a", i)]).forward(p)
    assert q == pytest.approx((12.5, 40.0))


def test_chain_describes_every_step_in_order():
    img, _ = marker_image()
    p = prepare_for_ocr(img, crop=(10, 20, 790, 480), rotate_deg=90, deskew_deg=2.0, min_long_side=1800)
    names = [s["step"] for s in p.chain.describe()]
    assert names[0] == "crop" and names[1].startswith("transpose") and "deskew" in names and names[-1] == "scale"
    assert p.chain.origin == "original_px" and p.chain.target == "processed_px"


def test_normalised_coordinates_roundtrip():
    r = (100.0, 50.0, 300.0, 150.0)
    n = to_normalised(r, 800, 500)
    assert n == pytest.approx((0.125, 0.1, 0.375, 0.3))
    assert from_normalised(n, 800, 500) == pytest.approx(r)


# ------------------------------------------------------------------------------ pixel transforms vs the real image ops
@pytest.mark.parametrize("kw", [
    dict(crop=(50, 20, 750, 480)),
    dict(rotate_deg=90), dict(rotate_deg=180), dict(rotate_deg=270),
    dict(deskew_deg=3.0), dict(deskew_deg=-4.5), dict(rotate_deg=33.0),
    dict(min_long_side=1800),
    dict(crop=(40, 30, 760, 470), rotate_deg=270, deskew_deg=2.0, min_long_side=1500),
], ids=lambda k: "+".join(k))
def test_pixel_chain_matches_actual_pixels(kw):
    """Apply the operations to a coloured marker image and compare the predicted box with the real one."""
    img, box = marker_image()
    prep = prepare_for_ocr(img.copy(), **kw)
    # the processed image is greyscale/autocontrast; recompute the same geometry on the colour image
    colour = img.copy()
    chain_img = _replay(colour, kw)
    got = red_box_bounds(chain_img)
    pred = prep.chain.map_rect(box)
    assert abs(got[0] - pred[0]) <= 2 and abs(got[1] - pred[1]) <= 2
    assert abs(got[2] - pred[2]) <= 2 and abs(got[3] - pred[3]) <= 2
    back = prep.to_original(got)                      # inverse mapping lands on the original box
    assert rect_iou(back, box) > 0.9 if "deskew_deg" not in kw and "rotate_deg" not in kw or kw.get("rotate_deg") in (90, 180, 270) and "deskew_deg" not in kw else rect_overlap_fraction(box, back) > 0.95


def _replay(img, kw):
    cur = img
    if kw.get("crop"):
        x0, y0, x1, y1 = kw["crop"]
        cur = cur.crop((x0, y0, x1, y1))
    rot = kw.get("rotate_deg", 0) % 360
    if rot in (90, 180, 270):
        cur = cur.transpose({90: Image.Transpose.ROTATE_90, 180: Image.Transpose.ROTATE_180, 270: Image.Transpose.ROTATE_270}[rot])
    elif rot:
        cur = cur.rotate(rot, expand=True, resample=Image.BICUBIC, fillcolor=(255, 255, 255))
    if kw.get("deskew_deg"):
        cur = cur.rotate(kw["deskew_deg"], expand=True, resample=Image.BICUBIC, fillcolor=(255, 255, 255))
    if kw.get("min_long_side") and max(cur.size) < kw["min_long_side"]:
        k = min(3.0, kw["min_long_side"] / max(cur.size))
        cur = cur.resize((int(round(cur.size[0] * k)), int(round(cur.size[1] * k))), Image.LANCZOS)
    return cur


def test_pil_rotate_matrix_matches_pillow_size():
    for ang in (3, 33, 90, 200, 271):
        _, (nw, nh) = pil_rotate_matrix(ang, (800, 500), True)
        assert Image.new("RGB", (800, 500)).rotate(ang, expand=True).size == (nw, nh)


def test_exif_orientation_step_matches_pillow():
    for exif_val, op in ((6, "ROTATE_270"), (8, "ROTATE_90"), (3, "ROTATE_180"), (2, "FLIP_LEFT_RIGHT")):
        img, box = marker_image()
        prep = prepare_for_ocr(img, exif_orient=exif_val)
        want = red_box_bounds(img.transpose(getattr(Image.Transpose, op)))
        got = prep.chain.map_rect(box)
        assert all(abs(a - b) <= 1 for a, b in zip(want, got)), (exif_val, want, got)


# ------------------------------------------------------------------------------ PDF points vs pixels
def test_pdf_chain_matches_pymupdf_rotation_matrix():
    import pymupdf
    for rot in (0, 90, 180, 270):
        doc = pymupdf.open()
        p = doc.new_page(width=595, height=842)
        p.set_rotation(rot)
        ch = pdf_page_chain(595, 842, rot)
        for pt in ((10, 20), (300, 400), (595, 842)):
            expected = pymupdf.Point(*pt) * p.rotation_matrix
            got = ch.forward(pt)
            assert got == pytest.approx((expected.x, expected.y), abs=1e-6), (rot, pt)
        doc.close()


def test_pdf_user_space_is_bottom_left_and_pixels_scale_by_dpi():
    ch = pdf_chain(cropbox=(0, 0, 612, 792), rotation=0, dpi=144)
    # PDF user (72, 72) is one inch from the LEFT and BOTTOM edge -> 144 px right, 792-72 pts = 720 pt -> 1440 px down
    assert ch.forward((72, 72)) == pytest.approx((144.0, 1440.0))
    # pixel coordinates are not points: the same box differs by dpi/72
    assert ch.target == "render_px" and render_to_points_chain(144).forward((144, 1440)) == pytest.approx((72, 720))


def test_cropbox_offset_is_applied():
    ch = pdf_chain(cropbox=(50, 100, 650, 900), rotation=0)
    assert ch.forward((50, 900)) == pytest.approx((0, 0))          # top-left of the visible box


# ------------------------------------------------------------------------------ OCR boxes land on the known text
@needs_ocr
@pytest.mark.parametrize("kw", [
    dict(), dict(crop=(30, 100, 960, 520)), dict(min_long_side=2000), dict(deskew_deg=2.0),
    dict(crop=(20, 80, 980, 540), deskew_deg=-2.5, min_long_side=1900),
], ids=lambda k: "+".join(k) or "identity")
def test_ocr_box_maps_back_to_known_text_location(kw):
    lines = [F.KEEP_PHRASE, f"SSN: {F.SSN}", "Ordinary caption line"]
    img = F.text_image(lines)
    d = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    f = F.font(34)
    x_ssn = 60 + d.textlength("SSN: ", font=f)
    expected = d.textbbox((x_ssn, 60 + 70), F.SSN, font=f)            # tight ink box of the number, in original px
    prep = prepare_for_ocr(img, **kw)
    words = ocr_words(prep.image, timeout=40)
    text, spans = words_to_text(words)
    m = re.search(r"\d{3}-\d{2}-\d{4}", text)
    assert m, f"OCR did not read the SSN: {text!r}"
    boxes = span_to_boxes(words, spans, m.start(), m.end())
    assert len(boxes) == 1
    mapped = prep.to_original(boxes[0])
    assert rect_iou(expected, mapped) > 0.5, (expected, mapped)
    assert rect_overlap_fraction(expected, rect_pad(mapped, 4)) > 0.9                # the real text is covered
    assert mapped[0] > x_ssn - 12                                                    # the "SSN:" label is not claimed


def test_rect_helpers():
    a, b = (0, 0, 10, 10), (5, 0, 15, 10)
    assert rect_iou(a, b) == pytest.approx(1 / 3)
    assert rect_union([a, b]) == (0, 0, 15, 10)
    assert rect_overlap_fraction(a, b) == pytest.approx(0.5)
    assert rect_pad(a, 2, (0, 0, 11, 11)) == (0, 0, 11, 11)
