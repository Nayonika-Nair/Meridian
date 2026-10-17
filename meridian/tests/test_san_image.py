"""A6 - image text redaction, orientation/cropping mapping, optional face-region detection, EXIF handling."""
import io

import numpy as np
import pytest
from PIL import Image

import fixtures_gen as F
from src.sanitize import APPROVED, REJECTED, REVIEW, Options, Policy, process_document
from src.sanitize import imaging
from src.sanitize.coords import rect_overlap_fraction
from src.sanitize.imaging import ocr_available

needs_ocr = pytest.mark.skipif(not ocr_available(), reason="tesseract not installed")


def run(src, out, policy=None, options=None):
    return process_document(src, out, policy or Policy(), options or Options())


def check(res, name):
    return next(c for c in res["verification"]["checks"] if c["name"] == name)


@needs_ocr
@pytest.mark.parametrize("fmt,ext", [("PNG", "png"), ("JPEG", "jpg"), ("BMP", "bmp"), ("TIFF", "tif")])
def test_text_regions_are_painted_out_in_each_supported_format(tmp_path, fmt, ext):
    src, out = tmp_path / f"in.{ext}", tmp_path / f"out.{ext}"
    F.make_image_file(src, fmt)
    res = run(src, out)
    assert res["release"]["status"] == APPROVED, res["verification"]["checks"]
    cats = {r["category"] for r in res["regions"]}
    assert {"GOV_ID", "EMAIL"} <= cats
    a, b = np.asarray(Image.open(src).convert("RGB"), dtype=int), np.asarray(Image.open(out).convert("RGB"), dtype=int)
    assert a.shape == b.shape
    changed = np.abs(a - b).sum(axis=2) > 60
    for r in res["regions"]:
        x0, y0, x1, y1 = [int(v) for v in r["bbox"]]
        assert changed[y0:y1, x0:x1].mean() > 0.3                         # the region is really painted
    outside = changed.copy()
    for r in res["regions"]:
        x0, y0, x1, y1 = [int(v) for v in r["bbox"]]
        outside[max(0, y0 - 6):y1 + 6, max(0, x0 - 6):x1 + 6] = False
    assert outside.mean() < 0.005 or fmt == "JPEG"                         # nothing else changed (JPEG re-encode noise aside)
    assert check(res, "ocr_recheck")["status"] == "passed"


@needs_ocr
def test_orientation_exif_is_honoured_and_regions_are_in_stored_pixel_space(tmp_path):
    src, out = tmp_path / "in.jpg", tmp_path / "out.jpg"
    upright = F.text_image([F.KEEP_PHRASE, f"SSN: {F.SSN}", "Ordinary caption line"])
    stored = upright.transpose(Image.Transpose.ROTATE_90)                  # what a camera stores; EXIF says "rotate back"
    ex = Image.Exif()
    ex[0x0112] = 6
    stored.save(src, "JPEG", exif=ex.tobytes(), quality=95)
    res = run(src, out)
    assert res["release"]["status"] == APPROVED, res["verification"]["checks"]
    ssn = next(r for r in res["regions"] if r["category"] == "GOV_ID")
    x0, y0, x1, y1 = ssn["bbox"]
    assert (y1 - y0) > 2.5 * (x1 - x0)                                     # vertical strip: stored pixels are sideways
    assert ssn["space"] == "original_px" and any(s["step"].startswith("transpose") for s in ssn["chain"])
    assert Image.open(out).getexif().get(0x0112) == 6                      # orientation tag preserved
    arr = np.asarray(Image.open(out).convert("L"))
    assert arr[int(y0):int(y1), int(x0):int(x1)].mean() < 40               # painted dark where the SSN was


def test_gps_and_free_text_exif_are_removed_but_orientation_and_pixels_are_kept(tmp_path):
    src, out = tmp_path / "in.jpg", tmp_path / "out.jpg"
    F.make_image_file(src, "JPEG", exif_gps=True)
    res = run(src, out, Policy(ocr=False, require_complete_inspection=False))
    ex = Image.open(out).getexif()
    assert 0x8825 not in ex and 0x010E not in ex and ex.get(0x0112) == 1
    meta = next(p for p in res["payloads"] if p["payload"] == "metadata")
    assert meta["outcome"] == "sanitised" and "GPS" in meta["detail"] and "ImageDescription" in meta["detail"]
    assert check(res, "metadata_checked")["status"] == "passed"


def test_png_text_chunks_containing_sensitive_values_are_dropped(tmp_path):
    from PIL.PngImagePlugin import PngInfo
    src, out = tmp_path / "in.png", tmp_path / "out.png"
    info = PngInfo()
    info.add_text("Comment", f"owner {F.EMAIL}")
    info.add_text("Software", "synthetic generator")
    Image.new("RGB", (80, 40), "white").save(src, pnginfo=info)
    res = run(src, out, Policy(ocr=False, require_complete_inspection=False))
    t = Image.open(out).text
    assert "Comment" not in t and t.get("Software") == "synthetic generator"
    assert res["release"]["status"] in (APPROVED, REVIEW)


# ------------------------------------------------------------------------------ faces: optional, detection only
def _face_policy():
    return Policy(ocr=False, faces=True, require_complete_inspection=False, required_payloads=["face_regions"])


def test_face_regions_use_the_injected_local_detector_and_are_blanked(tmp_path):
    src, out = tmp_path / "photo.png", tmp_path / "photo_out.png"
    img = Image.new("RGB", (400, 300), (120, 160, 200))
    img.save(src)
    calls = []

    def detector(im):
        calls.append(im.size)
        return [(100, 60, 180, 160)]                                       # a located region, never an identity

    res = run(src, out, _face_policy(), Options(face_detector=detector))
    assert calls and res["release"]["status"] == APPROVED, res["verification"]["checks"]
    face = next(r for r in res["regions"] if r["rule_id"] == "FACE.REGION")
    assert face["category"] == "FACE" and face["source_method"] == "face_detector" and face["action"] == "REDACT"
    assert face["text_hash"] == "" and face["text_shape"] == "face"        # no identity, no template, nothing recognisable stored
    arr = np.asarray(Image.open(out).convert("RGB"))
    assert arr[100:140, 120:160].max() == 0                                  # blanked
    assert (arr[10:40, 10:40] == (120, 160, 200)).all()                      # background untouched
    assert rect_overlap_fraction((100, 60, 180, 160), tuple(face["bbox"])) > 0.99


def test_face_detection_is_off_by_default(tmp_path):
    src, out = tmp_path / "photo.png", tmp_path / "o.png"
    Image.new("RGB", (100, 100), "gray").save(src)
    res = run(src, out, Policy(ocr=False), Options(face_detector=lambda im: [(0, 0, 50, 50)]))
    assert not any(r["rule_id"] == "FACE.REGION" for r in res["regions"])


def test_face_detection_without_a_backend_is_reported_not_faked(tmp_path, monkeypatch):
    monkeypatch.setattr(imaging, "face_backend_status", lambda: {"available": False, "backend": None, "detail": "none installed"})
    src, out = tmp_path / "photo.png", tmp_path / "o.png"
    Image.new("RGB", (100, 100), "gray").save(src)
    res = run(src, out, Policy(ocr=False, faces=True))
    assert any(p["payload"] == "face_regions" and p["outcome"] == "unsupported" for p in res["payloads"])
    assert res["release"]["status"] == REVIEW                              # faces were requested but could not be inspected


def test_face_detector_never_exposes_identity_api():
    assert not any(hasattr(imaging, n) for n in ("recognise_face", "recognize_face", "identify", "face_embedding"))


def test_face_regions_map_through_crop_and_exif_chain(tmp_path):
    src, out = tmp_path / "p.jpg", tmp_path / "o.jpg"
    stored = Image.new("RGB", (300, 400), (90, 90, 90))                     # stored portrait-as-landscape
    ex = Image.Exif()
    ex[0x0112] = 6
    stored.save(src, "JPEG", exif=ex.tobytes())

    def detector(im):                                                       # sees the UPRIGHT image (400x300)
        assert im.size[0] >= im.size[1]
        return [(40, 30, 140, 110)]

    res = run(src, out, _face_policy(), Options(face_detector=detector))
    face = next(r for r in res["regions"] if r["rule_id"] == "FACE.REGION")
    x0, y0, x1, y1 = face["bbox"]
    # upright (x,y) -> stored for EXIF 6 (rotate CW 90): stored_x = y, stored_y = H_up - x  with H_up = 400
    assert (x0, y0, x1, y1) == pytest.approx((30, 400 - 140, 110, 400 - 40), abs=2.5)


# ------------------------------------------------------------------------------ grouping inside one image
@needs_ocr
def test_multiword_value_is_one_region_with_merged_line_box_and_not_the_label(tmp_path):
    src, out = tmp_path / "in.png", tmp_path / "out.png"
    F.text_image([F.KEEP_PHRASE, f"Name: {F.NAME}", f"Account number: {F.ACCT}"]).save(src)
    res = run(src, out)
    names = [r for r in res["regions"] if r["rule_id"] == "PERSON.NAME_LABELLED"]
    assert len(names) == 1 and len(names[0]["rects"]) == 1                 # "Jane Doe" -> one merged box on one line
    arr = np.asarray(Image.open(out).convert("L"))
    ref = np.asarray(Image.open(src).convert("L"))
    x0, y0, x1, y1 = [int(v) for v in names[0]["bbox"]]
    label_zone = ref[y0:y1, 60:x0 - 8]
    assert (arr[y0:y1, 60:x0 - 8] == label_zone).mean() > 0.99               # "Name:" label untouched


# ------------------------------------------------------------------------------ rejections
def test_animated_or_multiframe_images_are_rejected_not_half_processed(tmp_path):
    src = tmp_path / "anim.gif"
    frames = [Image.new("RGB", (40, 40), c) for c in ((255, 0, 0), (0, 255, 0), (0, 0, 255))]
    frames[0].save(src, save_all=True, append_images=frames[1:], duration=100, loop=0)
    res = run(src, tmp_path / "o.gif", Policy(ocr=False))
    assert res["release"]["status"] == REJECTED and res["errors"][0]["code"] == "unsupported_multiframe"


def test_svg_and_text_masquerading_as_images_are_unsupported(tmp_path):
    p = tmp_path / "x.png"
    p.write_text("<svg xmlns='http://www.w3.org/2000/svg'><text>SSN 123-45-6789</text></svg>")
    res = run(p, tmp_path / "o.png", Policy(ocr=False))
    assert res["release"]["status"] == REJECTED and res["errors"][0]["code"] == "unsupported_type"
