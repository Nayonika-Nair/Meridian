"""Image-region sanitisation (A6) used for stand-alone images and for images embedded in DOCX/PPTX.

Text regions come from OCR (mapped back to ``original_px`` - the pixel grid of the stored image,
so EXIF orientation, cropping, rotation, deskewing and scaling are all handled by the chain in
``imaging.prepare_for_ocr``). Face regions come from an optional local detector (location only).
"""
from __future__ import annotations

import io
import time
from dataclasses import dataclass, field

from PIL import Image, ImageOps

from .context import RunContext
from .coords import Rect, rect_pad
from .detect import finding_to_region, scan_text
from .imaging import (FaceDetectorUnavailable, OcrTimeout, OcrUnavailable, detect_faces, ocr_oriented, open_image,
                      paint_regions, prepare_for_ocr, span_to_boxes, words_to_text)
from .regions import Region, group_regions, make_region_id, text_shape, keyed_hash
from .safety import InputRejected

SPACE = "original_px"
EXIF_ORIENTATION = 0x0112
SENSITIVE_EXIF = {0x8825: "GPS", 0x927C: "MakerNote", 0x9286: "UserComment", 0x010E: "ImageDescription",
                  0x013B: "Artist", 0x8298: "Copyright", 0xA430: "CameraOwnerName", 0xA431: "BodySerialNumber",
                  0xA435: "LensSerialNumber", 0x013C: "HostComputer", 0x9C9B: "XPTitle", 0x9C9C: "XPComment",
                  0x9C9D: "XPAuthor", 0x9C9E: "XPKeywords", 0x9C9F: "XPSubject"}
SAVE_FORMAT = {"png": "PNG", "jpeg": "JPEG", "tiff": "TIFF", "bmp": "BMP", "gif": "GIF", "webp": "WEBP"}


@dataclass
class ImageScan:
    size: tuple[int, int]
    fmt: str
    orientation: int | None
    regions: list[Region] = field(default_factory=list)
    exif_flags: list[str] = field(default_factory=list)
    text_chunk_flags: list[str] = field(default_factory=list)
    frames: int = 1
    ocr_status: str = "skipped"          # skipped | done | failed | unavailable


def _exif_orientation(img: Image.Image) -> int | None:
    try:
        return img.getexif().get(EXIF_ORIENTATION)
    except Exception:
        return None


def inspect_image(rc: RunContext, img: Image.Image, *, unit_id: str, space_label: str = "image") -> ImageScan:
    """OCR + optional face detection on one loaded image. Raises nothing for OCR problems (reports them)."""
    fmt = (img.format or "PNG").lower().replace("mpo", "jpeg")
    orient = _exif_orientation(img)
    scan = ImageScan(size=img.size, fmt=fmt, orientation=orient, frames=getattr(img, "n_frames", 1))
    try:
        for tag in img.getexif().keys():
            if tag in SENSITIVE_EXIF:
                scan.exif_flags.append(SENSITIVE_EXIF[tag])
        if 0x8825 in img.getexif() and img.getexif().get_ifd(0x8825):
            scan.exif_flags.append("GPS")
    except Exception:
        pass
    scan.exif_flags = sorted(set(scan.exif_flags))
    for k, v in (img.info or {}).items():
        if isinstance(v, str) and k.lower() not in ("dpi", "icc_profile", "transparency"):
            if scan_text(rc.detect, v, "image"):
                scan.text_chunk_flags.append(k)
    rc.counters["images_total"] += 0                       # callers account for totals
    if not rc.policy.ocr and not rc.policy.faces:
        return scan
    prep = None
    try:
        if rc.policy.ocr:
            if not rc.ocr_ready():
                scan.ocr_status = "unavailable"
            elif rc.ocr_budget_left() <= 0.5:
                scan.ocr_status = "failed"
                rc.counters["ocr_skipped_budget"] += 1
            else:
                t0 = time.monotonic()
                words = []
                try:
                    prep, words, _rot = ocr_oriented(img, exif_orient=orient, min_long_side=1400,
                                                     timeout=min(rc.limits.ocr_timeout_per_image_s, rc.ocr_budget_left()),
                                                     engine=rc.options.ocr_engine)
                    scan.ocr_status = "done"
                    rc.counters["images_ocr"] += 1
                except (OcrTimeout, OcrUnavailable):
                    scan.ocr_status = "failed"
                    rc.counters["images_ocr_failed"] += 1
                finally:
                    rc.charge_ocr(t0)
                if prep is not None:
                    for w in words:
                        w.box_original = prep.to_original(w.box)
                    text, spans = words_to_text(words)
                    for f in scan_text(rc.detect, text, "image"):
                        rects = span_to_boxes(words, spans, f.start, f.end, use_original=True)
                        if rects:
                            scan.regions.append(finding_to_region(
                                rc.detect, f, unit_id=unit_id, space=SPACE, rects=rects, source_method="ocr",
                                anchor={"image": space_label}, chain=prep.chain.describe(), page_size=img.size))
        if rc.policy.faces:
            rule = rc.detect.registry.get("FACE.REGION")
            if prep is None:
                prep = prepare_for_ocr(img, exif_orient=orient, min_long_side=0)
            try:
                boxes = detect_faces(prep.image, rc.options.face_detector)
                for k, b in enumerate(boxes):
                    rect = prep.to_original(b)
                    scan.regions.append(Region(
                        region_id=make_region_id(rc.doc_id, unit_id, rule.id, f"face-{k}-{round(rect[0])}-{round(rect[1])}"),
                        document_id=rc.doc_id, unit_id=unit_id, space=SPACE, rects=[rect], anchor={"image": space_label},
                        category=rule.category, rule_id=rule.id, entity_type=rule.entity_type, severity=rule.severity,
                        confidence=rule.base_confidence, source_method="face_detector", action="REDACT",
                        reason="face region (detection only, no identification)", chain=prep.chain.describe(),
                        page_size=img.size, text_shape="face"))
                rc.payloads.add("face_regions", "sanitised" if boxes else "inspected", len(boxes))
            except FaceDetectorUnavailable as exc:
                rc.payloads.add("face_regions", "unsupported", 1, f"face detector unavailable: {exc}")
    finally:
        if prep is not None:
            prep.close()
    scan.regions = group_regions(scan.regions)
    return scan


# --------------------------------------------------------------------------- applying
def scrub_exif_bytes(img: Image.Image, keep_orientation: bool = True) -> bytes | None:
    """EXIF payload with sensitive tags removed (GPS, maker notes, free text, serials). Keeps orientation."""
    try:
        ex = img.getexif()
    except Exception:
        return None
    if not len(ex):
        return None
    for tag in list(ex.keys()):
        if tag in SENSITIVE_EXIF:
            del ex[tag]
    if 0x8769 in ex:                                   # Exif sub-IFD: drop sensitive entries inside it
        try:
            sub = ex.get_ifd(0x8769)
            for t in list(sub.keys()):
                if t in SENSITIVE_EXIF:
                    del sub[t]
        except Exception:
            pass
    return ex.tobytes() if len(ex) else None


def render_sanitised(rc: RunContext, img: Image.Image, regions: list[Region], fmt: str, *, pad: float = 2.0) -> bytes:
    """Paint acting regions in original pixel space and re-encode in the same format with scrubbed metadata."""
    rects = []
    for r in regions:
        if r.action in ("REDACT", "MASK", "GENERALISE"):          # raster images only support full blanking
            rects.extend(r.rects)
            if r.action != "REDACT":
                r.action = "REDACT"
                r.reason += "; raster images are redacted rather than masked"
    out = paint_regions(img, rects, pad=pad) if rects else img.copy()
    fmt_pil = SAVE_FORMAT.get(fmt, "PNG")
    kwargs = {}
    exif = scrub_exif_bytes(img)
    if fmt_pil == "JPEG":
        if out.mode not in ("RGB", "L", "CMYK"):
            out = out.convert("RGB")
        kwargs = {"quality": 92, "optimize": True}
        if exif:
            kwargs["exif"] = exif
    elif fmt_pil in ("PNG", "WEBP", "TIFF") and exif:
        kwargs["exif"] = exif
    if fmt_pil == "PNG":
        from PIL.PngImagePlugin import PngInfo
        info = PngInfo()
        for k, v in (img.info or {}).items():
            if isinstance(v, str) and k not in ("exif",) and not scan_text(rc.detect, v, "image"):
                info.add_text(k, v)
        kwargs["pnginfo"] = info
    buf = io.BytesIO()
    out.save(buf, format=fmt_pil, **kwargs)
    return buf.getvalue()


# --------------------------------------------------------------------------- stand-alone image files
def scan_image_file(rc: RunContext, path: str) -> tuple[Image.Image, ImageScan]:
    try:
        img = open_image(path, rc.limits.max_image_pixels)
    except ValueError as exc:
        raise InputRejected("image_too_large", str(exc)) from exc
    except Exception as exc:
        raise InputRejected("malformed", f"image could not be decoded: {exc}") from exc
    if getattr(img, "n_frames", 1) > 1:
        img.close()
        raise InputRejected("unsupported_multiframe", "multi-frame / animated images are not supported (other frames would be left unchecked)")
    rc.counters["units_total"] += 1
    rc.counters["images_total"] += 1
    scan = inspect_image(rc, img, unit_id="image")
    rc.counters["units_inspected"] += 1
    rc.payloads.add("body_text", "sanitised" if scan.regions else "inspected", len([r for r in scan.regions if r.source_method == "ocr"]),
                    {"done": "", "skipped": "OCR disabled", "failed": "OCR failed or timed out", "unavailable": "OCR engine unavailable"}[scan.ocr_status])
    if scan.ocr_status in ("failed", "unavailable") or (scan.ocr_status == "skipped" and not rc.policy.ocr):
        rc.payloads.add("body_text", "unsupported" if scan.ocr_status != "failed" else "failed", 1,
                        f"image text could not be inspected ({scan.ocr_status})")
    if scan.exif_flags:
        rc.payloads.add("metadata", "sanitised", len(scan.exif_flags), "EXIF removed: " + ", ".join(scan.exif_flags))
    else:
        rc.payloads.add("metadata", "inspected", 0, "no sensitive EXIF tags")
    if scan.text_chunk_flags:
        rc.payloads.add("metadata", "sanitised", len(scan.text_chunk_flags), "text chunks removed: " + ", ".join(scan.text_chunk_flags))
    rc.regions.extend(scan.regions)
    return img, scan


def apply_image_file(rc: RunContext, img: Image.Image, scan: ImageScan, out_path: str) -> None:
    data = render_sanitised(rc, img, scan.regions, scan.fmt)
    with open(out_path, "wb") as fh:
        fh.write(data)
