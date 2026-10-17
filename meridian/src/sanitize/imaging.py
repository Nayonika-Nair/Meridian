"""Image preparation, OCR and optional face detection with an explicit transform chain (A1, A6).

``prepare_for_ocr`` applies orientation / crop / rotation / deskew / scale to the image that OCR
will see and records every step in a TransformChain, so each OCR word box can be mapped back to
``original_px`` (the pixel grid of the stored image or of the PDF page render).

OCR uses Tesseract through pytesseract with a hard per-call timeout. A custom engine can be
injected (``engine(image, timeout) -> list[word dicts]``) for tests or alternative OCR back-ends.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Callable, Iterable

from PIL import Image, ImageOps

from .coords import (EXIF_TO_OP, Rect, Step, TransformChain, crop_step, rect_norm, rect_pad, rect_union,
                     rotate_step, scale_step, transpose_step)

try:                                       # OCR is optional at import time; the engine reports its absence
    import pytesseract
    from pytesseract import Output
except Exception:                          # pragma: no cover
    pytesseract = None
    Output = None


class OcrUnavailable(RuntimeError):
    pass


class OcrTimeout(RuntimeError):
    pass


def ocr_available() -> bool:
    if pytesseract is None:
        return False
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def open_image(source, max_pixels: int) -> Image.Image:
    """Open an image with a decompression-bomb guard; the pixel is fully loaded before returning."""
    prev = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = max_pixels * 2          # Pillow warns at 1x and errors at 2x
    try:
        img = Image.open(source)
    except Image.DecompressionBombError as exc:
        raise ValueError(f"image dimensions exceed the {max_pixels} pixel limit") from exc
    finally:
        Image.MAX_IMAGE_PIXELS = prev                # never leak a per-run limit into the host process
    w, h = img.size
    if w * h > max_pixels:
        img.close()
        raise ValueError(f"image dimensions {w}x{h} exceed the {max_pixels} pixel limit")
    img.load()
    return img


# --------------------------------------------------------------------------- preparation
@dataclass
class Prepared:
    image: Image.Image
    chain: TransformChain
    original_size: tuple[int, int]
    notes: list[str] = field(default_factory=list)

    def to_original(self, box: Rect) -> Rect:
        return self.chain.map_rect(box, back=True)

    def close(self) -> None:
        try:
            self.image.close()
        except Exception:
            pass


def estimate_skew(gray: Image.Image, max_angle: float = 6.0, step: float = 0.5) -> float:
    """Projection-profile skew estimate in degrees (positive = text runs counter-clockwise upward).

    Returns 0.0 when the image has too little ink for a stable estimate.
    """
    import numpy as np
    small = gray.copy()
    small.thumbnail((900, 900))
    a = np.asarray(small.convert("L"), dtype="uint8")
    ink = (a < 140).astype("float32")
    if ink.sum() < 200:
        return 0.0
    img = Image.fromarray((ink * 255).astype("uint8"))
    best, best_score = 0.0, -1.0
    ang = -max_angle
    while ang <= max_angle + 1e-9:
        r = np.asarray(img.rotate(ang, resample=Image.BILINEAR, fillcolor=0), dtype="float32")
        score = float(np.var(r.sum(axis=1)))
        if score > best_score:
            best, best_score = ang, score
        ang += step
    return -best if abs(best) >= step else 0.0


def prepare_for_ocr(img: Image.Image, *, origin_space: str = "original_px", exif_orient: int | None = None,
                    crop: Rect | None = None, rotate_deg: float = 0.0, deskew_deg: float | None = None,
                    auto_deskew: bool = False, min_long_side: int = 0, max_scale: float = 3.0) -> Prepared:
    """Build the image OCR will see plus the chain original -> processed.

    Order of operations (each recorded as a chain step): EXIF orientation, crop, 90-degree/free rotation,
    deskew, scale. Grey-scale / autocontrast change pixel values only (no geometry) and are not steps.
    """
    chain = TransformChain()
    cur = img.convert("RGB") if img.mode not in ("L", "RGB") else img.copy()
    space = origin_space
    notes: list[str] = []
    orig_size = img.size

    def nxt(name: str) -> str:
        return f"{name}_px"

    if exif_orient and exif_orient in EXIF_TO_OP:
        op = EXIF_TO_OP[exif_orient]
        st, new = transpose_step(op, cur.size, space, nxt("upright"))
        cur = cur.transpose(getattr(Image.Transpose, op))
        chain.add(st)
        space = st.dst
        notes.append(f"exif orientation {exif_orient} -> {op}")
    if crop:
        x0, y0, x1, y1 = [int(round(v)) for v in rect_norm(crop)]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(cur.size[0], x1), min(cur.size[1], y1)
        if x1 <= x0 or y1 <= y0:
            raise ValueError("empty crop")
        cur = cur.crop((x0, y0, x1, y1))
        st = crop_step(x0, y0, space, nxt("cropped"))
        chain.add(st)
        space = st.dst
    rot = rotate_deg % 360
    if rot:
        if rot in (90, 180, 270):
            op = {90: "ROTATE_90", 180: "ROTATE_180", 270: "ROTATE_270"}[int(rot)]
            st, new = transpose_step(op, cur.size, space, nxt("rotated"))
            cur = cur.transpose(getattr(Image.Transpose, op))
        else:
            st, new = rotate_step(rot, cur.size, space, nxt("rotated"), expand=True)
            cur = cur.rotate(rot, expand=True, resample=Image.BICUBIC, fillcolor=(255, 255, 255) if cur.mode == "RGB" else 255)
        chain.add(st)
        space = st.dst
    angle = deskew_deg
    if angle is None and auto_deskew:
        angle = estimate_skew(cur.convert("L"))
    if angle and abs(angle) >= 0.25:
        st, new = rotate_step(angle, cur.size, space, nxt("deskewed"), expand=True, name="deskew")
        cur = cur.rotate(angle, expand=True, resample=Image.BICUBIC, fillcolor=(255, 255, 255) if cur.mode == "RGB" else 255)
        chain.add(st)
        space = st.dst
        notes.append(f"deskew {angle:+.2f} deg")
    if min_long_side and max(cur.size) < min_long_side:
        k = min(max_scale, min_long_side / max(cur.size))
        nw, nh = max(1, int(round(cur.size[0] * k))), max(1, int(round(cur.size[1] * k)))
        st = scale_step(nw / cur.size[0], nh / cur.size[1], space, nxt("scaled"))
        cur = cur.resize((nw, nh), Image.LANCZOS)
        chain.add(st)
        space = st.dst
        notes.append(f"upscale x{k:.2f}")
    cur = ImageOps.autocontrast(cur.convert("L")) if cur.mode != "L" else ImageOps.autocontrast(cur)
    # normalise the final space name so callers can rely on it
    if chain.steps:
        last = chain.steps[-1]
        chain.steps[-1] = Step(last.name, last.src, "processed_px", last.m, last.params)
    else:
        chain.add(Step("identity", origin_space, "processed_px", (1, 0, 0, 0, 1, 0, 0, 0, 1)))
    return Prepared(cur, chain, orig_size, notes)


# --------------------------------------------------------------------------- OCR
@dataclass
class Word:
    text: str
    box: Rect                   # processed_px
    conf: float
    line_key: tuple
    box_original: Rect | None = None


def ocr_words(img: Image.Image, *, timeout: float = 45.0, engine: Callable | None = None,
              lang: str = "eng", psm: int = 6) -> list[Word]:
    """Run OCR and return word boxes in the coordinates of ``img``."""
    if engine is not None:
        raw = engine(img, timeout)
        return [Word(w["text"], tuple(w["box"]), float(w.get("conf", 90)), tuple(w.get("line", (0, 0, 0)))) for w in raw]
    if not ocr_available():
        raise OcrUnavailable("tesseract is not installed or not on PATH")
    try:
        d = pytesseract.image_to_data(img, lang=lang, config=f"--psm {psm}", output_type=Output.DICT, timeout=timeout)
    except RuntimeError as exc:                       # pytesseract raises RuntimeError('Tesseract process timeout')
        raise OcrTimeout(str(exc)) from exc
    words = []
    for i, t in enumerate(d["text"]):
        t = (t or "").strip()
        if not t:
            continue
        conf = float(d["conf"][i])
        if conf < 0:
            continue
        x, y, w, h = d["left"][i], d["top"][i], d["width"][i], d["height"][i]
        words.append(Word(t, (float(x), float(y), float(x + w), float(y + h)), conf,
                          (d["block_num"][i], d["par_num"][i], d["line_num"][i])))
    return words


def words_to_text(words: list[Word]) -> tuple[str, list[tuple[int, int, int]]]:
    """Join words into reading-order text (one line per OCR line). Returns text and a span map
    [(word_index, char_start, char_end)] so a regex match can be mapped back to word boxes."""
    order = sorted(range(len(words)), key=lambda i: (words[i].line_key, words[i].box[0]))
    out, spans, prev_line = [], [], None
    pos = 0
    for i in order:
        w = words[i]
        if prev_line is not None:
            sep = "\n" if w.line_key != prev_line else " "
            out.append(sep)
            pos += 1
        out.append(w.text)
        spans.append((i, pos, pos + len(w.text)))
        pos += len(w.text)
        prev_line = w.line_key
    return "".join(out), spans


def span_to_boxes(words: list[Word], spans, start: int, end: int, *, use_original: bool = False) -> list[Rect]:
    """Boxes covering text[start:end]; partially covered words are cut proportionally (per-line union)."""
    by_line: dict[tuple, list[Rect]] = {}
    for wi, s, e in spans:
        if e <= start or s >= end:
            continue
        w = words[wi]
        box = w.box_original if (use_original and w.box_original) else w.box
        n = max(1, e - s)
        a, b = max(start, s) - s, min(end, e) - s
        x0 = box[0] + (box[2] - box[0]) * a / n
        x1 = box[0] + (box[2] - box[0]) * b / n
        by_line.setdefault(w.line_key, []).append((x0, box[1], x1, box[3]))
    return [rect_union(v) for _k, v in sorted(by_line.items())]


def read_score(words: list[Word]) -> float:
    """How much readable text OCR found: confident words made of letters/digits."""
    return float(sum(1 for w in words if w.conf >= 55 and sum(c.isalnum() for c in w.text) >= 3))


def osd_rotation(img: Image.Image, timeout: float = 15.0) -> tuple[int, float] | None:
    """Tesseract orientation-and-script detection. Returns (degrees to rotate clockwise, confidence) or None."""
    if not ocr_available():
        return None
    try:
        d = pytesseract.image_to_osd(img, output_type=Output.DICT, timeout=timeout)
        return int(d.get("rotate", 0)) % 360, float(d.get("orientation_conf", 0.0))
    except Exception:
        return None


def ocr_oriented(img: Image.Image, *, origin_space: str = "original_px", exif_orient: int | None = None,
                 timeout: float = 45.0, engine: Callable | None = None, auto_deskew: bool = False,
                 min_long_side: int = 0, try_rotations: bool = True) -> tuple["Prepared", list[Word], int]:
    """OCR with automatic page-orientation recovery (sideways / upside-down scans).

    Returns (prepared image + chain, words in processed_px, rotation_applied_degrees). When the first
    pass reads little, Tesseract OSD proposes a rotation; both rotation directions are tried and the
    reading with the best score wins. The chosen rotation is part of the returned transform chain.
    """
    prep = prepare_for_ocr(img, origin_space=origin_space, exif_orient=exif_orient, auto_deskew=auto_deskew,
                           min_long_side=min_long_side)
    words = ocr_words(prep.image, timeout=timeout, engine=engine)
    score = read_score(words)
    if engine is not None or not try_rotations or score >= 4:
        return prep, words, 0
    hint = osd_rotation(prep.image, timeout=min(timeout, 20.0))
    if not hint or hint[0] == 0 or hint[1] < 1.0:
        return prep, words, 0
    best = (score, prep, words, 0)
    for rot in (hint[0], (360 - hint[0]) % 360):
        p2 = prepare_for_ocr(img, origin_space=origin_space, exif_orient=exif_orient, auto_deskew=auto_deskew,
                             min_long_side=min_long_side, rotate_deg=float(rot))
        w2 = ocr_words(p2.image, timeout=timeout, engine=engine)
        s2 = read_score(w2)
        if s2 > best[0] + 1:
            best[1].close()
            best = (s2, p2, w2, rot)
        else:
            p2.close()
    return best[1], best[2], best[3]


# --------------------------------------------------------------------------- faces (optional)
class FaceDetectorUnavailable(RuntimeError):
    pass


def face_backend_status() -> dict:
    """Reports which optional local face-detector back-end is usable. Detection only - no recognition."""
    try:
        import cv2                                    # noqa: F401
    except Exception:
        return {"available": False, "backend": None, "detail": "opencv-python is not installed (optional)"}
    import cv2
    cascade = os.environ.get("MERIDIAN_FACE_CASCADE")
    if cascade and os.path.exists(cascade) and hasattr(cv2, "CascadeClassifier"):
        return {"available": True, "backend": "haar-cascade", "detail": cascade}
    data = getattr(getattr(cv2, "data", None), "haarcascades", None)
    if data and hasattr(cv2, "CascadeClassifier") and os.path.exists(os.path.join(data, "haarcascade_frontalface_default.xml")):
        return {"available": True, "backend": "haar-cascade", "detail": "bundled cascade"}
    model = os.environ.get("MERIDIAN_FACE_MODEL")
    if model and os.path.exists(model) and hasattr(cv2, "FaceDetectorYN"):
        return {"available": True, "backend": "yunet-onnx", "detail": model}
    return {"available": False, "backend": None,
            "detail": "no cascade/model found: set MERIDIAN_FACE_CASCADE or MERIDIAN_FACE_MODEL (YuNet ONNX)"}


def detect_faces(img: Image.Image, detector: Callable | None = None) -> list[Rect]:
    """Face boxes in ``img`` pixels. ``detector(img) -> [(x0,y0,x1,y1)]`` overrides the built-in back-end.

    The built-in back-ends locate face regions only; this module never identifies or compares people.
    """
    if detector is not None:
        return [rect_norm(b) for b in detector(img)]
    st = face_backend_status()
    if not st["available"]:
        raise FaceDetectorUnavailable(st["detail"])
    import cv2
    import numpy as np
    arr = np.asarray(img.convert("RGB"))[:, :, ::-1].copy()
    if st["backend"] == "haar-cascade":
        path = os.environ.get("MERIDIAN_FACE_CASCADE") or os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        clf = cv2.CascadeClassifier(path)
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        found = clf.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        return [(float(x), float(y), float(x + w), float(y + h)) for x, y, w, h in found]
    det = cv2.FaceDetectorYN.create(os.environ["MERIDIAN_FACE_MODEL"], "", (arr.shape[1], arr.shape[0]))
    _ok, faces = det.detect(arr)
    return [] if faces is None else [(float(f[0]), float(f[1]), float(f[0] + f[2]), float(f[1] + f[3])) for f in faces]


# --------------------------------------------------------------------------- painting
def paint_regions(img: Image.Image, rects: Iterable[Rect], *, pad: float = 2.0, fill=None) -> Image.Image:
    """Return a copy of ``img`` with every rect (original pixel space) filled solid black."""
    from PIL import ImageDraw
    out = img.copy()
    if out.mode in ("P", "1", "LA", "PA"):
        out = out.convert("RGBA" if "A" in out.mode else "RGB")
    d = ImageDraw.Draw(out)
    w, h = out.size
    color = fill if fill is not None else ((0, 0, 0, 255) if out.mode == "RGBA" else (0, 0, 0) if out.mode in ("RGB", "CMYK") else 0)
    for r in rects:
        x0, y0, x1, y1 = rect_pad(rect_norm(r), pad, (0, 0, w, h))
        d.rectangle([math.floor(x0), math.floor(y0), math.ceil(x1) - 1, math.ceil(y1) - 1], fill=color)
    return out
