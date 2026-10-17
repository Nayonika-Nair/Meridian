"""Explicit coordinate-transformation chains (A1).

Pixel coordinates and PDF coordinates are NOT interchangeable. Every mapping here is an
ordered chain of named affine steps between named coordinate spaces, so a bounding box
reported by OCR on a processed image can be carried back, step by step, to:

    processed_px  ->  (scale)  ->  (deskew)  ->  (rotate)  ->  (crop)  ->  original_px
    original_px (page render)  ->  pdf_points (top-left, rotated page)
    pdf_points                 ->  pdf_user  (bottom-left origin, unrotated, MediaBox/CropBox offsets)

Spaces used
-----------
original_px     pixels of the image as stored (or of the page render at ``dpi``)
processed_px    pixels of the image handed to OCR after crop/rotate/deskew/scale
render_px       pixels of a rendered PDF page (== original_px for scanned page images)
pdf_points      PyMuPDF page space: origin top-left, y down, 1/72 inch, AFTER /Rotate
pdf_unrot_tl    unrotated page, origin top-left (intermediate)
pdf_user        PDF native user space: origin bottom-left, y up, unrotated
normalised      0..1 relative to a stated width/height

All matrices are 3x3 homogeneous, applied to column vectors (x, y, 1). Continuous coordinates:
pixel ``i`` covers the interval [i, i+1).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

Rect = tuple[float, float, float, float]
Point = tuple[float, float]
Matrix = tuple[float, float, float, float, float, float, float, float, float]

IDENTITY: Matrix = (1, 0, 0, 0, 1, 0, 0, 0, 1)


def mat_mul(a: Matrix, b: Matrix) -> Matrix:
    """a @ b (apply b first, then a)."""
    return tuple(sum(a[r * 3 + k] * b[k * 3 + c] for k in range(3)) for r in range(3) for c in range(3))  # type: ignore


def mat_inv(m: Matrix) -> Matrix:
    a, b, c, d, e, f, g, h, i = m
    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if abs(det) < 1e-12:
        raise ValueError("singular transform")
    inv = (
        (e * i - f * h), (c * h - b * i), (b * f - c * e),
        (f * g - d * i), (a * i - c * g), (c * d - a * f),
        (d * h - e * g), (b * g - a * h), (a * e - b * d),
    )
    return tuple(v / det for v in inv)  # type: ignore


def apply(m: Matrix, p: Point) -> Point:
    x, y = p
    w = m[6] * x + m[7] * y + m[8]
    return ((m[0] * x + m[1] * y + m[2]) / w, (m[3] * x + m[4] * y + m[5]) / w)


def translate(tx: float, ty: float) -> Matrix:
    return (1, 0, tx, 0, 1, ty, 0, 0, 1)


def scale(sx: float, sy: float | None = None) -> Matrix:
    return (sx, 0, 0, 0, sy if sy is not None else sx, 0, 0, 0, 1)


# --------------------------------------------------------------------------- rect helpers
def rect_norm(r: Sequence[float]) -> Rect:
    x0, y0, x1, y1 = r
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def rect_area(r: Rect) -> float:
    return max(0.0, r[2] - r[0]) * max(0.0, r[3] - r[1])


def rect_intersection(a: Rect, b: Rect) -> Rect | None:
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def rect_iou(a: Rect, b: Rect) -> float:
    i = rect_intersection(a, b)
    if not i:
        return 0.0
    ia = rect_area(i)
    return ia / (rect_area(a) + rect_area(b) - ia)


def rect_overlap_fraction(inner: Rect, outer: Rect) -> float:
    """Fraction of ``inner`` that lies inside ``outer`` (1.0 = fully covered)."""
    i = rect_intersection(inner, outer)
    return rect_area(i) / rect_area(inner) if i and rect_area(inner) else 0.0


def rect_union(rects: Iterable[Rect]) -> Rect:
    rs = list(rects)
    return (min(r[0] for r in rs), min(r[1] for r in rs), max(r[2] for r in rs), max(r[3] for r in rs))


def rect_pad(r: Rect, pad: float, bounds: Rect | None = None) -> Rect:
    out = (r[0] - pad, r[1] - pad, r[2] + pad, r[3] + pad)
    if bounds:
        out = (max(out[0], bounds[0]), max(out[1], bounds[1]), min(out[2], bounds[2]), min(out[3], bounds[3]))
    return out


def to_normalised(r: Rect, width: float, height: float) -> Rect:
    return (r[0] / width, r[1] / height, r[2] / width, r[3] / height)


def from_normalised(r: Rect, width: float, height: float) -> Rect:
    return (r[0] * width, r[1] * height, r[2] * width, r[3] * height)


# --------------------------------------------------------------------------- chain
@dataclass(frozen=True)
class Step:
    """One affine step from ``src`` space to ``dst`` space."""
    name: str
    src: str
    dst: str
    m: Matrix
    params: tuple = ()

    def inverse(self) -> "Step":
        return Step(self.name + "^-1", self.dst, self.src, mat_inv(self.m), self.params)


@dataclass
class TransformChain:
    """Ordered steps taking points from ``origin`` space to ``target`` space."""
    steps: list[Step] = field(default_factory=list)

    @property
    def origin(self) -> str | None:
        return self.steps[0].src if self.steps else None

    @property
    def target(self) -> str | None:
        return self.steps[-1].dst if self.steps else None

    def add(self, step: Step) -> "TransformChain":
        if self.steps and self.steps[-1].dst != step.src:
            raise ValueError(f"chain break: {self.steps[-1].dst!r} -> {step.src!r} ({step.name})")
        self.steps.append(step)
        return self

    def matrix(self) -> Matrix:
        m = IDENTITY
        for s in self.steps:
            m = mat_mul(s.m, m)
        return m

    def forward(self, p: Point) -> Point:
        return apply(self.matrix(), p)

    def inverse(self) -> "TransformChain":
        return TransformChain([s.inverse() for s in reversed(self.steps)])

    def back(self, p: Point) -> Point:
        """Map a point from the chain's target space back to its origin space."""
        return apply(mat_inv(self.matrix()), p)

    def map_rect(self, r: Rect, *, back: bool = False) -> Rect:
        """Axis-aligned bounds of the transformed rectangle corners."""
        m = mat_inv(self.matrix()) if back else self.matrix()
        pts = [apply(m, c) for c in ((r[0], r[1]), (r[2], r[1]), (r[2], r[3]), (r[0], r[3]))]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        return (min(xs), min(ys), max(xs), max(ys))

    def polygon(self, r: Rect, *, back: bool = False) -> list[Point]:
        m = mat_inv(self.matrix()) if back else self.matrix()
        return [apply(m, c) for c in ((r[0], r[1]), (r[2], r[1]), (r[2], r[3]), (r[0], r[3]))]

    def extend(self, other: "TransformChain") -> "TransformChain":
        out = TransformChain(list(self.steps))
        for s in other.steps:
            out.add(s)
        return out

    def describe(self) -> list[dict]:
        return [{"step": s.name, "from": s.src, "to": s.dst, "params": list(s.params)} for s in self.steps]


# --------------------------------------------------------------------------- image steps
def crop_step(x0: float, y0: float, src: str = "original_px", dst: str = "cropped_px") -> Step:
    return Step("crop", src, dst, translate(-x0, -y0), (x0, y0))


def scale_step(sx: float, sy: float, src: str, dst: str) -> Step:
    return Step("scale", src, dst, scale(sx, sy), (sx, sy))


def pil_rotate_matrix(angle_deg: float, size: tuple[int, int], expand: bool = True) -> tuple[Matrix, tuple[int, int]]:
    """Forward matrix (input px -> output px) and output size of ``PIL.Image.rotate(angle, expand)``.

    Mirrors Pillow's own algorithm (counter-clockwise visual rotation about the centre) so a
    box found on the rotated image maps back exactly.
    """
    w, h = size
    angle = -math.radians(angle_deg % 360)
    c, s = round(math.cos(angle), 15), round(math.sin(angle), 15)
    m = [c, s, 0.0, -s, c, 0.0]          # output -> input (Pillow convention)

    def tr(x, y, mm):
        return mm[0] * x + mm[1] * y + mm[2], mm[3] * x + mm[4] * y + mm[5]

    cx, cy = w / 2.0, h / 2.0
    m[2], m[5] = tr(-cx, -cy, m)
    m[2] += cx
    m[5] += cy
    nw, nh = w, h
    if expand:
        xs, ys = [], []
        for x, y in ((0, 0), (w, 0), (w, h), (0, h)):
            x, y = tr(x, y, m)
            xs.append(x)
            ys.append(y)
        nw = math.ceil(max(xs)) - math.floor(min(xs))
        nh = math.ceil(max(ys)) - math.floor(min(ys))
        m[2], m[5] = tr(-(nw - w) / 2.0, -(nh - h) / 2.0, m)
    out_to_in: Matrix = (m[0], m[1], m[2], m[3], m[4], m[5], 0, 0, 1)
    return mat_inv(out_to_in), (int(nw), int(nh))


def rotate_step(angle_deg: float, size: tuple[int, int], src: str, dst: str, expand: bool = True,
                name: str = "rotate") -> tuple[Step, tuple[int, int]]:
    m, new = pil_rotate_matrix(angle_deg, size, expand)
    return Step(name, src, dst, m, (angle_deg, size, expand)), new


def transpose_step(op: str, size: tuple[int, int], src: str, dst: str) -> tuple[Step, tuple[int, int]]:
    """Exact 90-degree style ops matching PIL.Image.transpose names (ROTATE_90 = CCW)."""
    w, h = size
    ops = {
        "FLIP_LEFT_RIGHT": ((-1, 0, w, 0, 1, 0, 0, 0, 1), (w, h)),
        "FLIP_TOP_BOTTOM": ((1, 0, 0, 0, -1, h, 0, 0, 1), (w, h)),
        "ROTATE_90": ((0, 1, 0, -1, 0, w, 0, 0, 1), (h, w)),           # CCW
        "ROTATE_180": ((-1, 0, w, 0, -1, h, 0, 0, 1), (w, h)),
        "ROTATE_270": ((0, -1, h, 1, 0, 0, 0, 0, 1), (h, w)),          # CW
        "TRANSPOSE": ((0, 1, 0, 1, 0, 0, 0, 0, 1), (h, w)),
        "TRANSVERSE": ((0, -1, h, -1, 0, w, 0, 0, 1), (h, w)),
    }
    m, new = ops[op]
    return Step("transpose:" + op, src, dst, m, (op, size)), new


# EXIF orientation value -> PIL transpose op that makes the stored image upright
EXIF_TO_OP = {2: "FLIP_LEFT_RIGHT", 3: "ROTATE_180", 4: "FLIP_TOP_BOTTOM", 5: "TRANSPOSE",
              6: "ROTATE_270", 7: "TRANSVERSE", 8: "ROTATE_90"}


# --------------------------------------------------------------------------- PDF steps
def pdf_chain(cropbox: Rect, rotation: int, dpi: float | None = None,
              *, include_render: bool = True) -> TransformChain:
    """Chain pdf_user -> pdf_unrot_tl -> pdf_points [-> render_px].

    ``cropbox`` is the visible box in PDF user space (origin bottom-left): (x0, y0, x1, y1).
    ``rotation`` is the page /Rotate value (clockwise degrees: 0, 90, 180, 270).
    """
    x0, y0, x1, y1 = rect_norm(cropbox)
    w, h = x1 - x0, y1 - y0
    ch = TransformChain()
    # PDF native -> top-left origin, relative to the crop box
    ch.add(Step("pdf_user_to_topleft", "pdf_user", "pdf_unrot_tl", (1, 0, -x0, 0, -1, y1, 0, 0, 1), (x0, y1)))
    rot = rotation % 360
    if rot == 0:
        m = IDENTITY
    elif rot == 90:
        m = (0, -1, h, 1, 0, 0, 0, 0, 1)       # x' = h - y ; y' = x
    elif rot == 180:
        m = (-1, 0, w, 0, -1, h, 0, 0, 1)
    elif rot == 270:
        m = (0, 1, 0, -1, 0, w, 0, 0, 1)       # x' = y ; y' = w - x
    else:
        raise ValueError(f"unsupported /Rotate {rotation}")
    ch.add(Step(f"page_rotate_{rot}", "pdf_unrot_tl", "pdf_points", m, (rot, w, h)))
    if include_render and dpi:
        k = dpi / 72.0
        ch.add(Step("render", "pdf_points", "render_px", scale(k), (dpi,)))
    return ch


def pdf_page_chain(width: float, height: float, rotation: int, dpi: float | None = None) -> TransformChain:
    """Chain starting at ``pdf_unrot_tl`` (PyMuPDF's text/annotation space: unrotated, origin top-left).

    ``width``/``height`` are the UNROTATED page size in points. Matches ``page.rotation_matrix``
    followed by the render scale used by ``page.get_pixmap(dpi=...)``.
    """
    w, h = width, height
    rot = rotation % 360
    m = {0: IDENTITY, 90: (0, -1, h, 1, 0, 0, 0, 0, 1), 180: (-1, 0, w, 0, -1, h, 0, 0, 1),
         270: (0, 1, 0, -1, 0, w, 0, 0, 1)}.get(rot)
    if m is None:
        raise ValueError(f"unsupported /Rotate {rotation}")
    ch = TransformChain([Step(f"page_rotate_{rot}", "pdf_unrot_tl", "pdf_points", m, (rot, w, h))])
    if dpi:
        ch.add(Step("render", "pdf_points", "render_px", scale(dpi / 72.0), (dpi,)))
    return ch


def render_to_points_chain(dpi: float) -> TransformChain:
    """render_px -> pdf_points (inverse of the render step)."""
    return TransformChain([Step("unrender", "render_px", "pdf_points", scale(72.0 / dpi), (dpi,))])
