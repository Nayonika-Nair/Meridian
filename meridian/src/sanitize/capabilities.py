"""Format capability registry (A7) and per-run payload reporting.

``CAPABILITIES`` states, per format and payload type, the MAXIMUM the engine can do:

    sanitise     content is inspected and the matching part can be rewritten / removed
    inspect      content is read and reported, but not modified
    preserve     content is deliberately left as-is (it cannot hold the targeted data, or removing it
                 would damage the document); it is still listed in the report
    unsupported  the engine cannot look inside; presence is reported and may force review

At run time each payload gets one of five outcomes (``PayloadReport``):
    inspected | sanitised | preserved | unsupported | failed
Nothing is silently skipped: an absent payload is recorded as "inspected" with count 0.
"""
from __future__ import annotations

from dataclasses import dataclass, field

LEVELS = ("sanitise", "inspect", "preserve", "unsupported")
OUTCOMES = ("inspected", "sanitised", "preserved", "unsupported", "failed")

# payload -> (level, note)
CAPABILITIES: dict[str, dict[str, tuple[str, str]]] = {
    "pdf": {
        "body_text": ("sanitise", "native text removed with PyMuPDF content-stream redaction (not an overlay)"),
        "scanned_images": ("sanitise", "OCR + pixel blanking of the image under the region"),
        "embedded_images": ("sanitise", "images under a redaction are pixel-blanked; others preserved"),
        "annotations": ("sanitise", "annotation text scanned; matching annotations are rewritten or removed"),
        "form_fields": ("sanitise", "widget values scanned and rewritten"),
        "metadata": ("sanitise", "Info dictionary and XMP scanned; only matching values are changed"),
        "bookmarks": ("inspect", "outline titles scanned; reported when they match"),
        "embedded_objects": ("unsupported", "attached files are listed; contents are not sanitised"),
        "javascript_macros": ("inspect", "document-level JavaScript detected and reported"),
        "hidden_text": ("sanitise", "invisible text (render mode 3 / white-on-white) is part of native text"),
        "face_regions": ("sanitise", "optional local detector; detection only, no identification"),
    },
    "docx": {
        "body_text": ("sanitise", "paragraph/run-level replacement, spans across runs, tracks each occurrence"),
        "tables": ("sanitise", "table cell paragraphs"),
        "headers_footers": ("sanitise", "all header and footer parts"),
        "footnotes_endnotes": ("sanitise", "footnote and endnote parts"),
        "comments": ("sanitise", "comment text; comment authors are reported and preserved unless policy says otherwise"),
        "text_boxes": ("sanitise", "text boxes and alternate-content fallbacks"),
        "hidden_text": ("sanitise", "runs marked vanish and tracked-change deletions are scanned and rewritten"),
        "hyperlinks": ("sanitise", "external link targets scanned (e.g. mailto:)"),
        "field_codes": ("sanitise", "field instructions scanned"),
        "embedded_images": ("sanitise", "OCR + pixel blanking with copy-on-write per occurrence"),
        "thumbnails": ("sanitise", "package thumbnail removed when any finding exists (it is a rendered preview)"),
        "metadata": ("sanitise", "core/app/custom properties scanned; only matching values are changed"),
        "charts_diagrams": ("inspect", "chart and diagram text scanned; embedded workbooks are unsupported"),
        "custom_xml": ("inspect", "customXml parts scanned as text"),
        "embedded_objects": ("unsupported", "OLE objects, embedded workbooks, VBA projects, altChunk content"),
        "face_regions": ("sanitise", "optional local detector on embedded images"),
    },
    "pptx": {
        "body_text": ("sanitise", "slide shapes, placeholders and groups"),
        "tables": ("sanitise", "table cells"),
        "notes": ("sanitise", "speaker notes"),
        "comments": ("sanitise", "slide comments; comment authors reported and preserved unless policy says otherwise"),
        "masters_layouts": ("sanitise", "layouts, masters, headers and footers"),
        "alt_text": ("sanitise", "shape alt-text / titles"),
        "hyperlinks": ("sanitise", "external link targets"),
        "hidden_text": ("sanitise", "hidden slides are inspected and reported"),
        "embedded_images": ("sanitise", "OCR + pixel blanking with copy-on-write per occurrence"),
        "thumbnails": ("sanitise", "package thumbnail removed when any finding exists"),
        "metadata": ("sanitise", "core/app/custom properties scanned; only matching values are changed"),
        "charts_diagrams": ("inspect", "chart text scanned; embedded workbooks are unsupported"),
        "embedded_objects": ("unsupported", "OLE objects, embedded workbooks, media files, VBA projects"),
        "face_regions": ("sanitise", "optional local detector on embedded images"),
    },
    "image": {
        "body_text": ("sanitise", "OCR text regions are painted out in original pixel space"),
        "embedded_images": ("inspect", "n/a for a bare image"),
        "face_regions": ("sanitise", "optional local detector; detection only, no identification"),
        "metadata": ("sanitise", "EXIF: GPS, maker notes and free-text fields removed; orientation and technical tags kept"),
        "embedded_objects": ("preserve", "bare images carry no attachments; trailing data after the image is dropped on re-encode"),
    },
}

IMAGE_FORMATS = ("png", "jpeg", "tiff", "bmp", "gif", "webp")


def format_family(kind: str) -> str:
    return "image" if kind in IMAGE_FORMATS else kind


def capability_matrix() -> list[dict]:
    rows = []
    for fmt, payloads in CAPABILITIES.items():
        for payload, (level, note) in payloads.items():
            rows.append({"format": fmt, "payload": payload, "capability": level, "note": note})
    return rows


@dataclass
class PayloadReport:
    fmt: str
    items: dict[str, dict] = field(default_factory=dict)

    def add(self, payload: str, outcome: str, count: int = 0, detail: str = "") -> None:
        if outcome not in OUTCOMES:
            raise ValueError(outcome)
        cur = self.items.get(payload)
        rank = {o: i for i, o in enumerate(("preserved", "inspected", "sanitised", "unsupported", "failed"))}
        if cur:
            cur["count"] = cur["count"] + count if outcome == cur["outcome"] else max(cur["count"], count)
            if rank[outcome] > rank[cur["outcome"]]:
                cur["outcome"] = outcome
            if detail and detail not in cur["detail"]:
                cur["detail"] = (cur["detail"] + "; " + detail).strip("; ")
        else:
            level, note = CAPABILITIES.get(self.fmt, {}).get(payload, ("inspect", ""))
            self.items[payload] = {"payload": payload, "outcome": outcome, "count": count, "detail": detail,
                                   "max_capability": level}

    def declare_all(self) -> None:
        """Make sure every payload type in the registry appears (never silently skipped)."""
        for payload, (level, note) in CAPABILITIES.get(self.fmt, {}).items():
            if payload not in self.items:
                if level == "unsupported":
                    self.add(payload, "inspected", 0, "none present")
                else:
                    self.add(payload, "inspected", 0, "none present or not applicable")

    def rows(self) -> list[dict]:
        return sorted(self.items.values(), key=lambda r: r["payload"])

    def problems(self, required: list[str]) -> list[dict]:
        return [r for r in self.rows() if r["payload"] in required and r["outcome"] in ("unsupported", "failed")]

    def unsupported(self) -> list[dict]:
        return [r for r in self.rows() if r["outcome"] in ("unsupported", "failed")]
