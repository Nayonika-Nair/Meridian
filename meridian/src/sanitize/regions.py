"""Persistent detection regions (A2) and region grouping / duplicate suppression (A8).

A Region is the unit of accountability: every finding that the engine acts on (or deliberately
does not act on) is a Region with a stable ID, original coordinates, rule provenance, the applied
action and a verification status. Raw sensitive text is held in memory only (``raw``) and is never
serialised: persisted rows carry a keyed hash and a character-class "shape" instead.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .coords import (Rect, rect_area, rect_intersection, rect_iou, rect_norm, rect_overlap_fraction,
                     rect_pad, rect_union)
from .rules import STRICTNESS, SEVERITY

VERIFICATION_STATES = ("pending", "verified", "failed", "review_required", "not_applicable")
ACTIONS_NO_CHANGE = ("NONE", "ALLOWLISTED", "FLAG_REVIEW")


def text_shape(value: str, limit: int = 48) -> str:
    out = []
    for ch in value[:limit]:
        out.append("d" if ch.isdigit() else "A" if ch.isupper() else "a" if ch.isalpha() else ch if ch in " -.@/:+()_," else "*")
    return "".join(out) + ("..." if len(value) > limit else "")


def keyed_hash(salt: bytes, value: str) -> str:
    norm = "".join(ch for ch in value if ch.isalnum()).lower()          # formatting variants hash alike
    return hmac.new(salt, norm.encode("utf-8", "ignore"), hashlib.sha256).hexdigest()[:24]


def make_region_id(doc_sha: str, unit_id: str, rule_id: str, anchor_key: str) -> str:
    h = hashlib.sha256(f"{doc_sha}|{unit_id}|{rule_id}|{anchor_key}".encode()).hexdigest()
    return "R-" + h[:16]


@dataclass
class Region:
    region_id: str
    document_id: str
    unit_id: str                       # 'page:2' | 'slide:3' | 'docx:word/document.xml:p14' | 'image' | 'meta:core'
    space: str                         # coordinate space of rects: pdf_points | original_px | text_anchor
    rects: list[Rect] = field(default_factory=list)
    anchor: dict = field(default_factory=dict)
    category: str = ""
    rule_id: str = ""
    entity_type: str = ""
    severity: str = "HIGH"
    hard_block: bool = False
    confidence: float = 0.0
    source_method: str = ""            # native_text | ocr | docx_text | pptx_text | metadata | face_detector
    action: str = "NONE"
    verification_status: str = "pending"
    reason: str = ""
    text_hash: str = ""
    text_shape: str = ""
    text_len: int = 0
    also_matched: list[str] = field(default_factory=list)
    merged_from: list[str] = field(default_factory=list)
    chain: list[dict] = field(default_factory=list)       # transform chain description (OCR-derived)
    page_size: tuple[float, float] | None = None
    # --- in-memory only (never persisted / serialised) ---
    raw: str = field(default="", repr=False, compare=False)
    replacement: str = field(default="", repr=False, compare=False)
    handle: object = field(default=None, repr=False, compare=False)

    @property
    def bbox(self) -> Rect | None:
        return rect_union(self.rects) if self.rects else None

    def to_record(self) -> dict:
        d = {
            "region_id": self.region_id, "document_id": self.document_id, "unit_id": self.unit_id,
            "space": self.space, "rects": [list(map(lambda v: round(v, 3), r)) for r in self.rects],
            "bbox": [round(v, 3) for v in self.bbox] if self.bbox else None, "anchor": self.anchor,
            "category": self.category, "rule_id": self.rule_id, "entity_type": self.entity_type,
            "severity": self.severity, "hard_block": self.hard_block, "confidence": round(self.confidence, 4),
            "source_method": self.source_method, "action": self.action,
            "verification_status": self.verification_status, "reason": self.reason,
            "text_hash": self.text_hash, "text_shape": self.text_shape, "text_len": self.text_len,
            "also_matched": list(self.also_matched), "merged_from": list(self.merged_from),
            "chain": self.chain, "page_size": list(self.page_size) if self.page_size else None,
        }
        return d


# --------------------------------------------------------------------------- grouping (A8)
def _stricter(a: str, b: str) -> str:
    order = {"NONE": -2, "ALLOWLISTED": -2, "FLAG_REVIEW": 0}
    ra = STRICTNESS.get(a, order.get(a, -3))
    rb = STRICTNESS.get(b, order.get(b, -3))
    return a if ra >= rb else b


def merge_line_rects(rects: list[Rect], gap_tol: float) -> list[Rect]:
    """Join rectangles of ONE finding that sit on the same line with at most ``gap_tol`` between them.

    Intended for the word boxes of a single multi-word match (so no un-redacted gap is left between
    words); it is never applied across different findings.
    """
    out: list[Rect] = []
    for r in sorted((rect_norm(r) for r in rects), key=lambda r: (round((r[1] + r[3]) / 2, 0), r[0])):
        if out:
            p = out[-1]
            same_line = rect_overlap_vertical(p, r) > 0.5
            if same_line and r[0] - p[2] <= gap_tol:
                out[-1] = (min(p[0], r[0]), min(p[1], r[1]), max(p[2], r[2]), max(p[3], r[3]))
                continue
        out.append(r)
    return out


def rect_overlap_vertical(a: Rect, b: Rect) -> float:
    lo, hi = max(a[1], b[1]), min(a[3], b[3])
    h = min(a[3] - a[1], b[3] - b[1])
    return max(0.0, hi - lo) / h if h > 0 else 0.0


def group_regions(regions: list[Region], *, overlap_merge: float = 0.5, adjacent_gap_chars: int = 1) -> list[Region]:
    """Duplicate suppression + overlap handling + contiguous same-rule grouping.

    Merge rules (anything else is left separate, however close the boxes are):
      1. identical rule on the same unit with IoU >= 0.9            -> duplicate (dropped, provenance kept)
      2. different rules whose boxes overlap by >= ``overlap_merge`` of the smaller box
                                                                     -> one region with ``also_matched``
      3. same rule, same unit, text-contiguous (gap <= adjacent_gap_chars whitespace) and on one line
                                                                     -> one region
    Provenance: ``merged_from`` lists the region IDs that were folded into the surviving region.
    """
    groups: dict[tuple, list[Region]] = {}
    for r in regions:
        groups.setdefault((r.unit_id, r.space), []).append(r)
    out: list[Region] = []
    for key, items in groups.items():
        items = sorted(items, key=lambda r: (r.bbox or (0, 0, 0, 0))[:2] if r.bbox else (0, 0))
        merged: list[Region] = []
        for r in items:
            host = None
            for m in merged:
                if _should_merge(m, r, overlap_merge, adjacent_gap_chars):
                    host = m
                    break
            if host is None:
                merged.append(r)
            else:
                _absorb(host, r)
        out.extend(merged)
    return out


def _should_merge(a: Region, b: Region, overlap_merge: float, gap_chars: int) -> bool:
    if not a.rects or not b.rects:
        return False
    ab, bb = a.bbox, b.bbox
    if a.rule_id == b.rule_id and rect_iou(ab, bb) >= 0.9:
        return True
    inter = rect_intersection(ab, bb)
    if inter and a.rule_id != b.rule_id:
        small = min(rect_area(ab), rect_area(bb)) or 1.0
        if rect_area(inter) / small >= overlap_merge:
            return True
    if a.rule_id == b.rule_id and a.anchor.get("end") is not None and b.anchor.get("start") is not None:
        gap = b.anchor["start"] - a.anchor["end"]
        if 0 <= gap <= gap_chars and rect_overlap_vertical(ab, bb) > 0.5:
            return True
    return False


def _absorb(host: Region, other: Region) -> None:
    host.merged_from = sorted(set(host.merged_from) | {other.region_id} | set(other.merged_from))
    if other.rule_id != host.rule_id:
        host.also_matched = sorted(set(host.also_matched) | {other.rule_id} | set(other.also_matched))
    host.rects = merge_line_rects(host.rects + other.rects, gap_tol=1.5) if host.rule_id == other.rule_id else host.rects + other.rects
    host.confidence = max(host.confidence, other.confidence)
    host.hard_block = host.hard_block or other.hard_block
    if SEVERITY.index(other.severity) > SEVERITY.index(host.severity):
        host.severity = other.severity
    host.action = _stricter(host.action, other.action)
    if host.anchor.get("end") is not None and other.anchor.get("end") is not None:
        host.anchor = {**host.anchor, "start": min(host.anchor["start"], other.anchor["start"]),
                       "end": max(host.anchor["end"], other.anchor["end"])}
    if other.raw and not host.raw:
        host.raw = other.raw


def pad_rect(r: Rect, pad: float, bounds: Rect | None = None) -> Rect:
    return rect_pad(r, pad, bounds)


# --------------------------------------------------------------------------- persistence (A2)
SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY, document_id TEXT, created REAL, input_sha256 TEXT, output_sha256 TEXT,
  detected_type TEXT, release_status TEXT, summary TEXT);
CREATE TABLE IF NOT EXISTS regions (
  region_id TEXT PRIMARY KEY, run_id TEXT, document_id TEXT, unit_id TEXT, space TEXT, rects TEXT, anchor TEXT,
  category TEXT, rule_id TEXT, entity_type TEXT, severity TEXT, hard_block INTEGER, confidence REAL,
  source_method TEXT, action TEXT, verification_status TEXT, reason TEXT, text_hash TEXT, text_shape TEXT,
  text_len INTEGER, also_matched TEXT, merged_from TEXT, chain TEXT, page_size TEXT, updated REAL);
CREATE INDEX IF NOT EXISTS ix_regions_doc ON regions(document_id);
"""


class RegionStore:
    """SQLite persistence. Every call opens and closes its own connection (Windows-safe, no lingering locks)."""

    def __init__(self, path: str | os.PathLike):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)
            row = c.execute("SELECT v FROM meta WHERE k='salt'").fetchone()
            if not row:
                c.execute("INSERT INTO meta(k,v) VALUES('salt', ?)", (secrets.token_hex(32),))
            c.commit()

    def _conn(self):
        conn = sqlite3.connect(self.path, timeout=30)
        conn.execute("PRAGMA busy_timeout=30000")
        return closing(conn)

    @property
    def salt(self) -> bytes:
        with self._conn() as c:
            return bytes.fromhex(c.execute("SELECT v FROM meta WHERE k='salt'").fetchone()[0])

    def save_run(self, run_id: str, document_id: str, input_sha: str, output_sha: str | None, detected_type: str,
                 release_status: str, summary: dict, regions: Iterable[Region]) -> int:
        n = 0
        now = time.time()
        with self._conn() as c:
            c.execute("INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?)",
                      (run_id, document_id, now, input_sha, output_sha, detected_type, release_status,
                       json.dumps(summary, sort_keys=True, default=str)))
            for r in regions:
                d = r.to_record()
                c.execute(
                    "INSERT OR REPLACE INTO regions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (d["region_id"], run_id, d["document_id"], d["unit_id"], d["space"], json.dumps(d["rects"]),
                     json.dumps(d["anchor"], sort_keys=True), d["category"], d["rule_id"], d["entity_type"],
                     d["severity"], int(d["hard_block"]), d["confidence"], d["source_method"], d["action"],
                     d["verification_status"], d["reason"], d["text_hash"], d["text_shape"], d["text_len"],
                     json.dumps(d["also_matched"]), json.dumps(d["merged_from"]), json.dumps(d["chain"]),
                     json.dumps(d["page_size"]), now))
                n += 1
            c.commit()
        return n

    def regions_for(self, document_id: str) -> list[dict]:
        with self._conn() as c:
            c.row_factory = sqlite3.Row
            rows = c.execute("SELECT * FROM regions WHERE document_id=? ORDER BY unit_id, region_id", (document_id,)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            for k in ("rects", "anchor", "also_matched", "merged_from", "chain", "page_size"):
                d[k] = json.loads(d[k]) if d[k] else None
            d["hard_block"] = bool(d["hard_block"])
            out.append(d)
        return out

    def get_region(self, region_id: str) -> dict | None:
        with self._conn() as c:
            c.row_factory = sqlite3.Row
            r = c.execute("SELECT * FROM regions WHERE region_id=?", (region_id,)).fetchone()
        if not r:
            return None
        d = dict(r)
        for k in ("rects", "anchor", "also_matched", "merged_from", "chain", "page_size"):
            d[k] = json.loads(d[k]) if d[k] else None
        return d

    def update_status(self, region_id: str, status: str, reason: str = "") -> None:
        if status not in VERIFICATION_STATES:
            raise ValueError(status)
        with self._conn() as c:
            c.execute("UPDATE regions SET verification_status=?, reason=?, updated=? WHERE region_id=?",
                      (status, reason, time.time(), region_id))
            c.commit()

    def runs(self, limit: int = 50) -> list[dict]:
        with self._conn() as c:
            c.row_factory = sqlite3.Row
            rows = c.execute("SELECT * FROM runs ORDER BY created DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]
