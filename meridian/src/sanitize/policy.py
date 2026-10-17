"""Sanitisation policy and processing limits.

A Policy can only make the engine STRICTER than the mandatory rules; it cannot switch off
hard-block rules, allow-list their values, or choose an action weaker than REDACT for them
(see rules.build_registry and detect.resolve_action).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Limits:
    """Configurable resource limits (A10). Every limit is enforced before or during processing."""
    max_file_bytes: int = 50 * 1024 * 1024
    max_pages: int = 300
    max_slides: int = 400
    max_image_pixels: int = 60_000_000            # width * height of any single image
    max_archive_entries: int = 5_000
    max_archive_uncompressed_bytes: int = 400 * 1024 * 1024
    max_archive_ratio: float = 200.0              # per-entry compression ratio
    max_embedded_objects: int = 100
    ocr_timeout_per_image_s: float = 45.0
    ocr_total_s: float = 600.0
    max_duration_s: float = 900.0
    ocr_dpi: int = 200
    max_ocr_images: int = 400


@dataclass
class Policy:
    name: str = "default"
    version: str = "1"
    # rule control ---------------------------------------------------------
    rule_toggles: dict[str, bool] = field(default_factory=dict)
    confidence_overrides: dict[str, float] = field(default_factory=dict)
    category_actions: dict[str, str] = field(default_factory=dict)      # e.g. {"EMAIL": "MASK"}
    rule_actions: dict[str, str] = field(default_factory=dict)          # highest-priority action override
    custom_rules: list[dict[str, Any]] = field(default_factory=list)
    allowlist: list[str] = field(default_factory=list)                  # never applied to hard-block rules
    denylist: list[str] = field(default_factory=list)                   # always redacted
    # behaviour ------------------------------------------------------------
    destination_risk: str = "MODERATE"                                  # LOW|MODERATE|HIGH|CRITICAL
    ocr: bool = True
    faces: bool = False
    review_low_confidence: bool = True
    mask_keep_last: int = 4
    strip_author_metadata: bool = False                                 # blank author-like core properties
    allow_extension_mismatch: bool = False
    # release gates --------------------------------------------------------
    require_complete_inspection: bool = True
    required_payloads: list[str] = field(default_factory=lambda: ["body_text", "scanned_images", "embedded_images", "metadata", "embedded_objects"])
    release_requires_verification: bool = True
    release_on_review: bool = False                                     # REVIEW_REQUIRED artifacts are held unless True
    limits: Limits = field(default_factory=Limits)

    # ---------------------------------------------------------------- serialisation
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Policy":
        d = dict(d)
        lim = d.pop("limits", None)
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        p = cls(**known)
        if lim:
            p.limits = Limits(**{k: v for k, v in lim.items() if k in Limits.__dataclass_fields__})
        return p

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, default=str).encode()).hexdigest()[:16]


@dataclass
class Options:
    """Per-run options that are not part of the privacy policy."""
    store_path: str | None = None            # SQLite region store; None -> not persisted
    persist_regions: bool = True
    ledger: Any = None                       # optional src.ledger.Ledger for the audit record
    audit_destination: str = "sanitisation-engine"
    doc_id: str | None = None
    work_dir: str | None = None              # parent for temp files (default: system temp)
    face_detector: Any = None                # callable(PIL.Image) -> list[(x0,y0,x1,y1)] (tests / custom models)
    ocr_engine: Any = None                   # callable(PIL.Image, timeout) -> list[dict] (tests / alt engines)
    quarantine_failed: bool = True           # keep failed artifact as <output>.quarantined for analyst inspection
    store_raw_text: bool = False             # never True in normal operation
    salt: bytes | None = None                # override the per-store salt (tests)
