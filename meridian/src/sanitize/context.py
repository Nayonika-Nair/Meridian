"""Shared per-run context for all format handlers."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .capabilities import PayloadReport
from .detect import DetectContext
from .imaging import OcrTimeout, OcrUnavailable, ocr_available
from .policy import Limits, Options, Policy
from .regions import Region
from .safety import Deadline, WorkDir


@dataclass
class RunContext:
    detect: DetectContext
    policy: Policy
    options: Options
    limits: Limits
    deadline: Deadline
    payloads: PayloadReport
    workdir: WorkDir
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    counters: dict = field(default_factory=lambda: {
        "units_total": 0, "units_inspected": 0, "images_total": 0, "images_ocr": 0,
        "images_ocr_failed": 0, "ocr_seconds": 0.0, "ocr_skipped_budget": 0})
    regions: list[Region] = field(default_factory=list)
    src_path: str | None = None
    _ocr_checked: bool = False
    _ocr_ok: bool = False

    def warn(self, msg: str) -> None:
        if msg not in self.warnings:
            self.warnings.append(msg)

    @property
    def doc_id(self) -> str:
        return self.detect.doc_id

    def ocr_ready(self) -> bool:
        if not self.policy.ocr:
            return False
        if self.options.ocr_engine is not None:
            return True
        if not self._ocr_checked:
            self._ocr_ok = ocr_available()
            self._ocr_checked = True
            if not self._ocr_ok:
                self.warn("ocr_unavailable: tesseract not found - image-borne text cannot be inspected")
        return self._ocr_ok

    def ocr_budget_left(self) -> float:
        return max(0.0, self.limits.ocr_total_s - self.counters["ocr_seconds"])

    def charge_ocr(self, started: float) -> None:
        self.counters["ocr_seconds"] += time.monotonic() - started
