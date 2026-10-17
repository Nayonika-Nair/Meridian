"""Stand-alone document sanitisation engine (A11).

    result = process_document(input_path, output_path, policy, options)

Pipeline: validate content & limits -> inspect (detect) -> apply (format-aware) -> reopen and verify
independently -> decide release -> persist minimised region metadata -> clean up.

The returned ``ProcessingResult`` separates four things that must not be conflated:

    detection      what the rules found, with original coordinates and rule IDs
    sanitisation   what was actually applied, per region
    verification   what independent checks of the reopened artifact could confirm
    unsupported    payloads the engine cannot inspect or sanitise (reported, never skipped silently)

There is no universal guarantee: a clean verification means "the implemented rules, extractors and OCR
did not find the targets", not that no sensitive information of any kind remains.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from collections import Counter
from pathlib import Path

from . import image_ops, ooxml, pdf_ops
from .capabilities import CAPABILITIES, PayloadReport, format_family
from .context import RunContext
from .detect import DetectContext
from .policy import Options, Policy
from .regions import Region, RegionStore
from .rules import REGISTRY_VERSION, build_registry
from .safety import (Deadline, DeadlineExceeded, InputRejected, WorkDir, atomic_publish, sha256_file, validate_input)
from .verify import finalise_statuses, overall_status, verify_image, verify_ooxml, verify_pdf

ENGINE_VERSION = "1.0.0"

# final release statuses
APPROVED = "APPROVED_FOR_RELEASE"
REVIEW = "REVIEW_REQUIRED"
BLOCKED = "BLOCKED"
REJECTED = "REJECTED_INPUT"
FAILED = "FAILED"

NOTE = ("Verification means the implemented rules, text extractors and OCR could not find the targets in the reopened "
        "output. It is not a guarantee that no sensitive information of any kind remains.")


class ProcessingResult(dict):
    """A plain dict (JSON-serialisable) with a couple of convenience accessors."""

    @property
    def released(self) -> bool:
        return bool(self["release"]["released"])

    @property
    def status(self) -> str:
        return self["release"]["status"]

    def to_json(self, **kw) -> str:
        return json.dumps(self, default=str, **kw)


def _skeleton(run_id: str, policy: Policy, registry_version: str) -> ProcessingResult:
    return ProcessingResult({
        "run_id": run_id, "engine_version": ENGINE_VERSION, "rule_registry_version": registry_version,
        "policy": {"name": policy.name, "version": policy.version, "fingerprint": policy.fingerprint()},
        "input": {}, "output": None, "detected": {"categories": {}, "rule_ids": {}, "regions": 0},
        "regions": [], "actions": {}, "coverage": {}, "sanitisation": {}, "verification": {"status": "not_run", "checks": []},
        "payloads": [], "unsupported": [], "warnings": [], "errors": [],
        "release": {"status": FAILED, "released": False, "reasons": []}, "cleanup": {}, "timings": {}, "note": NOTE,
    })


def process_document(input_path: str | os.PathLike, output_path: str | os.PathLike,
                     policy: Policy | dict | None = None, options: Options | dict | None = None) -> ProcessingResult:
    """Sanitise ``input_path`` into ``output_path`` (a separate artifact; the input is never modified)."""
    policy = Policy.from_dict(policy) if isinstance(policy, dict) else (policy or Policy())
    options = Options(**options) if isinstance(options, dict) else (options or Options())
    input_path, output_path = str(input_path), str(output_path)
    run_id = uuid.uuid4().hex[:16]
    registry = build_registry(policy)
    res = _skeleton(run_id, policy, registry.version)
    res["warnings"].extend(registry.warnings)
    t_start = time.monotonic()
    store = RegionStore(options.store_path) if (options.store_path and options.persist_regions) else None
    salt = options.salt or (store.salt if store else os.urandom(32))
    rc: RunContext | None = None
    regions: list[Region] = []
    kind = None
    try:
        if os.path.abspath(input_path) == os.path.abspath(output_path):
            raise InputRejected("same_path", "output_path must differ from input_path (the input is never overwritten)")
        info = validate_input(input_path, policy.limits, allow_extension_mismatch=policy.allow_extension_mismatch)
        kind = info["kind"]
        in_sha = sha256_file(input_path)
        res["input"] = {"name": Path(input_path).name, "sha256": in_sha, "bytes": info["bytes"], "detected_type": kind,
                        "mime": info["mime"], "extension_mismatch": info["extension_mismatch"], "archive": info["archive"]}
        if info["extension_mismatch"]:
            res["warnings"].append("extension_mismatch: processed according to file content, not extension")
        out_ext = Path(output_path).suffix.lower()
        from .safety import EXTENSIONS
        if out_ext not in EXTENSIONS[kind] and not policy.allow_extension_mismatch:
            raise InputRejected("output_extension_mismatch", f"output extension '{out_ext}' does not match detected type {kind}")
        family = format_family(kind)
        detect = DetectContext(registry=registry, policy=policy, doc_id=options.doc_id or in_sha, salt=salt,
                               warnings=res["warnings"], deadline=None, store_raw_text=options.store_raw_text)
        deadline = Deadline(policy.limits.max_duration_s)
        detect.deadline = deadline
        with WorkDir(options.work_dir) as wd:
            rc = RunContext(detect=detect, policy=policy, options=options, limits=policy.limits, deadline=deadline,
                            payloads=PayloadReport(family), workdir=wd, warnings=res["warnings"], errors=res["errors"])
            rc.regions = regions
            rc.src_path = input_path
            ext = Path(output_path).suffix or ".bin"
            tmp_out = wd.new_path(ext)
            t0 = time.monotonic()
            checks = _run_pipeline(rc, kind, family, input_path, tmp_out)
            res["timings"]["pipeline_s"] = round(time.monotonic() - t0, 3)
            rc.payloads.declare_all()
            vstatus = overall_status(checks) if checks else "not_run"
            counts = finalise_statuses(regions)
            _assemble(res, rc, regions, checks, vstatus, counts, policy)
            status, reasons = _decide(res, rc, regions, vstatus, policy)
            artifact = _publish(res, tmp_out, output_path, status, options)
            res["release"] = {"status": status, "released": status == APPROVED and artifact["released"], "reasons": reasons}
            res["output"] = artifact
        res["cleanup"] = wd.report()
    except InputRejected as exc:
        res["errors"].append({"code": exc.code, "message": exc.message})
        res["release"] = {"status": REJECTED, "released": False, "reasons": [f"{exc.code}: {exc.message}"]}
        if rc is not None:
            res["cleanup"] = rc.workdir.report()
    except DeadlineExceeded as exc:
        res["errors"].append({"code": "duration_limit", "message": str(exc)})
        res["release"] = {"status": FAILED, "released": False, "reasons": ["processing duration limit exceeded"]}
    except Exception as exc:                               # never leak document content through an exception message
        res["errors"].append({"code": "processing_error", "message": f"{type(exc).__name__}: {str(exc)[:200]}"})
        res["release"] = {"status": FAILED, "released": False, "reasons": ["unexpected processing error"]}
    finally:
        res["timings"]["total_s"] = round(time.monotonic() - t_start, 3)
    _persist(res, regions, store, options, kind)
    for r in regions:                                      # drop raw sensitive strings as soon as the run is over
        r.raw = ""
        r.replacement = ""
        r.handle = None
    return res


# --------------------------------------------------------------------------- pipeline
def _run_pipeline(rc: RunContext, kind: str, family: str, input_path: str, tmp_out: str) -> list[dict]:
    if family == "pdf":
        doc = pdf_ops.open_pdf(input_path, rc.limits)
        try:
            plan = pdf_ops.scan_pdf(rc, doc)
            pdf_ops.apply_pdf(rc, doc, plan, tmp_out)
        finally:
            doc.close()
        return verify_pdf(rc, tmp_out, plan, rc.regions)
    if family in ("docx", "pptx"):
        obj = ooxml.open_package(input_path, family)
        if family == "pptx" and len(obj.slides) > rc.limits.max_slides:
            raise InputRejected("too_many_slides", f"{len(obj.slides)} slides exceeds limit {rc.limits.max_slides}")
        plan = ooxml.scan_ooxml(rc, obj, family)
        ooxml.apply_ooxml(rc, obj, plan, tmp_out)
        originals = {}
        return verify_ooxml(rc, tmp_out, plan, rc.regions, family, originals)
    # stand-alone image
    img, scan = image_ops.scan_image_file(rc, input_path)
    try:
        image_ops.apply_image_file(rc, img, scan, tmp_out)
        return verify_image(rc, tmp_out, img, rc.regions)
    finally:
        img.close()


# --------------------------------------------------------------------------- reporting
def _assemble(res, rc, regions, checks, vstatus, counts, policy) -> None:
    recs = [r.to_record() for r in sorted(regions, key=lambda r: (r.unit_id, r.region_id))]
    det = [r for r in regions if r.source_method != "verification_rescan"]
    res["regions"] = recs
    res["detected"] = {"regions": len(det), "categories": dict(Counter(r.category for r in det)),
                       "rule_ids": dict(Counter(r.rule_id for r in det)),
                       "hard_block_regions": sum(1 for r in det if r.hard_block)}
    res["actions"] = dict(Counter(r.action for r in regions))
    acting = [r for r in regions if r.action in ("REDACT", "MASK", "GENERALISE")]
    applied = [r for r in acting if r.verification_status != "failed"]
    res["sanitisation"] = {
        "eligible": len(acting), "applied": len(applied), "failed": len([r for r in acting if r.verification_status == "failed"]),
        "review_required": counts.get("review_required", 0), "not_applicable": counts.get("not_applicable", 0),
        "applied_ratio": round(len(applied) / len(acting), 4) if acting else 1.0}
    res["verification"] = {"status": vstatus, "checks": checks, "regions": counts,
                           "verified_ratio": round(counts.get("verified", 0) / len(acting), 4) if acting else 1.0}
    rows = rc.payloads.rows()
    res["payloads"] = rows
    res["unsupported"] = rc.payloads.unsupported()
    c = rc.counters
    ok_payloads = [p for p in rows if p["outcome"] in ("inspected", "sanitised", "preserved")]
    res["coverage"] = {
        "units_total": c["units_total"], "units_inspected": c["units_inspected"],
        "unit_ratio": round(c["units_inspected"] / c["units_total"], 4) if c["units_total"] else 1.0,
        "images_total": c["images_total"], "images_ocr": c["images_ocr"], "images_ocr_failed": c["images_ocr_failed"],
        "payload_types": len(rows), "payload_types_inspected": len(ok_payloads),
        "payload_ratio": round(len(ok_payloads) / len(rows), 4) if rows else 1.0,
        "ocr_seconds": round(c["ocr_seconds"], 2), "ocr_enabled": policy.ocr}
    res["capabilities"] = {p: {"max": CAPABILITIES.get(rc.payloads.fmt, {}).get(p, ("inspect", ""))[0]}
                           for p in CAPABILITIES.get(rc.payloads.fmt, {})}


def _decide(res, rc, regions, vstatus, policy) -> tuple[str, list[str]]:
    reasons: list[str] = []
    required = set(policy.required_payloads)
    if policy.faces:
        required.add("face_regions")
    problems = rc.payloads.problems(sorted(required)) if policy.require_complete_inspection else []
    failed_regions = [r for r in regions if r.verification_status == "failed"]
    review_regions = [r for r in regions if r.verification_status == "review_required"]
    if res["errors"]:
        return FAILED, ["processing errors"]
    status = APPROVED
    if vstatus == "failed" or failed_regions:
        reasons.append(f"verification failed ({len(failed_regions)} region(s) not confirmed removed)")
        status = BLOCKED if policy.release_requires_verification else REVIEW
    if status != BLOCKED:
        if vstatus == "incomplete":
            reasons.append("verification incomplete: a required independent check could not run")
            status = REVIEW
        if problems:
            reasons.append("incomplete inspection of required payload(s): " + ", ".join(sorted({p["payload"] for p in problems})))
            status = REVIEW
        if review_regions:
            reasons.append(f"{len(review_regions)} finding(s) need human review")
            status = REVIEW
    if status == REVIEW and policy.release_on_review is True:
        reasons.append("release_on_review policy override: artifact released despite open review items")
        status = APPROVED
    if status == APPROVED and not reasons:
        reasons.append("all findings verified; no required payload left uninspected")
    res["incomplete"] = bool(problems) or vstatus == "incomplete"
    return status, reasons


def _publish(res, tmp_out: str, output_path: str, status: str, options: Options) -> dict:
    if not os.path.exists(tmp_out):
        return {"path": None, "released": False}
    sha = sha256_file(tmp_out)
    size = os.path.getsize(tmp_out)
    if status == APPROVED:
        atomic_publish(tmp_out, output_path)
        return {"path": output_path, "sha256": sha, "bytes": size, "released": True}
    if options.quarantine_failed:
        q = output_path + ".quarantined"
        atomic_publish(tmp_out, q)
        return {"path": q, "sha256": sha, "bytes": size, "released": False,
                "note": "artifact held back - not released; inspect before any use"}
    return {"path": None, "sha256": sha, "bytes": size, "released": False}


def _persist(res, regions, store, options, kind) -> None:
    if store is not None and res.get("input"):
        try:
            summary = {"detected": res["detected"], "sanitisation": res["sanitisation"],
                       "verification_status": res["verification"]["status"], "unsupported": [u["payload"] for u in res["unsupported"]]}
            store.save_run(res["run_id"], options.doc_id or res["input"]["sha256"], res["input"]["sha256"],
                           (res["output"] or {}).get("sha256"), kind or "", res["release"]["status"], summary, regions)
        except Exception as exc:
            res["warnings"].append(f"region_store_failed: {type(exc).__name__}")
    if options.ledger is not None and res.get("input"):
        try:
            meta = {"event": "document_sanitisation", "run_id": res["run_id"], "release_status": res["release"]["status"],
                    "input_sha256": res["input"]["sha256"], "output_sha256": (res["output"] or {}).get("sha256"),
                    "categories": res["detected"]["categories"], "verification": res["verification"]["status"],
                    "policy": res["policy"]["fingerprint"], "rule_registry": res["rule_registry_version"], "synthetic": False}
            options.ledger.append(options.audit_destination, res["input"]["name"], 0.0, 0.0, 0.0, meta)
        except Exception as exc:
            res["warnings"].append(f"audit_ledger_failed: {type(exc).__name__}")


def verify_document(input_path: str | os.PathLike, candidate_path: str | os.PathLike,
                    policy: Policy | dict | None = None, options: Options | dict | None = None) -> ProcessingResult:
    """Independently verify ANY sanitised candidate against the original.

    The original is scanned with the policy's rules to find the targets; the candidate is then reopened
    and checked exactly like an engine-produced artifact (text channels, OCR, metadata, preserved content).
    Nothing is written and nothing is released: the result only states whether the candidate passes.
    """
    policy = Policy.from_dict(policy) if isinstance(policy, dict) else (policy or Policy())
    options = Options(**options) if isinstance(options, dict) else (options or Options())
    input_path, candidate_path = str(input_path), str(candidate_path)
    registry = build_registry(policy)
    res = _skeleton(uuid.uuid4().hex[:16], policy, registry.version)
    res["mode"] = "verify_only"
    res["warnings"].extend(registry.warnings)
    regions: list[Region] = []
    t_start = time.monotonic()
    try:
        info = validate_input(input_path, policy.limits, allow_extension_mismatch=policy.allow_extension_mismatch)
        kind = info["kind"]
        family = format_family(kind)
        in_sha = sha256_file(input_path)
        res["input"] = {"name": Path(input_path).name, "sha256": in_sha, "bytes": info["bytes"], "detected_type": kind, "mime": info["mime"]}
        if not os.path.exists(candidate_path):
            raise InputRejected("not_found", "candidate file not found")
        cand_info = validate_input(candidate_path, policy.limits, allow_extension_mismatch=True)
        if cand_info["kind"] != kind:
            raise InputRejected("type_changed", f"candidate is {cand_info['kind']} but the original is {kind}")
        res["output"] = {"path": candidate_path, "sha256": sha256_file(candidate_path), "bytes": cand_info["bytes"], "released": False}
        detect = DetectContext(registry=registry, policy=policy, doc_id=in_sha, salt=os.urandom(32), warnings=res["warnings"])
        deadline = Deadline(policy.limits.max_duration_s)
        detect.deadline = deadline
        with WorkDir(options.work_dir) as wd:
            rc = RunContext(detect=detect, policy=policy, options=options, limits=policy.limits, deadline=deadline,
                            payloads=PayloadReport(family), workdir=wd, warnings=res["warnings"], errors=res["errors"])
            rc.regions = regions
            rc.src_path = input_path
            if family == "pdf":
                doc = pdf_ops.open_pdf(input_path, rc.limits)
                try:
                    plan = pdf_ops.scan_pdf(rc, doc)
                finally:
                    doc.close()
                checks = verify_pdf(rc, candidate_path, plan, regions)
            elif family in ("docx", "pptx"):
                obj = ooxml.open_package(input_path, family)
                plan = ooxml.scan_ooxml(rc, obj, family)
                ooxml.simulate_expected(plan, regions)
                checks = verify_ooxml(rc, candidate_path, plan, regions, family, {})
            else:
                img, scan = image_ops.scan_image_file(rc, input_path)
                try:
                    checks = verify_image(rc, candidate_path, img, regions)
                finally:
                    img.close()
            rc.payloads.declare_all()
            vstatus = overall_status(checks)
            counts = finalise_statuses(regions)
            _assemble(res, rc, regions, checks, vstatus, counts, policy)
            status, reasons = _decide(res, rc, regions, vstatus, policy)
            res["release"] = {"status": status, "released": False, "reasons": reasons + ["verify-only: nothing was released"]}
            res["cleanup"] = wd.report()
    except InputRejected as exc:
        res["errors"].append({"code": exc.code, "message": exc.message})
        res["release"] = {"status": REJECTED, "released": False, "reasons": [f"{exc.code}: {exc.message}"]}
    except Exception as exc:
        res["errors"].append({"code": "verification_error", "message": f"{type(exc).__name__}: {str(exc)[:200]}"})
        res["release"] = {"status": FAILED, "released": False, "reasons": ["unexpected verification error"]}
    finally:
        res["timings"]["total_s"] = round(time.monotonic() - t_start, 3)
        for r in regions:
            r.raw = ""
            r.replacement = ""
            r.handle = None
    return res
