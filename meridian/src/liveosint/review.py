"""Analyst review -> Twin / Leak Budget. Only ACCEPTED, non-fixture, non-negated evidence becomes a Twin edge.

sync_vendor() is idempotent: it rebuilds the overlay rows owned by this module (those with `liveosint_id`) from the
accepted set and upserts them by evidence_id. Re-running it, or re-submitting the same review, changes nothing and
writes no audit row. Analyst-authored overlay rows (no `liveosint_id`) are never touched.
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone

from ..config import AI_USE_FLOW_DISCOUNT, BASE
from ..demo import vendor_assessment
from ..store import OVERLAY, _read, _write
from ..twin import AI_USE_ONLY, FLOW_PROVEN
from .extract import FLOW_CLASSES
from .store import STATUSES, Store

TAG = {"marketing_claim": "Marketing claim only", "technical_indication": "Technical proof (docs/SDK/job posts)",
       "corroborated_claim": "Corroborated (2+ independent sources)", "explicit_processing_disclosure": "Technical proof (docs/SDK/job posts)",
       "observed_flow": "Confirmed", "verified_flow": "Confirmed"}
CONF = {"marketing_claim": "Low", "technical_indication": "Medium", "corroborated_claim": "Medium",
        "explicit_processing_disclosure": "Medium", "observed_flow": "High", "verified_flow": "High"}


class ReviewError(ValueError):
    pass


def review_evidence(store: Store, evidence_id: str, decision: str, reviewer: str, note: str = "", *, observation_ref: str | None = None,
                    promote_to: str | None = None, now: float | None = None) -> dict:
    """Record a decision. Returns {'changed': bool, ...}. Re-submitting an identical decision is a no-op (no log row)."""
    now = time.time() if now is None else now
    if decision not in STATUSES or decision == "pending":
        raise ReviewError(f"decision must be one of accepted, rejected, needs_review (got {decision!r})")
    if not reviewer or not reviewer.strip():
        raise ReviewError("a reviewer name is required")
    ev = store.get(evidence_id)
    if ev is None:
        raise ReviewError("unknown evidence_id")
    if decision == "accepted":
        if ev.get("origin") == "fixture":
            raise ReviewError("offline-fixture evidence is test data and can never be accepted into the Twin")
        if ev.get("negated"):
            raise ReviewError("a negative statement cannot be accepted as AI evidence; reject it or mark it needs_review")
    if promote_to:
        if promote_to not in FLOW_CLASSES:
            raise ReviewError("promote_to must be observed_flow or verified_flow")
        if decision != "accepted":
            raise ReviewError("only an accepted item can be promoted")
        if not observation_ref or len(observation_ref.strip()) < 6:
            raise ReviewError("observed/verified flow requires an authorised observation reference (ticket, contract clause, audit id)")
    elif observation_ref:
        raise ReviewError("observation_ref only applies together with promote_to")
    same = (ev["review_status"] == decision and (ev["promoted_class"] or None) == (promote_to or None)
            and (ev["observation_ref"] or None) == ((observation_ref or "").strip() or None))
    if same:
        return {"changed": False, "evidence_id": evidence_id, "status": decision}
    age = None
    if decision == "accepted":
        age = max(0, int((now - float(ev["retrieved_at"])) / 86400))
    store.set_review(evidence_id, decision, reviewer.strip(), note.strip()[:500], (observation_ref or "").strip() or None, promote_to, age, now)
    return {"changed": True, "evidence_id": evidence_id, "status": decision}


def overlay_row(vendor: dict, ev: dict) -> dict:
    cls = ev["promoted_class"] or ev["evidence_class"]
    provs = ev.get("providers") or []
    flow = cls in FLOW_CLASSES
    row = {"parent": vendor["vendor_name"], "target": provs[0] if provs else "AI capability (provider unnamed)",
           "target_kind": "ai_provider" if provs else "ai", "tag": TAG[cls], "source": f"{ev['domain']} ({ev['source_type']})",
           "url": ev["final_url"], "excerpt": ev["excerpt"], "age_days": int(ev.get("accepted_age_days") or 0),
           "claim_type": FLOW_PROVEN if flow else AI_USE_ONLY, "evidence_type": f"Live OSINT - {cls}",
           "date": datetime.fromtimestamp(float(ev["retrieved_at"]), timezone.utc).strftime("%Y-%m-%d"),
           "date_basis": "Retrieval date recorded by the collector", "confidence": CONF[cls], "verified": True,
           "liveosint_id": ev["evidence_id"], "evidence_class": cls, "reviewed_by": ev["reviewer"], "page_sha256": ev["page_sha256"]}
    if flow:
        row["observation_ref"] = ev["observation_ref"]
    return row


def desired_rows(store: Store, vendor: dict) -> list[dict]:
    out = [overlay_row(vendor, e) for e in store.evidence(vendor["vendor_id"], "accepted")
           if e.get("origin") != "fixture" and not e.get("negated")]
    return sorted(out, key=lambda r: r["liveosint_id"])


def _snapshot(vendor, rows, ledger, base, trials, flow_discount):
    from ..engine import vendor_state
    s = vendor_state(vendor, rows, ledger, base, trials, flow_discount)
    return {"p_ai": s["p_ai"], "allowance": s["allowance"], "balance": s["balance"], "state": s["state"], "edges_from_osint": 0}


def sync_vendor(store: Store, vendor: dict, base_rows: list[dict], *, overlay_path=None, ledger=None, base: float = BASE, trials: int = 3000,
                flow_discount: float = AI_USE_FLOW_DISCOUNT, actor: str = "analyst", now: float | None = None, dry_run: bool = False) -> dict:
    """Apply reviewed evidence to the evidence overlay. Returns before/after budget numbers and whether anything changed."""
    vid, path = vendor["vendor_id"], (overlay_path or OVERLAY)
    overlay = _read(path, {})
    current = list(overlay.get(vid, []))
    mine_before = {r["liveosint_id"]: r for r in current if r.get("liveosint_id")}
    foreign = [r for r in current if not r.get("liveosint_id")]
    want = desired_rows(store, vendor)
    after_rows_overlay = foreign + want
    changed = json.dumps(sorted(mine_before.values(), key=lambda r: r["liveosint_id"]), sort_keys=True) != json.dumps(want, sort_keys=True)
    before = _snapshot(vendor, base_rows + current, ledger, base, trials, flow_discount)
    after = before if not changed else _snapshot(vendor, base_rows + after_rows_overlay, ledger, base, trials, flow_discount)
    added = sorted(set(r["liveosint_id"] for r in want) - set(mine_before))
    removed = sorted(set(mine_before) - set(r["liveosint_id"] for r in want))
    out = {"changed": changed, "dry_run": dry_run, "before": before, "after": after, "added": added, "removed": removed,
           "accepted_rows": len(want), "audit_hash": None}
    if not changed or dry_run:
        return out
    overlay[vid] = after_rows_overlay
    _write(path, overlay)
    if ledger is not None:
        digest = hashlib.sha256(json.dumps([r["liveosint_id"] for r in want]).encode()).hexdigest()[:12]
        meta = {"decision": "EVIDENCE_SYNC", "kind": "osint_review_sync", "actor": actor, "synthetic": False, "added": added, "removed": removed,
                "p_ai_before": round(before["p_ai"], 4), "p_ai_after": round(after["p_ai"], 4),
                "allowance_before": round(before["allowance"], 2), "allowance_after": round(after["allowance"], 2), "evidence_set": digest}
        out["audit_hash"] = ledger.append(vendor["vendor_name"], f"osint-sync:{digest}", 0.0, after["balance"], after["balance"], meta, ts=now)
    return out
