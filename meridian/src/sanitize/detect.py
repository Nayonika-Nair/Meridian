"""Text detection, precedence resolution and validated actions (A8, A9).

``scan_text`` turns one text unit (a PDF page, a Word paragraph, an OCR line group ...) into
resolved findings:

  1. run every enabled rule for the format (bounded - see rules.evaluate_rule)
  2. apply confidence gating (mandatory rules fail closed) and allow/deny lists
     (allow-lists never apply to hard-block rules)
  3. resolve overlaps deterministically: highest precedence wins, overlapping lower-precedence
     findings are folded into the same region (``also_matched``) so no part of a sensitive
     span is left behind
  4. validate the requested action against what the winning rule allows
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .policy import Policy
from .regions import Region, keyed_hash, make_region_id, text_shape
from .rules import (ACTIONS, STRICTNESS, SEVERITY, EvaluationBudgetExceeded, Rule, RuleRegistry, evaluate_rule)

SHORT = {"GOV_ID": "ID", "CARD": "CARD", "SECRET": "SECRET", "BANK": "BANK", "EMAIL": "EMAIL", "PHONE": "PHONE",
         "ADDRESS": "ADDRESS", "PERSON": "NAME", "DOB": "DOB", "EMP_ID": "EMPID", "FACE": "FACE"}


def norm_value(v: str) -> str:
    return "".join(v.split()).lower()


@dataclass
class Finding:
    rule: Rule
    start: int
    end: int
    confidence: float
    raw: str = field(repr=False)
    action: str = "NONE"
    reason: str = ""
    status: str = "pending"                  # pending | review_required | not_applicable
    also_matched: set[str] = field(default_factory=set)
    hard_block: bool = False
    severity: str = "HIGH"


@dataclass
class DetectContext:
    registry: RuleRegistry
    policy: Policy
    doc_id: str                              # document sha256 (or caller supplied id)
    salt: bytes
    warnings: list[str] = field(default_factory=list)
    deadline: object = None
    store_raw_text: bool = False

    def warn(self, msg: str) -> None:
        if msg not in self.warnings:
            self.warnings.append(msg)


# --------------------------------------------------------------------------- replacement text
def _alnum_mask(value: str, keep_last: int) -> str:
    alnum_idx = [i for i, ch in enumerate(value) if ch.isalnum()]
    k = min(keep_last, len(alnum_idx) // 3)       # never reveal more than a third of a value
    keep = set(alnum_idx[len(alnum_idx) - k:]) if k > 0 else set()
    return "".join(ch if (not ch.isalnum() or i in keep) else "*" for i, ch in enumerate(value))


def make_replacement(action: str, category: str, raw: str, keep_last: int = 4) -> str:
    tag = SHORT.get(category, "DATA")
    if action == "REDACT":
        return f"[REDACTED:{tag}]"
    if action == "MASK":
        if category == "EMAIL" and "@" in raw:
            local, _, dom = raw.partition("@")
            return f"{local[:1]}***@{dom}"
        if category in ("PERSON", "ADDRESS"):
            return f"[{tag}]"
        return _alnum_mask(raw, keep_last)
    if action == "GENERALISE":
        if category == "DOB":
            m = re.search(r"\d{4}", raw)
            return m.group(0) if m else "[DOB]"
        if category == "PHONE":
            return "(***) ***-****"
        if category == "EMAIL" and "@" in raw:
            return "*@" + raw.split("@", 1)[1]
        return f"[{tag}]"
    return raw


# --------------------------------------------------------------------------- action validation
def resolve_action(rule: Rule, policy: Policy, ctx: DetectContext | None = None) -> str:
    """Validated action for a rule. Policy may tighten but never loosen mandatory rules."""
    requested = (policy.rule_actions.get(rule.id) or policy.category_actions.get(rule.category)
                 or rule.default_action)
    if policy.destination_risk in ("HIGH", "CRITICAL") and rule.severity in ("HIGH", "CRITICAL"):
        requested = "REDACT" if "REDACT" in rule.allowed_actions else requested
    if requested not in ACTIONS:
        if ctx:
            ctx.warn(f"invalid_action:{rule.id}:{requested}")
        requested = rule.default_action
    if requested not in rule.allowed_actions:
        strictest = max(rule.allowed_actions, key=lambda a: STRICTNESS[a])
        if ctx:
            ctx.warn(f"action_not_allowed:{rule.id}:{requested}->{strictest}")
        requested = strictest
    if rule.hard_block:
        requested = "REDACT"
    return requested


# --------------------------------------------------------------------------- scanning
def scan_text(ctx: DetectContext, text: str, fmt: str, *, skip_rules: set[str] | None = None) -> list[Finding]:
    policy, registry = ctx.policy, ctx.registry
    allow = {norm_value(v) for v in policy.allowlist}
    cands: list[Finding] = []
    for rule in registry.for_format(fmt):
        if rule.kind == "face" or (skip_rules and rule.id in skip_rules):
            continue
        if rule.kind == "literal":
            matches = evaluate_rule(rule, text, deadline=ctx.deadline, literals=policy.denylist)
        else:
            try:
                matches = evaluate_rule(rule, text, deadline=ctx.deadline)
            except EvaluationBudgetExceeded:
                ctx.warn(f"rule_timeout:{rule.id}")
                if rule.hard_block:
                    raise
                continue
        for m in matches:
            f = Finding(rule=rule, start=m.start, end=m.end, confidence=m.confidence, raw=m.value,
                        hard_block=rule.hard_block, severity=rule.severity)
            if norm_value(m.value) in allow and rule.kind != "literal":
                if rule.hard_block:
                    ctx.warn(f"allowlist_ignored_hard_block:{rule.id}")
                else:
                    f.action, f.reason, f.status = "ALLOWLISTED", "value on allow-list", "not_applicable"
                    cands.append(f)
                    continue
            below = f.confidence < rule.min_confidence
            if below and not rule.hard_block:
                if policy.review_low_confidence:
                    f.action, f.reason, f.status = "FLAG_REVIEW", f"confidence {f.confidence:.2f} < threshold {rule.min_confidence:.2f}", "review_required"
                else:
                    f.action, f.reason, f.status = "NONE", "below confidence threshold", "not_applicable"
            else:
                f.action = resolve_action(rule, policy, ctx)
                f.reason = ("mandatory rule applied despite low confidence (fail closed)" if below else
                            "mandatory hard-block rule" if rule.hard_block else "rule matched")
            cands.append(f)
    return _resolve_overlaps(cands, text)


def _resolve_overlaps(cands: list[Finding], text: str) -> list[Finding]:
    order = sorted(cands, key=lambda f: (f.rule.precedence, -SEVERITY.index(f.rule.severity), -(f.end - f.start), f.start, f.rule.id))
    accepted: list[Finding] = []
    for c in order:
        host = next((a for a in accepted if a.start < c.end and c.start < a.end), None)
        if host is None:
            accepted.append(c)
            continue
        if c.rule.id == host.rule.id and (c.start, c.end) == (host.start, host.end):
            continue                                      # exact duplicate
        # allow-listed or non-acting overlaps never widen an acting finding
        if c.action in ("ALLOWLISTED", "NONE") and host.action not in ("ALLOWLISTED", "NONE"):
            continue
        host.also_matched.add(c.rule.id)
        host.also_matched |= c.also_matched
        host.start, host.end = min(host.start, c.start), max(host.end, c.end)
        host.hard_block = host.hard_block or c.hard_block
        if SEVERITY.index(c.rule.severity) > SEVERITY.index(host.severity):
            host.severity = c.rule.severity
        if STRICTNESS.get(c.action, -1) > STRICTNESS.get(host.action, -1):
            host.action = c.action
        if c.status == "pending" or host.status == "not_applicable":
            host.status = c.status if host.status == "not_applicable" else host.status
        host.confidence = max(host.confidence, c.confidence)
        host.raw = text[host.start:host.end]
    # a widened finding may now touch another accepted one
    accepted.sort(key=lambda f: f.start)
    merged: list[Finding] = []
    for f in accepted:
        if merged and f.start < merged[-1].end:
            m = merged[-1]
            m.also_matched.add(f.rule.id)
            m.end = max(m.end, f.end)
            m.raw = text[m.start:m.end]
            if STRICTNESS.get(f.action, -1) > STRICTNESS.get(m.action, -1):
                m.action = f.action
            m.hard_block = m.hard_block or f.hard_block
        else:
            merged.append(f)
    for f in merged:
        f.also_matched.discard(f.rule.id)
    return merged


# --------------------------------------------------------------------------- region construction
def finding_to_region(ctx: DetectContext, f: Finding, *, unit_id: str, space: str, rects=None, anchor: dict | None = None,
                      source_method: str, chain: list[dict] | None = None, page_size=None) -> Region:
    anchor = dict(anchor or {})
    anchor.update({"start": f.start, "end": f.end})
    key = f"{f.start}-{f.end}" if not rects else "|".join(f"{round(r[0], 1)},{round(r[1], 1)},{round(r[2], 1)},{round(r[3], 1)}" for r in rects[:1]) + f"#{f.start}"
    status = {"pending": "pending", "review_required": "review_required", "not_applicable": "not_applicable"}[f.status]
    reg = Region(
        region_id=make_region_id(ctx.doc_id, unit_id, f.rule.id, key), document_id=ctx.doc_id, unit_id=unit_id,
        space=space, rects=list(rects or []), anchor=anchor, category=f.rule.category, rule_id=f.rule.id,
        entity_type=f.rule.entity_type, severity=f.severity, hard_block=f.hard_block, confidence=f.confidence,
        source_method=source_method, action=f.action, verification_status=status, reason=f.reason,
        text_hash=keyed_hash(ctx.salt, f.raw), text_shape=text_shape(f.raw), text_len=len(f.raw),
        also_matched=sorted(f.also_matched), chain=chain or [], page_size=page_size, raw=f.raw)
    if f.action in ("REDACT", "MASK", "GENERALISE"):
        reg.replacement = make_replacement(f.action, f.rule.category, f.raw, ctx.policy.mask_keep_last)
    return reg
