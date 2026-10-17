"""Action space + the validator that makes constraints HARD (not reward penalties).

The policy engine (src/redaction.py::field_action) defines the FLOOR for every field. A strategy may only keep a field
at the floor or tighten it (KEEP < MASK < REDACT), and may never REDACT a field the task requires. `validate_actions`
recomputes the floor itself, so it does not trust the code that produced the actions. Strategies that fail validation are
removed from the action mask BEFORE the bandit sees them: they can be neither explored nor exploited.

LOOSENING strategies exist only for the *unconstrained ablation* that shows why masking is needed; they are never
available to the production advisor.
"""
from __future__ import annotations

from ..config import HARD_BLOCK_TYPES, WEIGHTS, p_ai_tier
from ..redaction import ENTITY_FIELD, field_action, load_policy

RANK = {"KEEP": 0, "MASK": 1, "REDACT": 2}
SAFE_STRATEGIES = ["POLICY", "MASK_SENSITIVE", "DROP_OPTIONAL_PII", "DROP_OPTIONAL_PII_MASK", "DROP_ALL_OPTIONAL_MASK"]
LOOSE_STRATEGIES = ["KEEP_REQUIRED_CLEAR", "KEEP_OPTIONAL", "KEEP_EVERYTHING"]
ALL_STRATEGIES = SAFE_STRATEGIES + LOOSE_STRATEGIES
DESCRIPTIONS = {
    "POLICY": "Exactly what the rule-based policy engine does.",
    "MASK_SENSITIVE": "Mask every PII field that the policy would send in the clear.",
    "DROP_OPTIONAL_PII": "Drop optional fields that carry PII.",
    "DROP_OPTIONAL_PII_MASK": "Drop optional PII fields and mask the rest of the PII.",
    "DROP_ALL_OPTIONAL_MASK": "Drop every optional field and mask all PII (maximum tightening that keeps the task possible).",
    "KEEP_REQUIRED_CLEAR": "ABLATION ONLY: send required fields in the clear (loosens the floor).",
    "KEEP_OPTIONAL": "ABLATION ONLY: keep optional fields regardless of destination risk.",
    "KEEP_EVERYTHING": "ABLATION ONLY: send everything present, including fields the task does not need.",
}


def floor_actions(fields: list[str], task: dict, tier: str, policy: dict, extra_redact=()) -> dict:
    return {f: field_action(f, task, tier, policy, extra_redact)[0] for f in fields}


def _status(f, task):
    return "required" if f in task["required"] else "optional" if f in task["optional"] else "not required"


def apply_strategy(name: str, fields: list[str], task: dict, tier: str, policy: dict, extra_redact=()) -> dict:
    base = floor_actions(fields, task, tier, policy, extra_redact)
    out = dict(base)
    ent = {f: policy["fields"][f].get("entity") for f in fields}
    for f in fields:
        st, e = _status(f, task), ent[f]
        pii = bool(e) and WEIGHTS.get(e, 0) > 0
        if name == "POLICY":
            continue
        if name in ("MASK_SENSITIVE", "DROP_OPTIONAL_PII_MASK", "DROP_ALL_OPTIONAL_MASK") and pii and out[f] == "KEEP":
            out[f] = "MASK"
        if name in ("DROP_OPTIONAL_PII", "DROP_OPTIONAL_PII_MASK") and st == "optional" and pii:
            out[f] = "REDACT"
        if name == "DROP_ALL_OPTIONAL_MASK" and st == "optional":
            out[f] = "REDACT"
        if name == "KEEP_REQUIRED_CLEAR" and st == "required":
            out[f] = "KEEP"
        if name == "KEEP_OPTIONAL" and st == "optional":
            out[f] = "KEEP"
        if name == "KEEP_EVERYTHING":
            out[f] = "KEEP"
    return out


def validate_actions(actions: dict, fields: list[str], task: dict, tier: str, policy: dict, extra_redact=()) -> list[str]:
    """Hard-constraint check. Returns violation codes (empty = valid). Recomputes the floor independently."""
    floor = floor_actions(fields, task, tier, policy, extra_redact)
    ent = {f: policy["fields"][f].get("entity") for f in fields}
    v = []
    for f in fields:
        a = actions.get(f)
        if a not in RANK:
            v.append(f"unknown_action:{f}")
            continue
        if _status(f, task) == "not required" and a != "REDACT":
            v.append(f"minimisation:{f}")
        if ent[f] in extra_redact and a != "REDACT":
            v.append(f"approved_policy:{f}")
        if ent[f] in HARD_BLOCK_TYPES and RANK[a] < RANK["MASK"]:
            v.append(f"hard_block_in_clear:{f}")
        if RANK[a] < RANK[floor[f]]:
            v.append(f"below_policy_floor:{f}")
        if _status(f, task) == "required" and a == "REDACT" and floor[f] != "REDACT":
            v.append(f"required_field_removed:{f}")
    return v


def allowed_strategies(fields, task, tier, policy, extra_redact=(), constrained: bool = True) -> list[str]:
    """The action mask. In constrained mode only strategies whose plan passes validation survive (POLICY always does)."""
    pool = SAFE_STRATEGIES if constrained else ALL_STRATEGIES
    ok = []
    for s in pool:
        a = apply_strategy(s, fields, task, tier, policy, extra_redact)
        if constrained and validate_actions(a, fields, task, tier, policy, extra_redact):
            continue
        ok.append(s)
    return ok


__all__ = ["SAFE_STRATEGIES", "LOOSE_STRATEGIES", "ALL_STRATEGIES", "DESCRIPTIONS", "apply_strategy", "validate_actions", "allowed_strategies",
           "floor_actions", "load_policy", "ENTITY_FIELD", "p_ai_tier"]
