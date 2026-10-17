"""Synthetic, offline environment. NOTHING here reads real documents, vendors or the ledger.

A *session* is a vendor with a leak allowance receiving a stream of synthetic documents. At each step the agent picks a
redaction strategy; the AUTHORITATIVE decision logic (budget.decision) then prices and gates the plan exactly as in
production. Documents for a given (seed, step) are identical across policies, so policies are compared on the same inputs.

Reward and utility parameters are ASSUMPTIONS, exposed in `RewardConfig` and reported with every result.
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass

from ..budget import decision, doc_cost
from ..config import BASE, MASK_RESIDUAL, WEIGHTS, HARD_BLOCK_TYPES, p_ai_tier
from ..redaction import ENTITY_FIELD, load_policy
from .strategies import apply_strategy, validate_actions

TASKS = ["statement_summary", "payment_routing", "statement_mailing", "tax_document", "batch_scheduling"]
TASK_PROB = [0.28, 0.24, 0.22, 0.06, 0.20]
CRIT = [0.50, 0.75, 1.00, 1.25]
OUTCOME_UTILITY = {"AUTO": 1.0, "STRICT": 1.0, "REVIEW": 0.4, "BLOCK": 0.0}     # assumption: REVIEW delays but does not lose the task
FIELD_UTILITY = {"required": 1.0, "optional": 0.3}
ACTION_UTILITY = {"KEEP": 1.0, "MASK": 0.7, "REDACT": 0.0}                       # assumption: tokenised values are mostly usable
PREV = ["NONE", "AUTO", "STRICT", "REVIEW", "BLOCK"]


@dataclass(frozen=True)
class RewardConfig:
    cost_scale: float = 50.0          # leak points that count as "1 unit" of cost
    cost_weight: float = 0.30         # lambda: value of one cost_scale of leakage relative to completing the task
    session_len: int = 15


def _step_rng(seed: int, step: int) -> random.Random:
    return random.Random((seed * 1_000_003 + step * 7919) & 0xFFFFFFFF)


def new_session(seed: int, cfg: RewardConfig = RewardConfig()) -> dict:
    r = random.Random(seed ^ 0x5EED)
    p_ai = r.choice([0.02, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95]) * r.uniform(0.9, 1.05)
    p_ai = min(0.99, p_ai)
    crit = r.choice(CRIT)
    allowance = BASE * (1 - p_ai) * crit
    return {"seed": seed, "p_ai": p_ai, "criticality_factor": crit, "allowance": allowance,
            "balance": allowance * r.uniform(0.3, 1.0), "prev": "NONE", "step": 0, "cfg": cfg}


def draw_document(seed: int, step: int, policy: dict) -> tuple[str, dict]:
    r = _step_rng(seed, step)
    task_key = r.choices(TASKS, TASK_PROB)[0]
    t = policy["tasks"][task_key]
    fields = list(t["required"])
    if r.random() < 0.08 and fields:
        fields.remove(r.choice(fields))                                          # sometimes a required field is missing
    fields += [f for f in t["optional"] if r.random() < 0.7]
    extras = [f for f in policy["fields"] if f not in t["required"] and f not in t["optional"]]
    fields += [f for f in extras if r.random() < (0.35 if policy["fields"][f].get("entity") else 0.1)]
    ents = []
    q = r.choice([0.99, 0.96, 0.93, 0.85, 0.7])                                  # detector quality of this document
    for f in fields:
        e = policy["fields"][f].get("entity")
        if e:
            for _ in range(r.choice([1, 1, 2, 3])):
                ents.append({"type": e, "confidence": round(min(0.995, max(0.3, r.gauss(q, 0.03))), 4)})
    return task_key, {"fields_present": list(dict.fromkeys(fields)), "entities": ents}


def score_actions(actions: dict, doc: dict, task_key: str, p_ai: float, policy: dict, session: dict) -> dict:
    """Authoritative pricing + gating of a set of per-field actions (same maths as redaction.plan_redaction)."""
    cfg: RewardConfig = session["cfg"]
    task, e2f = policy["tasks"][task_key], ENTITY_FIELD(policy)
    eff, unresolved = [], False
    for e in doc["entities"]:
        a = actions.get(e2f.get(e["type"]), "REDACT")
        conf = float(e["confidence"])
        ec = {"REDACT": conf, "MASK": max(0.0, 1 - MASK_RESIDUAL), "KEEP": 0.0}[a]
        eff.append({"type": e["type"], "confidence": ec})
        if e["type"] in HARD_BLOCK_TYPES and (a != "REDACT" or conf < 0.95):
            unresolved = True
    cost = doc_cost(eff, p_ai)
    dec = decision(session["balance"], session["allowance"], cost, unresolved, None)
    denom = sum(FIELD_UTILITY["required"] for _ in task["required"]) + sum(FIELD_UTILITY["optional"] for _ in task["optional"])
    num = 0.0
    for f in doc["fields_present"]:
        w = FIELD_UTILITY["required"] if f in task["required"] else FIELD_UTILITY["optional"] if f in task["optional"] else 0.0
        num += w * ACTION_UTILITY[actions.get(f, "REDACT")]
    field_u = (num / denom) if denom else 1.0
    utility = field_u * OUTCOME_UTILITY[dec]
    reward = utility - cfg.cost_weight * cost / cfg.cost_scale
    spends = dec in ("AUTO", "STRICT")
    return {"cost": cost, "decision": dec, "utility": utility, "field_utility": field_u, "reward": reward, "spends": spends}


def context(session: dict, doc: dict, task_key: str, policy: dict) -> dict:
    """Everything the agent may condition on. Contains no document text, only bucketable counts and decisions."""
    tier = p_ai_tier(session["p_ai"])
    task = policy["tasks"][task_key]
    e2f = ENTITY_FIELD(policy)
    sens = {e["type"] for e in doc["entities"] if WEIGHTS.get(e["type"], 0) >= 8}
    sc = [e["confidence"] for e in doc["entities"] if WEIGHTS.get(e["type"], 0) >= 8]
    mean_conf = sum(sc) / len(sc) if sc else 1.0
    frac = session["balance"] / session["allowance"] if session["allowance"] else 0.0
    return {"task": task_key, "tier": tier, "sensitive": "+".join(sorted(sens)) or "none",
            "conf_bucket": "low" if mean_conf < 0.9 else "mid" if mean_conf < 0.96 else "high",
            "budget_bucket": "<5%" if frac < 0.05 else "<20%" if frac < 0.2 else "<50%" if frac < 0.5 else ">=50%",
            "required_present": all(f in doc["fields_present"] for f in task["required"]), "prev": session["prev"],
            "fields": list(doc["fields_present"]), "tier_raw": tier}


def step(session: dict, strategy: str, doc: dict, task_key: str, policy: dict, extra_redact=()) -> dict:
    tier = p_ai_tier(session["p_ai"])
    task = policy["tasks"][task_key]
    actions = apply_strategy(strategy, doc["fields_present"], task, tier, policy, extra_redact)
    violations = validate_actions(actions, doc["fields_present"], task, tier, policy, extra_redact)
    res = score_actions(actions, doc, task_key, session["p_ai"], policy, session)
    res["violations"] = violations
    if res["spends"]:
        session["balance"] = max(0.0, session["balance"] - res["cost"])
    session["prev"] = res["decision"]
    session["step"] += 1
    return res


def reward_config_dict(cfg: RewardConfig) -> dict:
    return {**asdict(cfg), "outcome_utility": OUTCOME_UTILITY, "field_utility": FIELD_UTILITY, "action_utility": ACTION_UTILITY}
