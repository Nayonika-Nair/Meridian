"""Advisory-only integration with the Document Gate. The rule-based decision stays authoritative.

advise() never modifies the plan, decision, receipt or ledger of the gate. Its output is a recommendation plus a log
line (bucketed features only: no document text, no entity values, no file names) written to a JSONL file.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from ..config import ROOT, p_ai_tier
from ..redaction import load_policy
from .bandit import MODEL_VERSION, ContextualBandit
from .env import RewardConfig, context, score_actions
from .strategies import allowed_strategies, apply_strategy, validate_actions

MODEL_PATH = ROOT / "data" / "rl_model.json"
LOG_PATH = ROOT / "ledger" / "rl_advice.jsonl"


def load_model(path=MODEL_PATH):
    try:
        return ContextualBandit.load(path)
    except (FileNotFoundError, ValueError, json.JSONDecodeError, KeyError):
        return None


def advise(state: dict, doc: dict, task_key: str, *, model=None, policy=None, extra_redact=(), authoritative: dict | None = None,
           log_path=None, log: bool = True, now=None) -> dict:
    """state: engine.vendor_state() dict (needs p_ai, balance, allowance). authoritative: engine.evaluate_document() result (optional)."""
    log_path = log_path or os.environ.get("MERIDIAN_RL_LOG") or LOG_PATH
    policy = policy or load_policy()
    model = model if model is not None else load_model()
    if model is None:
        return {"available": False, "reason": "No trained advisory model found. Run scripts/train_rl.py.", "authoritative_decision": (authoritative or {}).get("decision")}
    sess = {"p_ai": state["p_ai"], "balance": state["balance"], "allowance": state["allowance"], "prev": "NONE", "cfg": RewardConfig()}
    ctx = context(sess, doc, task_key, policy)
    tier = p_ai_tier(state["p_ai"])
    task = policy["tasks"][task_key]
    allowed = allowed_strategies(doc["fields_present"], task, tier, policy, extra_redact, constrained=True)
    pick, vals = model.select(ctx, allowed, explore=False)
    actions = apply_strategy(pick, doc["fields_present"], task, tier, policy, extra_redact)
    if validate_actions(actions, doc["fields_present"], task, tier, policy, extra_redact):          # defence in depth: never surface an invalid plan
        pick = "POLICY"
        actions = apply_strategy("POLICY", doc["fields_present"], task, tier, policy, extra_redact)
    scored = score_actions(actions, doc, task_key, state["p_ai"], policy, sess)
    auth = (authoritative or {}).get("decision")
    out = {"available": True, "advisory_only": True, "model_version": MODEL_VERSION, "model_steps": model.t, "recommended_strategy": pick,
           "field_actions": actions, "allowed_strategies": allowed, "values": vals, "projected_decision": scored["decision"],
           "projected_cost": scored["cost"], "authoritative_decision": auth, "differs_from_policy": pick != "POLICY",
           "note": "Recommendation only. The rule-based decision above is what the gate enforces."}
    if log:
        rec = {"ts": time.time() if now is None else now, "model_version": MODEL_VERSION, "model_steps": model.t,
               "context": {k: ctx[k] for k in ("task", "tier", "sensitive", "conf_bucket", "budget_bucket", "required_present")},
               "n_fields": len(doc["fields_present"]), "n_entities": len(doc["entities"]), "recommended": pick, "allowed": allowed,
               "projected_decision": scored["decision"], "authoritative_decision": auth}
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
    return out
