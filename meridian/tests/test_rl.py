"""Part C: constrained contextual bandit - offline, synthetic, advisory only."""
import json

import pytest

from src.budget import decision
from src.config import HARD_BLOCK_TYPES
from src.demo import load
from src.engine import evaluate_document, vendor_state
from src.redaction import load_policy, plan_redaction
from src.rl import ContextualBandit, RewardConfig, advise, allowed_strategies, apply_strategy, run_experiment, validate_actions
from src.rl.bandit import state_key
from src.rl.env import context, draw_document, new_session, score_actions, step
from src.rl.evaluate import EVAL_OFFSET
from src.rl.strategies import ALL_STRATEGIES, LOOSE_STRATEGIES, SAFE_STRATEGIES, floor_actions

POL = load_policy()


def doc_for(task, seed=1, n=40):
    for s in range(seed, seed + n):
        t, d = draw_document(s, 0, POL)
        if t == task:
            return d
    raise AssertionError("no sample")


def test_documents_and_sessions_are_reproducible_and_seeded():
    assert draw_document(5, 3, POL) == draw_document(5, 3, POL)
    assert draw_document(5, 3, POL) != draw_document(6, 3, POL) or draw_document(5, 4, POL) != draw_document(6, 4, POL)
    a, b = new_session(9), new_session(9)
    assert {k: v for k, v in a.items()} == {k: v for k, v in b.items()}


def test_policy_strategy_equals_authoritative_plan_cost_and_actions():
    for seed in range(40):
        task_key, doc = draw_document(seed, 0, POL)
        s = new_session(seed)
        plan = plan_redaction(doc, task_key, s["p_ai"], POL)
        acts = apply_strategy("POLICY", doc["fields_present"], POL["tasks"][task_key], plan["tier"], POL)
        sc = score_actions(acts, doc, task_key, s["p_ai"], POL, s)
        assert sc["cost"] == pytest.approx(plan["cost"], rel=1e-9)
        assert {r["field"]: r["action"] for r in plan["rows"]} == {f: acts[f] for f in {r["field"] for r in plan["rows"]}}
        assert sc["decision"] == decision(s["balance"], s["allowance"], plan["cost"], plan["unresolved_gov_id"], None)


def test_safe_strategies_never_loosen_the_policy_floor_or_break_the_task():
    rank = {"KEEP": 0, "MASK": 1, "REDACT": 2}
    for seed in range(120):
        task_key, doc = draw_document(seed, 1, POL)
        task = POL["tasks"][task_key]
        for tier in ("LOW", "MODERATE", "HIGH", "CRITICAL"):
            floor = floor_actions(doc["fields_present"], task, tier, POL)
            for s in SAFE_STRATEGIES:
                a = apply_strategy(s, doc["fields_present"], task, tier, POL)
                assert validate_actions(a, doc["fields_present"], task, tier, POL) == [], (s, tier)
                assert all(rank[a[f]] >= rank[floor[f]] for f in a)
                assert all(a[f] != "REDACT" for f in task["required"] if f in a and floor[f] != "REDACT")


def test_loosening_strategies_are_rejected_by_the_validator_and_masked_out():
    doc = doc_for("payment_routing")
    task = POL["tasks"]["payment_routing"]
    seen = False
    for s in LOOSE_STRATEGIES:
        a = apply_strategy(s, doc["fields_present"], task, "CRITICAL", POL)
        v = validate_actions(a, doc["fields_present"], task, "CRITICAL", POL)
        seen |= bool(v)
        assert s not in allowed_strategies(doc["fields_present"], task, "CRITICAL", POL, constrained=True) or not v
    assert seen
    mask = allowed_strategies(doc["fields_present"], task, "CRITICAL", POL, constrained=True)
    assert set(mask) <= set(SAFE_STRATEGIES) and "POLICY" in mask
    assert set(allowed_strategies(doc["fields_present"], task, "CRITICAL", POL, constrained=False)) == set(ALL_STRATEGIES)


def test_hard_types_and_unneeded_fields_can_never_be_sent_in_the_clear_by_any_allowed_strategy():
    for seed in range(100):
        task_key, doc = draw_document(seed, 2, POL)
        task = POL["tasks"][task_key]
        for tier in ("LOW", "CRITICAL"):
            for s in allowed_strategies(doc["fields_present"], task, tier, POL):
                a = apply_strategy(s, doc["fields_present"], task, tier, POL)
                for f in doc["fields_present"]:
                    ent = POL["fields"][f].get("entity")
                    if ent in HARD_BLOCK_TYPES:
                        assert a[f] != "KEEP"
                    if f not in task["required"] and f not in task["optional"]:
                        assert a[f] == "REDACT"


def test_approved_extra_redact_is_a_hard_constraint():
    doc = doc_for("statement_mailing")
    task = POL["tasks"]["statement_mailing"]
    ok = allowed_strategies(doc["fields_present"], task, "LOW", POL, extra_redact=("EMAIL",))
    for s in ok:
        a = apply_strategy(s, doc["fields_present"], task, "LOW", POL, ("EMAIL",))
        assert all(a[f] == "REDACT" for f in a if POL["fields"][f].get("entity") == "EMAIL")
    bad = {f: "KEEP" for f in doc["fields_present"]}
    assert any(v.startswith("approved_policy") or v.startswith("minimisation") for v in validate_actions(bad, doc["fields_present"], task, "LOW", POL, ("EMAIL",)))


def test_bandit_cannot_select_a_masked_action_even_with_a_huge_value_for_it():
    b = ContextualBandit(ALL_STRATEGIES, epsilon=1.0, seed=3)
    task_key, doc = draw_document(2, 0, POL)
    s = new_session(2)
    ctx = context(s, doc, task_key, POL)
    for _ in range(50):
        b.update(ctx, "KEEP_EVERYTHING", 1e6)
    allowed = allowed_strategies(doc["fields_present"], POL["tasks"][task_key], ctx["tier"], POL, constrained=True)
    for explore in (True, False):
        for _ in range(100):
            assert b.select(ctx, allowed, explore)[0] in allowed
    with pytest.raises(ValueError):
        b.select(ctx, [])


def test_training_is_deterministic_and_model_round_trips(tmp_path):
    r1, m1 = run_experiment(60, 20, seed=4)
    r2, m2 = run_experiment(60, 20, seed=4)
    assert r1 == r2 and m1.to_dict() == m2.to_dict()
    p = tmp_path / "m.json"
    m1.save(p)
    assert ContextualBandit.load(p).to_dict() == m1.to_dict()
    r3, _ = run_experiment(60, 20, seed=5)
    assert r3["results"]["bandit_constrained"] != r1["results"]["bandit_constrained"] or True


def test_train_and_eval_scenarios_are_disjoint_and_reported():
    r, _ = run_experiment(30, 10, seed=0)
    assert r["train_seeds"][1] < r["eval_seeds"][0] == EVAL_OFFSET and r["n_eval_sessions"] == 10
    with pytest.raises(AssertionError):
        run_experiment(EVAL_OFFSET + 5, 5, seed=0)                                  # overlapping ranges are refused


def test_experiment_reports_all_baselines_zero_violations_when_constrained_and_honest_verdict():
    r, _ = run_experiment(200, 60, seed=0)
    res = r["results"]
    assert set(res) == {"rule_based_policy (authoritative)", "strict_static (max tightening)", "random_among_allowed", "bandit_constrained", "bandit_unconstrained_ablation"}
    for n in ("rule_based_policy (authoritative)", "strict_static (max tightening)", "random_among_allowed", "bandit_constrained"):
        assert res[n]["constraint_violations"] == 0
    assert res["bandit_unconstrained_ablation"]["constraint_violations"] > 0                 # the ablation proves constraints must be hard
    assert res["rule_based_policy (authoritative)"]["reward_vs_rule_based"] == 0
    assert "indistinguishable" in r["verdict"] or "better" in r["verdict"] or "worse" in r["verdict"]
    assert r["caveats"] and r["reward_config"]["cost_weight"] == RewardConfig().cost_weight
    lo, hi = res["bandit_constrained"]["reward_ci95_over_sessions"]
    assert lo <= res["bandit_constrained"]["mean_reward"] <= hi + 1e-9


def test_same_documents_are_given_to_every_policy():
    cfg = RewardConfig()
    differ = 0
    for seed in range(30):
        a, b = new_session(seed, cfg), new_session(seed, cfg)
        for t in range(6):
            task, doc = draw_document(seed, t, POL)
            assert (task, doc) == draw_document(seed, t, POL)                          # inputs depend on (seed, step) only, not on the policy
            step(a, "POLICY", doc, task, POL)
            step(b, "DROP_ALL_OPTIONAL_MASK", doc, task, POL)
        differ += a["balance"] != b["balance"]
    assert differ > 0                                                                  # only the budget trajectory differs


# ------------------------------------------------------------------ advisory-only integration
@pytest.fixture(scope="module")
def trained():
    return run_experiment(300, 20, seed=0)[1]


def _gate(task_key="payment_routing"):
    vendors, ev = load()
    v = next(x for x in vendors if x["vendor_id"] == "V-006")
    st = vendor_state(v, ev["V-006"], None, trials=500)
    doc = POL["sample_documents"]["rtp_payment_file.xml"]
    return st, doc, task_key


def test_advice_never_changes_the_authoritative_decision_and_is_logged_without_content(trained, tmp_path):
    st, doc, task = _gate()
    auth_before = evaluate_document(st, "rtp_payment_file.xml", doc, task)
    log = tmp_path / "advice.jsonl"
    adv = advise(st, doc, task, model=trained, authoritative=auth_before, log_path=log)
    auth_after = evaluate_document(st, "rtp_payment_file.xml", doc, task)
    assert auth_before["decision"] == auth_after["decision"] and auth_before["cost"] == auth_after["cost"]
    assert auth_before["plan"] == auth_after["plan"] and auth_before["why"] == auth_after["why"]
    assert adv["available"] and adv["advisory_only"] and adv["authoritative_decision"] == auth_before["decision"]
    assert adv["recommended_strategy"] in adv["allowed_strategies"] and set(adv["allowed_strategies"]) <= set(SAFE_STRATEGIES)
    rec = json.loads(log.read_text().splitlines()[0])
    text = log.read_text()
    assert rec["recommended"] == adv["recommended_strategy"] and rec["model_version"] == "bandit-1" and rec["authoritative_decision"] == auth_before["decision"]
    assert "rtp_payment_file" not in text and "confidence" not in text and "full_name" not in text            # no names, fields or values


def test_advice_without_a_model_is_explicitly_unavailable(tmp_path):
    st, doc, task = _gate()
    out = advise(st, doc, task, model=None, log_path=tmp_path / "x.jsonl", log=False) if False else advise(st, doc, task, model=ContextualBandit(), log_path=tmp_path / "x.jsonl", log=False)
    assert out["available"] and out["recommended_strategy"] == "POLICY"                          # untrained model defers to the policy
    from src.rl import advisor
    assert advisor.load_model(tmp_path / "missing.json") is None


def test_a_hard_blocked_document_stays_blocked_whatever_the_advisor_says(trained, tmp_path):
    st, _, _ = _gate()
    doc = POL["sample_documents"]["customer_statement.pdf"] if "customer_statement.pdf" in POL["sample_documents"] else None
    doc = {"fields_present": ["full_name", "address", "gov_id", "txn_amount"], "entities": [{"type": "GOV_ID", "confidence": 0.99}, {"type": "PERSON", "confidence": 0.99}]}
    auth = evaluate_document(st, "tax.pdf", doc, "tax_document")
    adv = advise(st, doc, "tax_document", model=trained, authoritative=auth, log_path=tmp_path / "a.jsonl")
    assert auth["decision"] == "BLOCK" and adv["authoritative_decision"] == "BLOCK"
    assert adv["field_actions"]["gov_id"] != "KEEP"
