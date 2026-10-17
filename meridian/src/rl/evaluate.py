"""Train / evaluate with disjoint seed ranges and honest baselines.

train sessions: seeds 0..n_train-1          eval sessions: seeds EVAL_OFFSET..EVAL_OFFSET+n_eval-1   (disjoint, asserted)
Every policy is run on the SAME eval sessions (same documents per (seed, step)); only the budget trajectory differs.
"""
from __future__ import annotations

import random
import statistics

from ..redaction import load_policy
from .bandit import ContextualBandit, MODEL_VERSION
from .env import RewardConfig, context, draw_document, new_session, reward_config_dict, step
from .strategies import ALL_STRATEGIES, SAFE_STRATEGIES, allowed_strategies

EVAL_OFFSET = 100_000


def _run_session(seed, chooser, cfg, policy, constrained=True, learn=None):
    s = new_session(seed, cfg)
    rows = []
    for t in range(cfg.session_len):
        task_key, doc = draw_document(seed, t, policy)
        ctx = context(s, doc, task_key, policy)
        tier = ctx["tier"]
        allowed = allowed_strategies(doc["fields_present"], policy["tasks"][task_key], tier, policy, (), constrained)
        strat = chooser(ctx, allowed)
        r = step(s, strat, doc, task_key, policy)
        if learn is not None:
            learn(ctx, strat, r["reward"])
        rows.append({**{k: r[k] for k in ("reward", "cost", "utility", "decision")}, "violations": len(r["violations"]), "strategy": strat})
    return rows


def _summ(rows):
    n = len(rows)
    dec = {d: sum(1 for r in rows if r["decision"] == d) / n for d in ("AUTO", "STRICT", "REVIEW", "BLOCK")}
    return {"steps": n, "mean_reward": statistics.fmean(r["reward"] for r in rows), "mean_utility": statistics.fmean(r["utility"] for r in rows),
            "mean_cost": statistics.fmean(r["cost"] for r in rows), "decisions": dec, "constraint_violations": sum(r["violations"] for r in rows),
            "violating_steps": sum(1 for r in rows if r["violations"])}


def _ci(per_session, boot=400, seed=7):
    rng = random.Random(seed)
    n = len(per_session)
    means = sorted(statistics.fmean(rng.choice(per_session) for _ in range(n)) for _ in range(boot))
    return [means[int(0.025 * boot)], means[int(0.975 * boot) - 1]]


def run_experiment(n_train=1500, n_eval=400, seed=0, cfg: RewardConfig = RewardConfig(), epsilon=0.2) -> dict:
    policy = load_policy()
    train_seeds = list(range(n_train))
    eval_seeds = list(range(EVAL_OFFSET, EVAL_OFFSET + n_eval))
    assert not set(train_seeds) & set(eval_seeds)
    models = {}
    for name, constrained in (("constrained", True), ("unconstrained_ablation", False)):
        b = ContextualBandit(SAFE_STRATEGIES if constrained else ALL_STRATEGIES, epsilon=epsilon, seed=seed)
        for sd in train_seeds:
            _run_session(sd, lambda c, a: b.select(c, a, True)[0], cfg, policy, constrained, learn=b.update)
        models[name] = b
    rnd = random.Random(seed + 99)
    policies = {
        "rule_based_policy (authoritative)": (lambda c, a: "POLICY", True),
        "strict_static (max tightening)": (lambda c, a: "DROP_ALL_OPTIONAL_MASK" if "DROP_ALL_OPTIONAL_MASK" in a else "POLICY", True),
        "random_among_allowed": (lambda c, a: rnd.choice(a), True),
        "bandit_constrained": (lambda c, a: models["constrained"].select(c, a, False)[0], True),
        "bandit_unconstrained_ablation": (lambda c, a: models["unconstrained_ablation"].select(c, a, False)[0], False),
    }
    results = {}
    for name, (ch, constrained) in policies.items():
        per_sess, rows_all = [], []
        for sd in eval_seeds:
            rows = _run_session(sd, ch, cfg, policy, constrained)
            rows_all += rows
            per_sess.append(statistics.fmean(r["reward"] for r in rows))
        results[name] = {**_summ(rows_all), "reward_ci95_over_sessions": _ci(per_sess)}
    base = results["rule_based_policy (authoritative)"]["mean_reward"]
    for n, r in results.items():
        r["reward_vs_rule_based"] = r["mean_reward"] - base
    b = results["bandit_constrained"]
    verdict = ("Constrained bandit is " + ("better than" if b["reward_ci95_over_sessions"][0] > base else "worse than" if b["reward_ci95_over_sessions"][1] < base else "statistically indistinguishable from")
               + " the rule-based policy on mean reward (95% CI over eval sessions vs the baseline mean).")
    return {"model_version": MODEL_VERSION, "seed": seed, "n_train_sessions": n_train, "n_eval_sessions": n_eval, "train_seeds": [0, n_train - 1],
            "eval_seeds": [EVAL_OFFSET, EVAL_OFFSET + n_eval - 1], "reward_config": reward_config_dict(cfg), "results": results, "verdict": verdict,
            "caveats": ["Synthetic scenarios and assumed utility/cost weights: results say nothing about real documents or vendors.",
                        "The bandit is myopic and only tightens the authoritative policy; it cannot loosen any rule.",
                        "The unconstrained ablation exists to show constraint violations; it is never used by the advisor."]}, models["constrained"]
