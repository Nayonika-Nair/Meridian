"""Policy Lab: the advisory bandit. Shows the offline experiment honestly and lets an analyst see a recommendation beside the
authoritative rule-based decision. Nothing here can change what the Document Gate does."""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from ..rl import advise, load_model
from ..rl.advisor import MODEL_PATH
from ..rl.strategies import DESCRIPTIONS
from ..ui import theme as T

REPORT = MODEL_PATH.with_name("rl_report.json")


def render(vendors, get_state, POL, evaluate_document) -> None:
    T.page_header("Policy Lab", "An offline, synthetic experiment: can a small contextual bandit suggest tighter redaction than the rule-based policy? "
                  "It can only tighten, never loosen, and it only advises.", "Advisory learning")
    try:
        rep = json.loads(REPORT.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        rep = None
    model = load_model()
    t1, t2, t3 = st.tabs(["Advisor", "Experiment", "Guarantees"])
    with t1:
        if model is None:
            T.empty_state("No trained model", "Run  python scripts/train_rl.py  to create data/rl_model.json.")
        else:
            c = st.columns(3)
            v = c[0].selectbox("Vendor", list(vendors), format_func=lambda x: f"{x} - {vendors[x]['vendor_name']}", key="rl_v")
            names = list(POL["sample_documents"])
            dn = c[1].selectbox("Document", names, key="rl_d")
            task = c[2].selectbox("Task", list(POL["tasks"]), format_func=lambda t: POL["tasks"][t]["label"], key="rl_t")
            s = get_state(vendors[v])
            doc = POL["sample_documents"][dn]
            auth = evaluate_document(s, dn, doc, task)
            adv = advise(s, doc, task, model=model, authoritative=auth, log=st.session_state.get("rl_log", True))
            l, r = st.columns(2, gap="large")
            with l:
                st.markdown("##### Authoritative (rule-based)")
                T.verdict(auth["decision"], f"Cost {auth['cost']:.1f} · budget {s['balance']:,.0f} → {auth['balance_after']:,.0f}", auth["decision"])
            with r:
                st.markdown("##### Advisory (bandit)")
                T.verdict(adv["projected_decision"], f"Strategy {adv['recommended_strategy']} · projected cost {adv['projected_cost']:.1f}", adv["projected_decision"])
            T.callout(DESCRIPTIONS[adv["recommended_strategy"]] + (" This differs from the policy." if adv["differs_from_policy"] else " Same as the policy."), "info")
            st.dataframe(pd.DataFrame([{"Strategy": k, "Estimated value": round(x["value"], 3), "Observations": x["n"], "From": x["source"]} for k, x in adv["values"].items()]), hide_index=True, width="stretch")
            st.caption("Recommendation only. The decision on the left is what the gate enforces. Each advisory is logged with bucketed features only (no content).")
            st.session_state["rl_log"] = st.toggle("Log advisories", st.session_state.get("rl_log", True))
    with t2:
        if not rep:
            T.empty_state("No report", "Run scripts/train_rl.py to generate one.")
        else:
            T.callout(rep["verdict"], "info")
            res = rep["results"]
            st.dataframe(pd.DataFrame([{"Policy": n, "Mean reward": round(r["mean_reward"], 3), "95% CI": f"[{r['reward_ci95_over_sessions'][0]:.3f}, {r['reward_ci95_over_sessions'][1]:.3f}]",
                                        "Utility": round(r["mean_utility"], 3), "Mean cost": round(r["mean_cost"], 2), "Block %": round(100 * r["decisions"]["BLOCK"], 1),
                                        "Review %": round(100 * r["decisions"]["REVIEW"], 1), "Constraint violations": r["constraint_violations"]} for n, r in res.items()]),
                         hide_index=True, width="stretch")
            st.caption(f"Train sessions seeds {rep['train_seeds']}, eval seeds {rep['eval_seeds']} (disjoint). Seed {rep['seed']}.")
            for c in rep["caveats"]:
                T.callout(c, "warn")
            with st.expander("Reward and utility assumptions"):
                st.json(rep["reward_config"])
    with t3:
        for g in ["Hard rules are an action mask, not a penalty: a strategy that would loosen the policy floor, send an unneeded field, or reveal a mandatory type is removed before the bandit sees it.",
                  "The bandit can only keep or tighten what the rule-based engine decided. It cannot unblock a document.",
                  "The authoritative decision, receipt and ledger are produced by the existing engine; the advisor reads them and writes nothing back.",
                  "Training uses synthetic scenarios only. No real document, vendor or ledger row is read.",
                  "The unconstrained ablation scores higher on reward and breaks the hard rules thousands of times: that is why constraints are not left to reward shaping."]:
            T.callout(g, "ok")
