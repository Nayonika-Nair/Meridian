"""Live OSINT Explorer: configure -> collect -> review -> assess -> apply.

Only analyst-reviewed evidence can change the Twin or the Leak Budget. Everything fetched from the web is untrusted
text: it is HTML-escaped before display and never interpreted."""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import streamlit as st

from ..config import ROOT
from ..liveosint import (DEFAULT_CATEGORIES, FixtureFetcher, ReviewError, SafeFetcher, Store, exposure_assessment, review_evidence, run_collection, score_pair,
                         sync_vendor, vendor_domain)
from ..liveosint.extract import CLASSES
from ..liveosint.search import configured_provider_name
from ..ui import theme as T

FIXTURES = ROOT / "data" / "osint_fixtures"
CLASS_TONE = {"marketing_claim": "muted", "technical_indication": "info", "corroborated_claim": "accent", "explicit_processing_disclosure": "warn",
              "observed_flow": "crit", "verified_flow": "crit"}
CLASS_HELP = {"marketing_claim": "AI wording on a marketing page", "technical_indication": "Provider, capability or endpoint named in docs, jobs or code",
              "corroborated_claim": "Same AI use asserted on 2+ domains, one independent", "explicit_processing_disclosure": "Vendor legal text ties AI to customer data or lists the provider as a sub-processor",
              "observed_flow": "Analyst-recorded observation of Meridian data reaching AI", "verified_flow": "Observation verified against an authorised source"}
LEVEL_TONE = {"NO_EVIDENCE": "muted", "NO_FLOW_EVIDENCE": "info", "DISCLOSED_PROCESSING": "high", "FLOW_OBSERVED": "crit"}


def _age(s: float) -> str:
    return f"{s / 3600:.1f} h" if s >= 3600 else f"{s / 60:.0f} min" if s >= 60 else f"{s:.0f} s"


def _source_card(u: dict) -> str:
    tone = "ok" if u["ok"] else "crit"
    chips = [T.badge("OK" if u["ok"] else (u["error"] or "failed"), tone)]
    if u["origin"] == "cache":
        chips.append(T.badge(("STALE cache " if u["stale"] else "cache ") + _age(u["age_s"]) + " old", "warn" if u["stale"] else "info"))
    if u["origin"] == "fixture":
        chips.append(T.badge("FIXTURE", "high"))
    if u.get("robots") not in ("n/a", "allowed", None):
        chips.append(T.badge("robots: " + u["robots"], "warn"))
    detail = f'{T.esc(u["category"])} · via {T.esc(u["via"])} · HTTP {T.esc(u["status"] or "-")} · {u["evidence"]} evidence · {u["bytes"]:,} B'
    if u["redirects"]:
        detail += f' · redirected {len(u["redirects"])}×'
    link = f'<a href="{T.esc(u["final_url"])}" target="_blank" rel="noopener noreferrer nofollow">open source</a>' if u["ok"] and u["origin"] != "fixture" else ""
    err = '<div class="m-mono">' + T.esc(u["detail"]) + "</div>" if u["detail"] else ""
    return ('<div class="m-src"><div>' + "".join(chips) + '</div><div><div class="u">' + T.esc(u["url"]) + '</div>'
            '<div style="color:var(--m-muted);font-size:.78rem">' + detail + " " + link + "</div>" + err + '</div><div class="m-mono">' + T.esc(u["sha256"][:10]) + "</div></div>")


def _evidence_card(e: dict) -> str:
    cls = e["effective_class"]
    chips = [T.badge(cls.replace("_", " "), CLASS_TONE[cls]), T.badge(f"tier {e['reliability_tier']}", "muted"), T.badge(e["specificity"], "muted")]
    if e["providers"]:
        chips.append(T.badge(", ".join(e["providers"]), "accent"))
    if e["negated"]:
        chips.append(T.badge("negative statement", "ok"))
    if e["origin"] == "fixture":
        chips.append(T.badge("FIXTURE", "high"))
    if e["independent"]:
        chips.append(T.badge("independent", "info"))
    seen = f" · also seen on {len(e['also_seen_at'])} other URL(s)" if e["also_seen_at"] else ""
    return (f'<div class="m-card">{"".join(chips)}<div class="m-quote">“{T.esc(e["excerpt"])}”</div>'
            f'<div class="m-mono">{T.esc(e["domain"])} · {T.esc(e["source_type"])} · {T.esc(e["final_url"])}{seen}<br>'
            f'page sha256 {T.esc(e["page_sha256"][:16])}… · excerpt {T.esc(e["excerpt_sha256"][:12])}… · {time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(e["retrieved_at"]))} · {T.esc(e["evidence_id"])}</div></div>')


def render(vendors: dict, evidence_base: dict, ledger, base: float, trials: int, disc: float, reviewer_default: str = "analyst") -> None:
    T.page_header("Live OSINT Explorer", "Collect public evidence responsibly, review it, and decide what the Digital Twin may believe. "
                  "AI adoption and Meridian exposure are scored separately: using AI is not the same as receiving your data.", "Evidence collection")
    store = Store()
    T.stepper(["Configure", "Collect", "Review", "Assess", "Apply"], st.session_state.get("lo_step", 0))
    c1, c2 = st.columns([1, 1.6], gap="large")
    with c1:
        vid = st.selectbox("Vendor", list(vendors), format_func=lambda x: f"{x} - {vendors[x]['vendor_name']}", key="lo_vendor")
        v = vendors[vid]
        provider = configured_provider_name()
        mode = st.radio("Source", ["Search", "Manual URLs", "Offline fixtures"], key="lo_mode",
                        help="Search needs an API key in the environment. Offline fixtures are synthetic test pages.")
        if mode == "Search":
            T.callout(f"Search provider: {provider}." if provider else "No search provider configured. Set BRAVE_SEARCH_API_KEY, GOOGLE_CSE_API_KEY + GOOGLE_CSE_CX, or SEARXNG_URL - or use Manual URLs. Nothing will be invented.", "ok" if provider else "warn")
            cats = st.multiselect("Search categories", list(DEFAULT_CATEGORIES), default=list(DEFAULT_CATEGORIES)[:4], format_func=lambda c: DEFAULT_CATEGORIES[c]["label"])
            urls = []
        elif mode == "Manual URLs":
            cats = []
            urls = st.text_area("URLs (one per line)", value=f"https://{vendor_domain(v)}/", height=120, help="Public http(s) pages only. Private and internal addresses are refused.").splitlines()
        else:
            cats = []
            names = sorted(p.relative_to(FIXTURES).as_posix() for p in FIXTURES.rglob("*.html"))
            pick = st.multiselect("Fixture pages", names, default=names)
            urls = [f"fixture://{n}" for n in pick]
            T.callout("Synthetic pages. Evidence from them is labelled FIXTURE and cannot be accepted into the Twin.", "warn")
        with st.expander("Collection settings"):
            use_cache = st.checkbox("Use cached pages", True)
            ttl_h = st.number_input("Treat cache as current for (hours)", 1, 168, 6)
            allow_stale = st.checkbox("Show older cached pages (flagged stale)", False)
            refresh = st.checkbox("Force refresh", False)
            max_urls = st.number_input("Max pages per run", 1, 40, 12)
            interval = st.number_input("Seconds between requests to one site", 0.5, 30.0, 1.5, 0.5)
        run = st.button("Collect evidence", type="primary")
    with c2:
        if run:
            bar, msg = st.progress(0.0), st.empty()
            fetcher = FixtureFetcher(FIXTURES) if mode == "Offline fixtures" else SafeFetcher(min_interval=float(interval))
            try:
                rep = run_collection(v, store=store, fetcher=fetcher, mode={"Search": "search", "Manual URLs": "manual", "Offline fixtures": "fixture"}[mode],
                                     categories=cats or None, manual_urls=urls if mode == "Manual URLs" else None, fixture_urls=urls if mode == "Offline fixtures" else None,
                                     max_urls=int(max_urls), cache_ttl_s=float(ttl_h) * 3600, use_cache=use_cache, refresh=refresh, allow_stale=allow_stale,
                                     progress=lambda f, m: (bar.progress(min(1.0, f)), msg.caption(m)))
            finally:
                fetcher.close()
            bar.empty()
            msg.empty()
            st.session_state["lo_report"] = {"vid": vid, "run_id": rep.run_id, "urls": rep.urls, "errors": rep.errors, "limitations": rep.limitations, "partial": rep.partial,
                                            "summary": rep.summary(), "mode": mode}
            st.session_state["lo_step"] = 1
        rep = st.session_state.get("lo_report")
        if not rep or rep["vid"] != vid:
            T.empty_state("No collection yet", "Choose a source on the left and collect. Every page, error and skipped URL is listed; partial results are kept.")
        else:
            s = rep["summary"]
            T.kpi_row([("Pages tried", s["urls_attempted"], f"{s['urls_ok']} ok", "accent"), ("Failed", s["urls_failed"], "see cards", "crit" if s["urls_failed"] else "ok"),
                       ("Evidence", s["evidence"], f"{s['stored']['new']} new", "accent"), ("From cache", s["from_cache"], "labelled with age", "info")])
            if rep["partial"]:
                T.callout("Partial results: some sources failed or were not allowed. What was collected is shown below.", "warn")
            for e in rep["errors"]:
                if e["stage"] != "fetch":
                    T.callout(f"{e['stage']}: {e['detail']}", "warn")
            for l in rep["limitations"]:
                T.callout(l, "info")
            T.render("".join(_source_card(u) for u in rep["urls"]) or "")

    # ----------------------------------------------------------------- review, assess, apply
    st.divider()
    ev = store.evidence(vid)
    if not ev:
        return
    who = st.text_input("Reviewer", reviewer_default, key="lo_who")
    tabs = st.tabs([f"Pending ({sum(e['review_status'] == 'pending' for e in ev)})", f"Needs review ({sum(e['review_status'] == 'needs_review' for e in ev)})",
                    f"Accepted ({sum(e['review_status'] == 'accepted' for e in ev)})", f"Rejected ({sum(e['review_status'] == 'rejected' for e in ev)})"])
    for tab, status in zip(tabs, ["pending", "needs_review", "accepted", "rejected"]):
        with tab:
            rows = [e for e in ev if e["review_status"] == status]
            if not rows:
                T.empty_state("Nothing here", "")
            for e in rows:
                T.render(_evidence_card(e))
                b = st.columns([1, 1, 1, 3])
                for col, (label, dec) in zip(b, (("Accept", "accepted"), ("Reject", "rejected"), ("Needs review", "needs_review"))):
                    if e["review_status"] != dec and col.button(label, key=f"lo_{dec}_{e['evidence_id']}", disabled=(dec == "accepted" and (e["origin"] == "fixture" or e["negated"]))):
                        try:
                            review_evidence(store, e["evidence_id"], dec, who)
                            st.rerun()
                        except ReviewError as exc:
                            st.error(str(exc))
                if status == "accepted":
                    with st.expander("Record an authorised observation (flow evidence)"):
                        ref = st.text_input("Reference (ticket, contract clause, audit id)", key=f"lo_ref_{e['evidence_id']}")
                        kind = st.selectbox("Class", ["observed_flow", "verified_flow"], key=f"lo_pc_{e['evidence_id']}")
                        if st.button("Promote", key=f"lo_pr_{e['evidence_id']}"):
                            try:
                                review_evidence(store, e["evidence_id"], "accepted", who, observation_ref=ref, promote_to=kind)
                                st.rerun()
                            except ReviewError as exc:
                                st.error(str(exc))

    st.subheader("Assessments")
    st.session_state["lo_step"] = 3
    base_rows = evidence_base.get(vid, [])
    syn = sync_vendor(store, v, base_rows, overlay_path=None, ledger=None, base=base, trials=int(trials), flow_discount=disc, dry_run=True)
    sc = score_pair(ev)
    ex = exposure_assessment(v, ev, syn["before"]["p_ai"], disc)
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("##### AI Adoption Evidence Score")
        hyp, rev = sc["hypothesis_all_unrejected"], sc["reviewed_only"]
        T.render(T.kpi("Hypothesis (all unrejected)", f"{hyp['score']:.0f} / 100", f"{hyp['items_counted']} items · {hyp['domains']} domain(s) · breadth {hyp['evidence_breadth']}", "info")
                 + "<div style='height:8px'></div>" + T.kpi("Reviewed only", f"{rev['score']:.0f} / 100", f"{rev['items_counted']} accepted items", "ok"))
        st.caption("How strongly the evidence suggests the VENDOR adopts AI. Weights are configurable assumptions, not measurements. Fixtures, negations and rejected items are excluded; at most 2 items per domain count.")
        with st.expander("Weights in force"):
            st.json(hyp["weights"])
    with b:
        st.markdown("##### Meridian AI Exposure Assessment")
        T.render(T.kpi("Level", ex["level"].replace("_", " "), ex["summary"], LEVEL_TONE[ex["level"]]))
        for r in ex["reasons"]:
            T.callout(r, "info")
        st.caption(ex["never_implied"])

    st.subheader("Apply reviewed evidence")
    st.session_state["lo_step"] = 4
    bf, af = syn["before"], syn["after"]
    rows_kv = [("Accepted rows that would be in the Twin", syn["accepted_rows"]),
               ("Chance of AI (P_ai)", "%.1f%%  ->  %.1f%%" % (100 * bf["p_ai"], 100 * af["p_ai"])),
               ("Leak budget (allowance)", "{:,.0f}  ->  {:,.0f}".format(bf["allowance"], af["allowance"])),
               ("Send state", "%s  ->  %s" % (bf["state"], af["state"]))]
    T.render('<div class="m-card">' + T.kv(rows_kv) + "</div>")
    if not syn["changed"]:
        T.callout("Nothing to apply: the Twin already reflects the accepted evidence.", "ok")
    elif st.button("Apply to Twin and Leak Budget", type="primary"):
        out = sync_vendor(store, v, base_rows, ledger=ledger, base=base, trials=int(trials), flow_discount=disc, actor=who)
        st.success(f"Applied. Audit entry {out['audit_hash'][:12] if out['audit_hash'] else '-'} written to the ledger. Re-applying is a no-op.")
        st.rerun()
    with st.expander("Review log"):
        log = store.review_log(vid)
        st.dataframe(pd.DataFrame(log) if log else pd.DataFrame({"info": ["No reviews yet"]}), hide_index=True, width="stretch")
