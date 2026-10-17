"""B1/B2/B4/B5/B6/B7: search providers, extraction, dedupe, pipeline modes, caching, scoring, review -> Twin. Offline."""
import json

import httpx
import pytest

from liveosint_helpers import FIX, VENDOR, Clock, page, resolver, site
from src.ledger import Ledger
from src.liveosint import (FixtureFetcher, ReviewError, SafeFetcher, SearchUnavailable, Store, adoption_score, corroborate, dedupe, exposure_assessment,
                           extract, provider_from_env, review_evidence, run_collection, score_pair, sync_vendor)
from src.liveosint.fetcher import FetchResult
from src.liveosint.search import SearchError

H = "www.acme-test.com"


def fr(url, text, ctype="text/html", origin="live"):
    return FetchResult(url=url, final_url=url, ok=True, status=200, content_type=ctype, text=text, sha256="0" * 64, n_bytes=len(text), fetched_at=1_000_000.0, origin=origin)


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "osint.db")


def fixture_run(store, urls, **kw):
    return run_collection(VENDOR, store=store, fetcher=FixtureFetcher(FIX), mode="fixture", fixture_urls=urls, **kw)


# ----------------------------------------------------------------------------- search providers (B2)
def test_no_provider_configured_is_explicit_not_fabricated(store):
    with pytest.raises(SearchUnavailable):
        provider_from_env({})
    rep = run_collection(VENDOR, store=store, fetcher=None, mode="search", env={})
    assert rep.candidates == [] and rep.urls == [] and rep.partial
    assert rep.errors[0]["code"] == "no_provider" and "nothing was searched or fabricated" in " ".join(rep.limitations)
    assert store.evidence("V-T1") == []


def test_provider_selection_from_env():
    assert provider_from_env({"BRAVE_SEARCH_API_KEY": "k"}).name == "brave"
    assert provider_from_env({"GOOGLE_CSE_API_KEY": "k", "GOOGLE_CSE_CX": "c"}).name == "google_cse"
    assert provider_from_env({"SEARXNG_URL": "http://localhost:8080"}).name == "searxng"
    with pytest.raises(SearchUnavailable):
        provider_from_env({"GOOGLE_CSE_API_KEY": "k"})                                  # half-configured
    with pytest.raises(SearchUnavailable):
        provider_from_env({"MERIDIAN_SEARCH_PROVIDER": "bing", "BRAVE_SEARCH_API_KEY": "k"})


def _search_client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_brave_google_searxng_parse_responses():
    def brave(req):
        assert req.headers["X-Subscription-Token"] == "secret" and req.url.params["q"] == "acme"
        return httpx.Response(200, json={"web": {"results": [{"url": "https://a.test/1", "title": "t", "description": "d"}, {"title": "no url"}]}})
    p = provider_from_env({"BRAVE_SEARCH_API_KEY": "secret"}, client=_search_client(brave), min_interval=0)
    hits = p.search("acme")
    assert [h.url for h in hits] == ["https://a.test/1"] and hits[0].provider == "brave" and hits[0].rank == 1

    g = provider_from_env({"GOOGLE_CSE_API_KEY": "k", "GOOGLE_CSE_CX": "cx"}, client=_search_client(lambda r: httpx.Response(200, json={"items": [{"link": "https://b.test", "title": "x", "snippet": "y"}]})), min_interval=0)
    assert g.search("q")[0].url == "https://b.test"
    s = provider_from_env({"SEARXNG_URL": "http://sx"}, client=_search_client(lambda r: httpx.Response(200, json={"results": [{"url": "https://c.test", "content": "z"}]})), min_interval=0)
    assert s.search("q")[0].snippet == "z"


@pytest.mark.parametrize("status,code", [(429, "rate_limited"), (401, "auth"), (500, "http")])
def test_search_errors_are_mapped_and_never_leak_the_key(status, code):
    p = provider_from_env({"BRAVE_SEARCH_API_KEY": "TOPSECRETKEY"}, client=_search_client(lambda r: httpx.Response(status)), min_interval=0)
    with pytest.raises(SearchError) as e:
        p.search("q")
    assert e.value.code == code and "TOPSECRETKEY" not in str(e.value)


def test_search_timeout_and_bad_json_are_mapped():
    def to(r):
        raise httpx.ReadTimeout("slow")
    p = provider_from_env({"BRAVE_SEARCH_API_KEY": "k"}, client=_search_client(to), min_interval=0)
    with pytest.raises(SearchError) as e:
        p.search("q")
    assert e.value.code == "timeout"
    p = provider_from_env({"BRAVE_SEARCH_API_KEY": "k"}, client=_search_client(lambda r: httpx.Response(200, text="<html>")), min_interval=0)
    with pytest.raises(SearchError) as e:
        p.search("q")
    assert e.value.code == "bad_response"


def test_search_pipeline_end_to_end_with_mocked_search_and_site(store):
    routes = {(H, "/subprocessors"): (200, {}, page("acme/subprocessors.html")), (H, "/"): (200, {}, page("acme/index.html")),
              ("news.example.org", "/acme"): (200, {}, page("press/news.html"))}
    hits = [{"url": f"https://{H}/subprocessors"}, {"url": f"https://{H}/"}, {"url": "https://news.example.org/acme"}, {"url": f"https://{H}/missing"}]
    sp = provider_from_env({"BRAVE_SEARCH_API_KEY": "k"}, client=_search_client(lambda r: httpx.Response(200, json={"web": {"results": hits}})), min_interval=0)
    clk = Clock()
    f = SafeFetcher(transport=site(routes), resolver=resolver, sleep=clk.sleep, clock=clk.now, min_interval=0)
    prog = []
    rep = run_collection(VENDOR, store=store, fetcher=f, mode="search", search_provider=sp, categories=["subprocessors"], progress=lambda a, b: prog.append(a), now=clk.now)
    assert rep.partial and any(e["code"] == "http_404" for e in rep.errors)                # partial results kept alongside the failure
    assert len(rep.candidates) >= 3 and prog[-1] == 1.0 and prog == sorted(prog)
    assert {u["url"] for u in rep.urls if u["ok"]} >= {f"https://{H}/subprocessors", "https://news.example.org/acme"}
    assert all(c.origin == "live" for c in rep.candidates)


# ----------------------------------------------------------------------------- extraction + provenance (B4)
def test_extraction_records_provenance_hashes_and_classes():
    c = {x.excerpt: x for x in extract(VENDOR, fr(f"https://{H}/subprocessors", page("acme/subprocessors.html")), "subprocessors")}
    disc = next(v for k, v in c.items() if "customer data to OpenAI" in k)
    assert disc.evidence_class == "explicit_processing_disclosure" and disc.providers == ["OpenAI"] and disc.reliability_tier == 1
    assert disc.url == f"https://{H}/subprocessors" and disc.source_type == "subprocessor_list" and disc.origin == "live"
    assert len(disc.page_sha256) == 64 and len(disc.excerpt_sha256) == 64 and disc.retrieved_at == 1_000_000.0 and disc.evidence_id.startswith("EV-")
    assert any(v.specificity == "endpoint" and v.evidence_class == "technical_indication" for v in c.values())      # script endpoint
    assert all(v.evidence_class != "observed_flow" for v in c.values())


def test_marketing_page_is_marketing_and_hidden_text_comments_and_scripts_are_ignored():
    out = extract(VENDOR, fr(f"https://{H}/", page("acme/index.html")), "ai_features")
    assert len(out) == 1 and out[0].evidence_class == "marketing_claim" and out[0].specificity == "vague"
    blob = " ".join(o.excerpt for o in out)
    assert "OpenAI" not in blob and "Anthropic" not in blob                              # comment + script text never become evidence


def test_negation_is_recorded_but_flagged():
    out = extract(VENDOR, fr(f"https://{H}/trust", page("acme/trust.html")), "trust")
    neg = [o for o in out if o.negated]
    assert neg and "do not use customer data" in neg[0].excerpt and neg[0].evidence_class != "explicit_processing_disclosure"


def test_prompt_injection_text_is_inert():
    out = extract(VENDOR, fr(f"https://{H}/x", page("acme/injection.html")), "ai_features")
    assert all(o.evidence_class in ("marketing_claim", "technical_indication") for o in out)
    assert not any(o.evidence_class in ("observed_flow", "verified_flow") for o in out)


def test_control_and_bidi_characters_are_stripped_from_excerpts():
    out = extract(VENDOR, fr(f"https://{H}/", "<p>Our ‮AI‬-powered\x07 assistant is great.</p>"), "x")
    assert out and "‮" not in out[0].excerpt and "\x07" not in out[0].excerpt


def test_evidence_id_is_stable_and_dedupe_collapses_boilerplate():
    a = extract(VENDOR, fr(f"https://{H}/", page("acme/index.html")), "x")
    b = extract(VENDOR, fr(f"https://{H}/", page("acme/index.html")), "x")
    assert a[0].evidence_id == b[0].evidence_id
    foot = "<p>Our AI-powered assistant helps teams move faster.</p>"
    p1, p2 = extract(VENDOR, fr(f"https://{H}/blog/x", foot), "x"), extract(VENDOR, fr(f"https://{H}/security", foot), "x")
    merged = dedupe(p1 + p2 + p1)
    assert len(merged) == 1 and merged[0].reliability_tier == 1 and merged[0].also_seen_at          # kept the most reliable, remembered the other


def test_corroboration_needs_two_domains_one_independent_and_never_fixtures():
    live = extract(VENDOR, fr(f"https://{H}/careers", page("acme/careers.html")), "x")
    third = extract(VENDOR, fr("https://news.example.org/acme", page("press/news.html")), "x")
    assert third[0].independent and third[0].source_type == "third_party"
    out = corroborate(live + third)
    assert {o.evidence_class for o in out if "Anthropic" in o.excerpt or "OpenAI" in o.excerpt} <= {"corroborated_claim", "marketing_claim", "technical_indication"}
    openai = [o for o in out if o.providers == ["OpenAI"]]
    assert openai and all(o.evidence_class != "corroborated_claim" for o in openai)               # only 1 domain says OpenAI here
    same = extract(VENDOR, fr(f"https://{H}/a", "<p>Built on OpenAI models.</p>"), "x") + third
    assert {o.evidence_class for o in corroborate(same)} == {"corroborated_claim"}
    fx = extract(VENDOR, fr("fixture://acme/a.html", "<p>Built on OpenAI models.</p>", origin="fixture"), "x", independent=False) \
        + extract(VENDOR, fr("fixture://press/b.html", "<p>Built on OpenAI models too.</p>", origin="fixture"), "x", independent=True)
    assert all(o.evidence_class != "corroborated_claim" for o in corroborate(fx))


def test_pages_without_ai_wording_yield_nothing():
    assert extract(VENDOR, fr(f"https://{H}/", "<p>We make pencils.</p>"), "x") == []
    assert extract(VENDOR, FetchResult(url="u", ok=False, error="http_404"), "x") == []


# ----------------------------------------------------------------------------- pipeline modes / cache (B2, B6)
def test_fixture_mode_is_labelled_and_never_cached(store):
    rep = fixture_run(store, ["fixture://acme/subprocessors.html", "fixture://acme/nope.html"])
    assert rep.partial and rep.candidates and all(c.origin == "fixture" for c in rep.candidates)
    assert "OFFLINE FIXTURE MODE" in rep.limitations[0]
    assert store.cache_get("fixture://acme/subprocessors.html", 1e9) is None


def test_manual_url_mode_marks_origin_manual_and_reports_per_url_errors(store):
    routes = {(H, "/careers"): (200, {}, page("acme/careers.html"))}
    f = SafeFetcher(transport=site(routes), resolver=resolver, min_interval=0, sleep=lambda s: None)
    rep = run_collection(VENDOR, store=store, fetcher=f, mode="manual", manual_urls=[f"https://{H}/careers", f"https://{H}/gone", "http://169.254.169.254/x"])
    assert {c.origin for c in rep.candidates} == {"manual"} and rep.partial
    codes = {u["url"]: u["error"] for u in rep.urls}
    assert codes["http://169.254.169.254/x"].startswith("blocked:") and codes[f"https://{H}/gone"] == "http_404"
    assert run_collection(VENDOR, store=store, fetcher=f, mode="manual", manual_urls=[]).errors[0]["code"] == "no_urls"


def test_cache_is_labelled_stale_data_is_not_presented_as_current_and_refresh_refetches(store):
    n = {"n": 0}

    def counted(req):
        n["n"] += 1
        return httpx.Response(200, headers={"content-type": "text/html"}, text=page("acme/careers.html"))
    routes = {(H, "/careers"): counted}
    clk = Clock()
    f = SafeFetcher(transport=site(routes), resolver=resolver, min_interval=0, sleep=clk.sleep, clock=clk.now, wall=clk.now)
    kw = dict(store=store, fetcher=f, mode="manual", manual_urls=[f"https://{H}/careers"], cache_ttl_s=3600, now=lambda: clk.t + 0)
    run_collection(VENDOR, **kw)
    assert n["n"] == 1
    r2 = run_collection(VENDOR, **kw)                                                                # within TTL -> cache, labelled
    assert n["n"] == 1 and r2.urls[0]["origin"] == "cache" and not r2.urls[0]["stale"] and any("cache" in x for x in r2.limitations)
    clk.t += 7200
    r3 = run_collection(VENDOR, **kw)                                                                # beyond TTL -> refetched, not served stale
    assert n["n"] == 2 and r3.urls[0]["origin"] == "live"
    clk.t += 7200
    r4 = run_collection(VENDOR, **{**kw, "allow_stale": True})                                      # explicit stale allowed -> flagged
    assert r4.urls[0]["origin"] == "cache" and r4.urls[0]["stale"] and r4.urls[0]["age_s"] >= 7200 and any("stale" in x for x in r4.limitations)
    run_collection(VENDOR, **{**kw, "refresh": True})
    assert n["n"] == 3


def test_rerun_does_not_duplicate_evidence_and_keeps_review_state(store):
    r1 = fixture_run(store, ["fixture://acme/subprocessors.html"])
    assert r1.stored["new"] > 0
    ev = store.evidence("V-T1")[0]
    store.set_review(ev["evidence_id"], "needs_review", "ana", "check", None, None, None, 1.0)
    r2 = fixture_run(store, ["fixture://acme/subprocessors.html"])
    assert r2.stored["new"] == 0 and r2.stored["already_known"] == r1.stored["new"]
    assert store.get(ev["evidence_id"])["review_status"] == "needs_review"


# ----------------------------------------------------------------------------- scoring (B5)
def E(cls, spec="provider", tier=1, dom="a.com", status="pending", origin="live", neg=False):
    return {"effective_class": cls, "specificity": spec, "reliability_tier": tier, "domain": dom, "review_status": status, "origin": origin, "negated": neg}


def test_adoption_score_is_monotonic_capped_per_domain_and_excludes_fixture_rejected_negated():
    base = adoption_score([E("marketing_claim", "vague", 3)])
    more = adoption_score([E("marketing_claim", "vague", 3), E("technical_indication", dom="b.com")])
    assert 0 < base["score"] < more["score"] <= 100
    same_dom = adoption_score([E("technical_indication", dom="a.com") for _ in range(10)])
    assert same_dom["items_counted"] == 2 and same_dom["excluded"]["domain_cap"] == 8
    ex = adoption_score([E("technical_indication", origin="fixture"), E("technical_indication", status="rejected"), E("technical_indication", neg=True)])
    assert ex["score"] == 0 and ex["excluded"] == {"rejected": 1, "negated": 1, "fixture": 1, "domain_cap": 0}
    assert "assumption" in base["weights_are"]


def test_ai_use_alone_never_produces_a_flow_level():
    strong_use = [E("technical_indication", dom=f"d{i}.com", status="accepted") for i in range(6)] + [E("corroborated_claim", dom="z.com", status="accepted")]
    a = exposure_assessment(VENDOR, strong_use, p_ai=0.5)
    assert adoption_score(strong_use)["score"] > 80 and a["level"] == "NO_FLOW_EVIDENCE"
    assert a["never_implied"].startswith("AI use alone")
    assert exposure_assessment(VENDOR, [], 0.0)["level"] == "NO_EVIDENCE"
    assert exposure_assessment(VENDOR, [E("explicit_processing_disclosure")], 0.1)["level"] == "DISCLOSED_PROCESSING"
    assert exposure_assessment(VENDOR, [E("verified_flow", status="accepted")], 0.9)["level"] == "FLOW_OBSERVED"
    assert exposure_assessment(VENDOR, [E("verified_flow", status="pending")], 0.9)["level"] != "FLOW_OBSERVED"       # unreviewed never counts
    assert exposure_assessment(VENDOR, [E("explicit_processing_disclosure", origin="fixture")], 0)["level"] == "NO_EVIDENCE"


def test_score_pair_separates_hypothesis_from_reviewed():
    ev = [E("technical_indication", status="accepted", dom="a.com"), E("technical_indication", status="pending", dom="b.com")]
    p = score_pair(ev)
    assert p["hypothesis_all_unrejected"]["score"] > p["reviewed_only"]["score"] > 0


# ----------------------------------------------------------------------------- review -> Twin / budget / audit (B7)
@pytest.fixture
def live_store(store):
    """Real-origin evidence (as if fetched) stored directly, plus one fixture row."""
    live = extract(VENDOR, fr(f"https://{H}/subprocessors", page("acme/subprocessors.html")), "subprocessors", now=1_000_100.0)
    live += extract(VENDOR, fr(f"https://{H}/", page("acme/index.html")), "x")
    store.upsert_candidates(live, "run1", 1_000_100.0)
    fx = extract(VENDOR, fr("fixture://acme/careers.html", page("acme/careers.html"), origin="fixture"), "fixture")
    store.upsert_candidates(fx, "run2", 1_000_100.0)
    return store


def ids(store, pred):
    return [e["evidence_id"] for e in store.evidence("V-T1") if pred(e)]


def test_pending_evidence_does_not_touch_twin_or_budget(live_store, tmp_path):
    ov = tmp_path / "overlay.json"
    led = Ledger(str(tmp_path / "l.db"))
    out = sync_vendor(live_store, VENDOR, [], overlay_path=ov, ledger=led, trials=500)
    assert out["changed"] is False and not ov.exists() and led.rows() == []
    led.conn.close()


def test_accept_updates_twin_with_before_after_and_one_audit_row_and_is_idempotent(live_store, tmp_path):
    ov, led = tmp_path / "overlay.json", Ledger(str(tmp_path / "l.db"))
    disc = ids(live_store, lambda e: e["evidence_class"] == "explicit_processing_disclosure" and e["providers"] == ["OpenAI"] and e["specificity"] == "provider")[0]
    assert review_evidence(live_store, disc, "accepted", "ana", "matches DPA", now=1_086_500.0)["changed"]
    out = sync_vendor(live_store, VENDOR, [], overlay_path=ov, ledger=led, trials=1500, now=1_086_600.0)
    assert out["changed"] and out["added"] == [disc]
    assert out["before"]["p_ai"] == 0.0 and out["after"]["p_ai"] > 0.5 and out["after"]["allowance"] < out["before"]["allowance"]
    row = json.loads(ov.read_text())["V-T1"][0]
    assert row["liveosint_id"] == disc and row["claim_type"] == "ai_use_only" and row["tag"] != "Confirmed" and row["verified"] is True
    assert row["target"] == "OpenAI" and row["target_kind"] == "ai_provider" and row["age_days"] == 1
    rows = led.records()
    assert len(rows) == 1 and rows[0]["cost"] == 0 and rows[0]["meta"]["kind"] == "osint_review_sync" and rows[0]["meta"]["p_ai_after"] > 0.5
    assert led.verify()
    again = sync_vendor(live_store, VENDOR, [], overlay_path=ov, ledger=led, trials=1500, now=1_999_999.0)           # idempotent, even much later
    assert again["changed"] is False and len(led.records()) == 1 and json.loads(ov.read_text())["V-T1"] == [row]
    assert review_evidence(live_store, disc, "accepted", "ana", now=2_000_000.0)["changed"] is False
    assert len(live_store.review_log("V-T1")) == 1
    led.conn.close()


def test_reject_after_accept_retracts_and_foreign_rows_survive(live_store, tmp_path):
    ov, led = tmp_path / "overlay.json", Ledger(str(tmp_path / "l.db"))
    manual = {"parent": "Acme", "target": "Hand-added", "target_kind": "service", "tag": "Marketing claim only", "age_days": 1}
    ov.write_text(json.dumps({"V-T1": [manual]}))
    e1 = ids(live_store, lambda e: e["providers"] == ["OpenAI"])[0]
    review_evidence(live_store, e1, "accepted", "ana")
    sync_vendor(live_store, VENDOR, [], overlay_path=ov, ledger=led, trials=500)
    assert len(json.loads(ov.read_text())["V-T1"]) == 2
    review_evidence(live_store, e1, "rejected", "ana", "wrong vendor")
    out = sync_vendor(live_store, VENDOR, [], overlay_path=ov, ledger=led, trials=500)
    assert out["changed"] and out["removed"] == [e1] and json.loads(ov.read_text())["V-T1"] == [manual]
    assert len(led.records()) == 2 and led.verify()
    led.conn.close()


def test_needs_review_and_fixture_and_negated_cannot_enter_the_twin(live_store):
    fx = ids(live_store, lambda e: e["origin"] == "fixture")[0]
    with pytest.raises(ReviewError, match="fixture"):
        review_evidence(live_store, fx, "accepted", "ana")
    nr = ids(live_store, lambda e: e["origin"] == "live")[0]
    review_evidence(live_store, nr, "needs_review", "ana", "unsure")
    assert live_store.get(nr)["review_status"] == "needs_review"
    from src.liveosint.review import desired_rows
    assert desired_rows(live_store, VENDOR) == []
    neg = extract(VENDOR, fr(f"https://{H}/trust", page("acme/trust.html")), "trust")
    live_store.upsert_candidates(neg, "r3")
    nid = next(n.evidence_id for n in neg if n.negated)
    with pytest.raises(ReviewError, match="negative"):
        review_evidence(live_store, nid, "accepted", "ana")


def test_flow_classes_need_an_authorised_observation_reference(live_store, tmp_path):
    e1 = ids(live_store, lambda e: e["providers"] == ["OpenAI"])[0]
    with pytest.raises(ReviewError, match="observation"):
        review_evidence(live_store, e1, "accepted", "ana", promote_to="verified_flow")
    with pytest.raises(ReviewError):
        review_evidence(live_store, e1, "needs_review", "ana", promote_to="verified_flow", observation_ref="TICKET-123")
    with pytest.raises(ReviewError):
        review_evidence(live_store, e1, "accepted", "ana", promote_to="made_up", observation_ref="TICKET-123")
    review_evidence(live_store, e1, "accepted", "ana", promote_to="verified_flow", observation_ref="SEC-4411 egress capture")
    ov = tmp_path / "o.json"
    sync_vendor(live_store, VENDOR, [], overlay_path=ov, trials=500)
    row = json.loads(ov.read_text())["V-T1"][0]
    assert row["tag"] == "Confirmed" and row["claim_type"] == "data_flow_proven" and row["observation_ref"].startswith("SEC-4411")


def test_review_input_validation(live_store):
    e1 = ids(live_store, lambda e: e["origin"] == "live")[0]
    for bad in [("pending", "ana"), ("accepted", ""), ("nonsense", "ana")]:
        with pytest.raises(ReviewError):
            review_evidence(live_store, e1, *bad)
    with pytest.raises(ReviewError):
        review_evidence(live_store, "EV-unknown", "accepted", "ana")


def test_dry_run_changes_nothing(live_store, tmp_path):
    e1 = ids(live_store, lambda e: e["providers"] == ["OpenAI"])[0]
    review_evidence(live_store, e1, "accepted", "ana", now=1_086_500.0)
    ov = tmp_path / "o.json"
    out = sync_vendor(live_store, VENDOR, [], overlay_path=ov, trials=500, dry_run=True)
    assert out["changed"] and not ov.exists() and out["after"]["p_ai"] > out["before"]["p_ai"]


def test_store_connections_are_closed_so_the_file_can_be_removed(tmp_path):
    import os
    s = Store(tmp_path / "x.db")
    s.upsert_candidates([], "r")
    s.evidence("V")
    os.remove(tmp_path / "x.db")                                                            # would fail on Windows if a handle leaked
