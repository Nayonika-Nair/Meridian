"""Collection orchestrator: search -> guarded fetch -> extract -> dedupe -> corroborate -> store.

Modes
  search    real search provider (env-configured) restricted/augmented per category; fails explicitly if none configured
  manual    analyst-supplied URLs (real, guarded fetch)
  fixture   offline fixtures (fixture:// URLs); results are labelled origin='fixture' and can never reach the Twin
Failures never abort the run: every URL records its own status, and partial results are returned.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from .extract import corroborate, dedupe, extract, registrable, vendor_domain
from .fetcher import FetchResult
from .search import SearchError, SearchUnavailable, provider_from_env
from .store import Store

DEFAULT_CATEGORIES = {
    "subprocessors": {"label": "Sub-processor list", "queries": ["{name} subprocessors list", "site:{domain} subprocessors"]},
    "privacy": {"label": "Privacy policy / DPA", "queries": ["site:{domain} privacy policy AI", "site:{domain} data processing addendum"]},
    "trust": {"label": "Trust / security centre", "queries": ["site:{domain} trust center AI", "site:{domain} security AI"]},
    "ai_features": {"label": "AI features & docs", "queries": ["site:{domain} AI features", "{name} generative AI OpenAI OR Anthropic OR Azure"]},
    "careers": {"label": "Job postings", "queries": ["site:{domain} careers machine learning engineer"]},
    "independent": {"label": "Independent mentions", "queries": ["\"{name}\" AI partnership OpenAI OR Anthropic OR LLM"]},
}


@dataclass
class RunReport:
    run_id: str
    vendor_id: str
    mode: str
    started: float
    finished: float = 0.0
    categories: list = field(default_factory=list)
    urls: list = field(default_factory=list)            # per-URL outcome dicts
    errors: list = field(default_factory=list)
    candidates: list = field(default_factory=list)
    stored: dict = field(default_factory=lambda: {"new": 0, "already_known": 0})
    limitations: list = field(default_factory=list)
    partial: bool = False

    def summary(self) -> dict:
        ok = sum(1 for u in self.urls if u["ok"])
        return {"urls_attempted": len(self.urls), "urls_ok": ok, "urls_failed": len(self.urls) - ok, "from_cache": sum(1 for u in self.urls if u.get("origin") == "cache"),
                "evidence": len(self.candidates), "stored": self.stored, "errors": len(self.errors), "partial": self.partial, "limitations": self.limitations}


def _fmt(t: str, vendor: dict) -> str:
    return t.format(name=vendor["vendor_name"], domain=vendor_domain(vendor))


def run_collection(vendor: dict, *, store: Store, fetcher, mode: str = "search", categories: list[str] | None = None, category_config: dict | None = None,
                   manual_urls: list[str] | None = None, fixture_urls: list[str] | None = None, search_provider=None, env=None,
                   max_urls: int = 12, results_per_query: int = 5, cache_ttl_s: float = 6 * 3600, use_cache: bool = True, refresh: bool = False,
                   allow_stale: bool = False, progress=None, now=time.time) -> RunReport:
    cfg = category_config or DEFAULT_CATEGORIES
    cats = categories or list(cfg)
    rep = RunReport(uuid.uuid4().hex[:12], vendor["vendor_id"], mode, now(), categories=cats)
    _p, _last = progress or (lambda frac, msg: None), [0.0]

    def tick(frac, msg):                                    # monotonic, clamped to [0, 1]
        _last[0] = min(1.0, max(_last[0], frac))
        _p(round(_last[0], 4), msg)
    plan: list[tuple[str, str, str]] = []                  # (url, category, via)
    vdom = vendor_domain(vendor)

    if mode == "search":
        tick(0.02, "Searching")
        try:
            prov = search_provider or provider_from_env(env)
        except SearchUnavailable as exc:
            rep.errors.append({"stage": "search", "code": "no_provider", "detail": str(exc)})
            rep.limitations.append("No search provider configured: nothing was searched or fabricated. Use manual-URL or offline-fixture mode.")
            rep.partial, rep.finished = True, now()
            store.save_run(rep.run_id, rep.vendor_id, rep.started, rep.finished, mode, cats, rep.summary())
            return rep
        seen = set()
        for i, cat in enumerate(cats):
            for q in cfg.get(cat, {}).get("queries", []):
                try:
                    hits = prov.search(_fmt(q, vendor), results_per_query)
                except SearchError as exc:
                    rep.errors.append({"stage": "search", "category": cat, "code": exc.code, "detail": exc.detail})
                    rep.partial = True
                    continue
                for h in hits:
                    if h.url not in seen:
                        seen.add(h.url)
                        plan.append((h.url, cat, f"search:{prov.name}"))
            tick(0.02 + 0.1 * (i + 1) / len(cats), f"Searched {cfg.get(cat, {}).get('label', cat)}")
    elif mode == "manual":
        plan = [(u.strip(), "manual", "manual") for u in (manual_urls or []) if u.strip()]
        if not plan:
            rep.errors.append({"stage": "input", "code": "no_urls", "detail": "No URLs supplied."})
    elif mode == "fixture":
        plan = [(u, "fixture", "fixture") for u in (fixture_urls or [])]
        rep.limitations.append("OFFLINE FIXTURE MODE: results are test data, not retrieved evidence. They cannot be accepted into the Twin.")
    else:
        raise ValueError("mode must be search, manual or fixture")

    if len(plan) > max_urls:
        rep.limitations.append(f"Plan truncated to {max_urls} of {len(plan)} URLs.")
        plan = plan[:max_urls]
    all_c = []
    for i, (url, cat, via) in enumerate(plan):
        tick(0.12 + 0.85 * i / max(1, len(plan)), f"Reading {urlsplit(url).netloc or url}")
        res: FetchResult | None = None
        if use_cache and not refresh and mode != "fixture":
            res = store.cache_get(url, cache_ttl_s, now(), allow_stale=allow_stale)
        if res is None:
            try:
                res = fetcher.fetch(url)
            except Exception as exc:                        # a bug or odd input in one URL must not discard the others
                res = FetchResult(url=url, error="internal", detail=type(exc).__name__)
            if mode != "fixture" and res.ok:
                store.cache_put(res)
        entry = {"url": url, "final_url": res.final_url or url, "category": cat, "via": via, "ok": res.ok, "status": res.status, "error": res.error,
                 "detail": res.detail, "origin": res.origin, "robots": res.robots, "bytes": res.n_bytes, "age_s": res.age_s, "stale": res.stale,
                 "sha256": res.sha256, "redirects": res.redirects, "attempts": res.attempts, "evidence": 0}
        if res.ok:
            origin = "manual" if mode == "manual" and res.origin == "live" else res.origin
            cands = extract(vendor, res, cat, origin=origin)
            entry["evidence"] = len(cands)
            all_c += cands
        else:
            rep.partial = True
            rep.errors.append({"stage": "fetch", "url": url, "code": res.error, "detail": res.detail})
        rep.urls.append(entry)
    tick(0.98, "Deduplicating")
    rep.candidates = corroborate(dedupe(all_c))
    rep.stored = store.upsert_candidates(rep.candidates, rep.run_id, now())
    rep.finished = now()
    if any(u.get("stale") for u in rep.urls):
        rep.limitations.append("Some pages came from a cache older than the freshness window and are flagged stale.")
    if any(u.get("origin") == "cache" and not u.get("stale") for u in rep.urls):
        rep.limitations.append("Some pages came from the cache within the freshness window (age shown per source); use Refresh to re-fetch.")
    rep.limitations.append("Absence of evidence is not evidence of absence: login-gated product behaviour is invisible to public collection.")
    store.save_run(rep.run_id, rep.vendor_id, rep.started, rep.finished, mode, cats, rep.summary())
    tick(1.0, "Done")
    return rep
