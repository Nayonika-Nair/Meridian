"""Search providers, configured ONLY through environment variables. No provider configured = an explicit
`SearchUnavailable`; the pipeline then offers manual-URL and offline-fixture modes. Nothing is ever invented.

  MERIDIAN_SEARCH_PROVIDER   brave | google_cse | searxng     (optional; auto-detected from the keys below)
  BRAVE_SEARCH_API_KEY
  GOOGLE_CSE_API_KEY, GOOGLE_CSE_CX
  SEARXNG_URL                e.g. http://localhost:8080   (self-hosted; private addresses allowed for this URL only)
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import httpx


class SearchUnavailable(Exception):
    pass


class SearchError(Exception):
    def __init__(self, code: str, detail: str = ""):
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code, self.detail = code, detail


@dataclass(frozen=True)
class SearchHit:
    url: str
    title: str
    snippet: str
    rank: int
    provider: str


class _Base:
    name = "base"

    def __init__(self, client: httpx.Client | None = None, timeout: float = 10.0, min_interval: float = 1.0, sleep=time.sleep, clock=time.monotonic):
        self.client = client or httpx.Client(timeout=timeout, follow_redirects=False)
        self.min_interval, self.sleep, self.clock, self._last = min_interval, sleep, clock, None

    def _pace(self):
        if self._last is not None:
            wait = self._last + self.min_interval - self.clock()
            if wait > 0:
                self.sleep(wait)
        self._last = self.clock()

    def _json(self, url, params, headers=None):
        self._pace()
        try:
            r = self.client.get(url, params=params, headers=headers or {})
        except httpx.TimeoutException as exc:
            raise SearchError("timeout", "search provider timed out") from exc
        except httpx.HTTPError as exc:
            raise SearchError("network", type(exc).__name__) from exc
        if r.status_code == 429:
            raise SearchError("rate_limited", "search provider rate limit")
        if r.status_code in (401, 403):
            raise SearchError("auth", f"provider rejected credentials (HTTP {r.status_code})")
        if r.status_code != 200:
            raise SearchError("http", f"HTTP {r.status_code}")
        try:
            return r.json()
        except ValueError as exc:
            raise SearchError("bad_response", "not JSON") from exc

    def search(self, query: str, limit: int = 8) -> list[SearchHit]:       # pragma: no cover
        raise NotImplementedError


class BraveSearch(_Base):
    name = "brave"

    def __init__(self, api_key: str, **kw):
        super().__init__(**kw)
        self._key = api_key

    def search(self, query, limit=8):
        d = self._json("https://api.search.brave.com/res/v1/web/search", {"q": query, "count": min(limit, 20)},
                       {"X-Subscription-Token": self._key, "Accept": "application/json"})
        rows = ((d or {}).get("web") or {}).get("results") or []
        return [SearchHit(r["url"], r.get("title", ""), r.get("description", ""), i + 1, self.name) for i, r in enumerate(rows[:limit]) if r.get("url")]


class GoogleCSE(_Base):
    name = "google_cse"

    def __init__(self, api_key: str, cx: str, **kw):
        super().__init__(**kw)
        self._key, self._cx = api_key, cx

    def search(self, query, limit=8):
        d = self._json("https://www.googleapis.com/customsearch/v1", {"key": self._key, "cx": self._cx, "q": query, "num": min(limit, 10)})
        rows = (d or {}).get("items") or []
        return [SearchHit(r["link"], r.get("title", ""), r.get("snippet", ""), i + 1, self.name) for i, r in enumerate(rows[:limit]) if r.get("link")]


class SearXNG(_Base):
    name = "searxng"

    def __init__(self, base_url: str, **kw):
        super().__init__(**kw)
        self._base = base_url.rstrip("/")

    def search(self, query, limit=8):
        d = self._json(self._base + "/search", {"q": query, "format": "json"})
        rows = (d or {}).get("results") or []
        return [SearchHit(r["url"], r.get("title", ""), r.get("content", ""), i + 1, self.name) for i, r in enumerate(rows[:limit]) if r.get("url")]


def provider_from_env(env=None, client: httpx.Client | None = None, **kw):
    """-> provider instance. Raises SearchUnavailable (with the reason) if nothing usable is configured."""
    env = os.environ if env is None else env
    want = (env.get("MERIDIAN_SEARCH_PROVIDER") or "").strip().lower()
    have = {"brave": bool(env.get("BRAVE_SEARCH_API_KEY")),
            "google_cse": bool(env.get("GOOGLE_CSE_API_KEY") and env.get("GOOGLE_CSE_CX")),
            "searxng": bool(env.get("SEARXNG_URL"))}
    if want and want not in have:
        raise SearchUnavailable(f"MERIDIAN_SEARCH_PROVIDER={want!r} is not one of brave, google_cse, searxng")
    pick = want or next((k for k, v in have.items() if v), "")
    if not pick or not have.get(pick):
        raise SearchUnavailable("No search provider configured. Set BRAVE_SEARCH_API_KEY, GOOGLE_CSE_API_KEY + GOOGLE_CSE_CX, or SEARXNG_URL "
                                "- or use manual-URL or offline-fixture mode.")
    if pick == "brave":
        return BraveSearch(env["BRAVE_SEARCH_API_KEY"], client=client, **kw)
    if pick == "google_cse":
        return GoogleCSE(env["GOOGLE_CSE_API_KEY"], env["GOOGLE_CSE_CX"], client=client, **kw)
    return SearXNG(env["SEARXNG_URL"], client=client, **kw)


def configured_provider_name(env=None) -> str | None:
    try:
        return provider_from_env(env).name
    except SearchUnavailable:
        return None
