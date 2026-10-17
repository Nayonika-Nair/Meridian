"""Responsible HTTP collection: robots.txt, per-domain rate limit, bounded retries, content-type + size caps,
manual redirects with SSRF re-validation on every hop, and an explicit offline-fixture fetcher.

Fetched bytes are untrusted data: decoded as text, never executed, never rendered as HTML by the app.
"""
from __future__ import annotations

import hashlib
import re
import time
import urllib.robotparser
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urljoin, urldefrag, urlsplit

import httpx

from .urlguard import UrlBlocked, check_url

USER_AGENT = "MeridianTPRM-OSINT/2.0 (+analyst-initiated; respects robots.txt)"
ROBOTS_TOKEN = "MeridianTPRM-OSINT"
TEXT_TYPES = ("text/html", "application/xhtml+xml", "text/plain", "application/json", "application/xml", "text/xml")
REDIRECTS = (301, 302, 303, 307, 308)
RETRY_STATUS = (429, 502, 503, 504)


@dataclass
class FetchResult:
    url: str
    final_url: str = ""
    ok: bool = False
    status: int | None = None
    content_type: str = ""
    text: str = ""
    sha256: str = ""
    n_bytes: int = 0
    fetched_at: float = 0.0
    origin: str = "live"                    # live | cache | fixture
    error: str = ""                         # machine code: blocked:<code>, robots_disallowed, too_large, bad_content_type, http_<n>, timeout, network, redirect_loop ...
    detail: str = ""
    robots: str = "n/a"                     # allowed | disallowed | unavailable_allowed | unavailable_disallowed | n/a
    redirects: list = field(default_factory=list)
    attempts: int = 0
    age_s: float = 0.0                      # set for cache hits
    stale: bool = False


class SafeFetcher:
    def __init__(self, *, transport=None, client: httpx.Client | None = None, timeout: float = 10.0, max_bytes: int = 1_000_000,
                 max_redirects: int = 4, retries: int = 2, backoff: float = 0.5, max_retry_after: float = 10.0,
                 min_interval: float = 1.0, resolver=None, allow_private: bool = False, respect_robots: bool = True,
                 sleep=time.sleep, clock=time.monotonic, wall=time.time, user_agent: str = USER_AGENT, robots_ttl: float = 3600.0,
                 content_types=TEXT_TYPES):
        self._client = client or httpx.Client(transport=transport, timeout=timeout, follow_redirects=False,
                                              headers={"User-Agent": user_agent, "Accept": "text/html,text/plain;q=0.9,*/*;q=0.1"})
        self._own_client = client is None
        self.max_bytes, self.max_redirects, self.retries = max_bytes, max_redirects, retries
        self.backoff, self.max_retry_after, self.min_interval = backoff, max_retry_after, min_interval
        self.resolver, self.allow_private, self.respect_robots = resolver, allow_private, respect_robots
        self.sleep, self.clock, self.wall, self.robots_ttl, self.content_types = sleep, clock, wall, robots_ttl, tuple(content_types)
        self._last: dict[str, float] = {}
        self._robots: dict[str, tuple[float, urllib.robotparser.RobotFileParser | None, str]] = {}
        self.request_log: list[dict] = []

    # ---------------------------------------------------------------- lifecycle
    def close(self):
        if self._own_client:
            self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    # ---------------------------------------------------------------- rate limit
    def _throttle(self, host: str, extra: float = 0.0):
        gap = max(self.min_interval, extra)
        last = self._last.get(host)
        if last is not None:
            wait = last + gap - self.clock()
            if wait > 0:
                self.sleep(wait)
        self._last[host] = self.clock()

    # ---------------------------------------------------------------- single guarded GET with retries
    def _get(self, url: str, *, max_bytes: int, crawl_delay: float = 0.0):
        """-> (status, headers, body_bytes, attempts, err_code, err_detail). Never follows redirects."""
        attempts, delay = 0, self.backoff
        while True:
            attempts += 1
            tgt = check_url(url, resolver=self.resolver, allow_private=self.allow_private)
            self._throttle(tgt.host, crawl_delay)
            self.request_log.append({"url": url, "t": self.clock()})
            try:
                with self._client.stream("GET", url) as r:
                    status, headers = r.status_code, r.headers
                    if status in REDIRECTS:
                        return status, headers, b"", attempts, "", ""
                    if status in RETRY_STATUS and attempts <= self.retries:
                        ra = headers.get("retry-after", "")
                        wait = min(float(ra), self.max_retry_after) if ra.replace(".", "", 1).isdigit() else delay
                        self.sleep(max(0.0, wait))
                        delay *= 2
                        continue
                    ctype = headers.get("content-type", "").split(";")[0].strip().lower()
                    if status == 200 and self.content_types and ctype and not any(ctype == c or ctype.endswith("+xml") for c in self.content_types):
                        return status, headers, b"", attempts, "bad_content_type", ctype
                    cl = headers.get("content-length", "")
                    if cl.isdigit() and int(cl) > max_bytes:
                        return status, headers, b"", attempts, "too_large", f"content-length {cl} > {max_bytes}"
                    buf = bytearray()
                    for chunk in r.iter_bytes():                           # decoded bytes: also bounds decompression bombs
                        buf += chunk
                        if len(buf) > max_bytes:
                            return status, headers, b"", attempts, "too_large", f"body exceeded {max_bytes} bytes"
                    return status, headers, bytes(buf), attempts, "", ""
            except httpx.TimeoutException:
                if attempts <= self.retries:
                    self.sleep(delay)
                    delay *= 2
                    continue
                return None, {}, b"", attempts, "timeout", "request timed out"
            except httpx.HTTPError as exc:
                if attempts <= self.retries:
                    self.sleep(delay)
                    delay *= 2
                    continue
                return None, {}, b"", attempts, "network", type(exc).__name__

    # ---------------------------------------------------------------- robots.txt
    def robots_state(self, url: str):
        """-> (allowed: bool, state: str, crawl_delay: float)"""
        sp = urlsplit(url)
        origin = f"{sp.scheme}://{sp.netloc}"
        now = self.clock()
        hit = self._robots.get(origin)
        if hit is None or now - hit[0] > self.robots_ttl:
            parser, state = None, "unavailable_allowed"
            try:
                target = origin + "/robots.txt"
                for _hop in range(3):                                      # http->https / www redirects are common for robots.txt
                    status, hdrs, body, _a, err, _d = self._get(target, max_bytes=500_000)
                    if status in REDIRECTS and hdrs.get("location"):
                        target = urljoin(target, hdrs["location"])
                        continue
                    break
                if err or status is None or status >= 500 or status in REDIRECTS:
                    state = "unavailable_disallowed"                        # RFC 9309: unreachable / server error -> assume complete disallow
                elif status == 200:
                    parser = urllib.robotparser.RobotFileParser()
                    parser.parse(body.decode("utf-8", "replace").splitlines())
                    state = "allowed"
                else:                                                       # 4xx -> no restrictions
                    state = "unavailable_allowed"
            except UrlBlocked:
                state = "unavailable_disallowed"
            hit = (now, parser, state)
            self._robots[origin] = hit
        _, parser, state = hit
        if state == "unavailable_disallowed":
            return False, state, 0.0
        if parser is None:
            return True, state, 0.0
        path = sp.path or "/"
        if sp.query:
            path += "?" + sp.query
        ok = parser.can_fetch(ROBOTS_TOKEN, origin + path)
        cd = parser.crawl_delay(ROBOTS_TOKEN) or 0
        return ok, ("allowed" if ok else "disallowed"), min(float(cd), 10.0)

    # ---------------------------------------------------------------- public API
    def fetch(self, url: str) -> FetchResult:
        res = FetchResult(url=url, fetched_at=self.wall())
        cur = urldefrag(url)[0]
        try:
            for hop in range(self.max_redirects + 1):
                check_url(cur, resolver=self.resolver, allow_private=self.allow_private)
                delay = 0.0
                if self.respect_robots:
                    ok, state, delay = self.robots_state(cur)
                    res.robots = state
                    if not ok:
                        res.error, res.detail, res.final_url = "robots_disallowed" if state == "disallowed" else "robots_unavailable", cur, cur
                        return res
                status, headers, body, attempts, err, detail = self._get(cur, max_bytes=self.max_bytes, crawl_delay=delay)
                res.attempts += attempts
                res.status, res.final_url = status, cur
                if err:
                    res.error, res.detail = err, detail
                    return res
                if status in REDIRECTS:
                    loc = headers.get("location")
                    if not loc:
                        res.error, res.detail = "bad_redirect", "no Location header"
                        return res
                    nxt = urldefrag(urljoin(cur, loc))[0]
                    res.redirects.append(nxt)
                    if nxt in res.redirects[:-1] or nxt == cur:
                        res.error, res.detail = "redirect_loop", nxt
                        return res
                    cur = nxt
                    continue
                res.content_type = headers.get("content-type", "").split(";")[0].strip().lower()
                if status != 200:
                    res.error, res.detail = f"http_{status}", ""
                    return res
                m = re.search(r"charset=([\w\-]+)", headers.get("content-type", ""), re.I)
                try:
                    text = body.decode(m.group(1) if m else "utf-8", "replace")
                except LookupError:
                    text = body.decode("utf-8", "replace")
                res.text, res.n_bytes, res.ok = text, len(body), True
                res.sha256 = hashlib.sha256(body).hexdigest()
                return res
            res.error, res.detail = "too_many_redirects", str(self.max_redirects)
            return res
        except UrlBlocked as exc:
            res.error, res.detail = f"blocked:{exc.code}", exc.detail
            res.final_url = res.final_url or cur
            return res


class FixtureFetcher:
    """Explicit offline mode: serves files from a local directory for fixture://<name> URLs. Results are labelled
    origin='fixture' so nothing downstream can mistake them for retrieved evidence."""

    def __init__(self, root):
        self.root = Path(root).resolve()

    def close(self):
        pass

    def fetch(self, url: str) -> FetchResult:
        res = FetchResult(url=url, final_url=url, origin="fixture", fetched_at=time.time())
        sp = urlsplit(url)
        if sp.scheme != "fixture":
            res.error, res.detail = "not_a_fixture_url", url
            return res
        rel = (sp.netloc + sp.path).lstrip("/")
        p = (self.root / rel).resolve()
        if self.root not in p.parents and p != self.root:
            res.error, res.detail = "blocked:path_traversal", rel
            return res
        if not p.is_file():
            res.error, res.detail, res.status = "http_404", rel, 404
            return res
        data = p.read_bytes()
        res.text, res.n_bytes, res.ok, res.status = data.decode("utf-8", "replace"), len(data), True, 200
        res.content_type = "text/html" if p.suffix in (".html", ".htm") else "text/plain"
        res.sha256 = hashlib.sha256(data).hexdigest()
        return res
