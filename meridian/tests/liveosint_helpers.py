"""Shared helpers: public-looking fake DNS, scripted httpx MockTransport sites, a fake clock."""
from pathlib import Path

import httpx

FIX = Path(__file__).parent / "fixtures" / "liveosint"
VENDOR = {"vendor_id": "V-T1", "vendor_name": "Acme", "website": "www.acme-test.com", "criticality": "High", "criticality_factor": 0.75,
          "category": "SaaS", "business_process": "x", "dependency": "High", "data_classification": "x", "annual_volume": "x",
          "criticality_rationale": "x"}
PUBLIC = {"www.acme-test.com": ["93.184.216.34"], "acme-test.com": ["93.184.216.34"], "news.example.org": ["198.51.100.0"] if False else ["8.8.4.4"],
          "evil.example.net": ["10.0.0.5"], "api.search.test": ["8.8.8.8"]}


def resolver(host, port):
    if host in PUBLIC:
        return PUBLIC[host]
    raise OSError("NXDOMAIN " + host)


def page(name):
    return (FIX / name).read_text()


class Clock:
    def __init__(self):
        self.t = 1000.0
        self.slept = []

    def now(self):
        return self.t

    def sleep(self, s):
        self.slept.append(s)
        self.t += s


def site(routes, robots="User-agent: *\nAllow: /\n"):
    """routes: {(host,path): (status, headers, body) or callable(request)->Response}"""
    calls = []

    def handler(req: httpx.Request):
        calls.append(str(req.url))
        key = (req.url.host, req.url.path)
        if req.url.path == "/robots.txt" and key not in routes:
            return httpx.Response(200, text=robots, headers={"content-type": "text/plain"}) if robots is not None else httpx.Response(404)
        r = routes.get(key)
        if r is None:
            return httpx.Response(404, text="nope")
        if callable(r):
            return r(req)
        st, hd, body = r
        return httpx.Response(st, headers={"content-type": "text/html", **hd}, content=body if isinstance(body, bytes) else body.encode())

    t = httpx.MockTransport(handler)
    t.calls = calls
    return t
