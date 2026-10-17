"""B3: SSRF guard, robots.txt, rate limits, retries, size/type caps. No network: MockTransport + fake DNS."""
import httpx
import pytest

from liveosint_helpers import VENDOR, Clock, page, resolver, site
from src.liveosint import FixtureFetcher, SafeFetcher, UrlBlocked, check_url
from src.liveosint.urlguard import ip_is_public

H = "www.acme-test.com"


# --------------------------------------------------------------------------- URL guard
@pytest.mark.parametrize("url,code", [
    ("http://127.0.0.1/", "non_public_address"), ("http://10.1.2.3/x", "non_public_address"), ("http://192.168.0.1/", "non_public_address"),
    ("http://172.16.5.5/", "non_public_address"), ("http://169.254.169.254/latest/meta-data", "non_public_address"),
    ("http://[::1]/", "non_public_address"), ("http://[::ffff:127.0.0.1]/", "non_public_address"), ("http://100.64.0.1/", "non_public_address"),
    ("http://0.0.0.0/", "non_public_address"), ("http://[fe80::1]/", "non_public_address"), ("http://[64:ff9b::7f00:1]/", "non_public_address"),
    ("http://2130706433/", "obfuscated_ip"), ("http://0x7f.1/", "obfuscated_ip"), ("http://017700000001/", "obfuscated_ip"),
    ("http://localhost/", "internal_hostname"), ("http://intranet/", "internal_hostname"), ("http://printer.local/", "internal_hostname"),
    ("file:///etc/passwd", "scheme_not_allowed"), ("ftp://www.acme-test.com/", "scheme_not_allowed"), ("gopher://x/", "scheme_not_allowed"),
    ("http://user:pw@www.acme-test.com/", "credentials_in_url"), ("http://www.acme-test.com:8080/", "port_not_allowed"),
    ("http://www.acme-test.com:22/", "port_not_allowed"), ("http://evil.example.net/", "non_public_address"),
    ("http://nonexistent.invalid.test/", "dns_failure"), ("http://www.acme-test.com/a b", "bad_url"), ("", "bad_url"), ("http:///x", "bad_url"),
])
def test_blocked_urls(url, code):
    with pytest.raises(UrlBlocked) as e:
        check_url(url, resolver=resolver)
    assert e.value.code == code


def test_public_url_allowed_and_mixed_resolution_blocked():
    t = check_url(f"https://{H}/a?b=1#f", resolver=resolver)
    assert t.host == H and t.port == 443
    with pytest.raises(UrlBlocked):                                                   # one private answer is enough to refuse (rebinding mix)
        check_url("https://x.example.com/", resolver=lambda h, p: ["8.8.8.8", "10.0.0.1"])


def test_ip_classifier_edge_cases():
    assert ip_is_public("8.8.8.8") and ip_is_public("2606:4700:4700::1111")
    assert not any(ip_is_public(x) for x in ["::ffff:10.0.0.1", "224.0.0.1", "240.0.0.1", "192.0.2.1", "not-an-ip"])


def test_allow_private_is_explicit_and_off_by_default():
    with pytest.raises(UrlBlocked):
        check_url("http://127.0.0.1:8000/")
    assert check_url("http://127.0.0.1:8000/", allow_private=True).port == 8000


# --------------------------------------------------------------------------- fetcher
def mk(routes, robots="User-agent: *\nAllow: /\n", **kw):
    clk = Clock()
    tr = site(routes, robots)
    kw.setdefault("min_interval", 0.0)
    f = SafeFetcher(transport=tr, resolver=resolver, sleep=clk.sleep, clock=clk.now, **kw)
    return f, tr, clk


def test_successful_fetch_has_provenance_fields():
    f, tr, _ = mk({(H, "/"): (200, {}, page("acme/index.html"))})
    r = f.fetch(f"https://{H}/")
    assert r.ok and r.status == 200 and r.content_type == "text/html" and len(r.sha256) == 64 and r.n_bytes > 0
    assert r.robots == "allowed" and r.origin == "live" and r.final_url == f"https://{H}/"
    f.close()


def test_redirect_to_private_address_is_blocked_on_the_hop():
    f, tr, _ = mk({(H, "/go"): (302, {"location": "http://169.254.169.254/latest/meta-data"}, "")})
    r = f.fetch(f"https://{H}/go")
    assert not r.ok and r.error == "blocked:non_public_address"
    assert not any("169.254" in c for c in tr.calls)                                # never even requested


def test_redirect_to_disallowed_scheme_and_internal_host_blocked():
    for loc, code in [("file:///etc/passwd", "blocked:scheme_not_allowed"), ("http://localhost/admin", "blocked:internal_hostname"),
                      ("http://evil.example.net/", "blocked:non_public_address")]:
        f, _, _ = mk({(H, "/go"): (301, {"location": loc}, "")})
        assert f.fetch(f"https://{H}/go").error == code


def test_redirects_are_followed_and_bounded():
    f, _, _ = mk({(H, "/a"): (301, {"location": "/b"}, ""), (H, "/b"): (200, {}, "<p>AI-powered</p>")})
    r = f.fetch(f"https://{H}/a")
    assert r.ok and r.final_url.endswith("/b") and r.redirects == [f"https://{H}/b"]
    routes = {(H, f"/r{i}"): (302, {"location": f"/r{i + 1}"}, "") for i in range(20)}
    f, _, _ = mk(routes, max_redirects=3)
    assert f.fetch(f"https://{H}/r0").error == "too_many_redirects"
    f, _, _ = mk({(H, "/l1"): (302, {"location": "/l2"}, ""), (H, "/l2"): (302, {"location": "/l1"}, "")})
    assert f.fetch(f"https://{H}/l1").error == "redirect_loop"


def test_robots_disallow_is_respected_and_target_never_requested():
    f, tr, _ = mk({(H, "/private/x"): (200, {}, "secret")}, robots="User-agent: *\nDisallow: /private/\n")
    r = f.fetch(f"https://{H}/private/x")
    assert r.error == "robots_disallowed" and r.robots == "disallowed"
    assert not any(c.endswith("/private/x") for c in tr.calls)
    f2, _, _ = mk({(H, "/ok"): (200, {}, "<p>x</p>")}, robots="User-agent: *\nDisallow: /private/\n")
    assert f2.fetch(f"https://{H}/ok").ok


def test_robots_agent_specific_rule_and_site_wide_block():
    f, _, _ = mk({(H, "/x"): (200, {}, "x")}, robots="User-agent: MeridianTPRM-OSINT\nDisallow: /\n")
    assert f.fetch(f"https://{H}/x").error == "robots_disallowed"


def test_robots_server_error_is_treated_as_disallow_but_404_as_allow():
    f, _, _ = mk({(H, "/robots.txt"): (503, {}, ""), (H, "/x"): (200, {}, "x")}, retries=0)
    assert f.fetch(f"https://{H}/x").error == "robots_unavailable"
    f, _, _ = mk({(H, "/robots.txt"): (404, {}, ""), (H, "/x"): (200, {}, "<p>x</p>")})
    r = f.fetch(f"https://{H}/x")
    assert r.ok and r.robots == "unavailable_allowed"


def test_robots_can_be_switched_off_only_explicitly():
    f, tr, _ = mk({(H, "/p/x"): (200, {}, "<p>x</p>")}, robots="User-agent: *\nDisallow: /\n", respect_robots=False)
    assert f.fetch(f"https://{H}/p/x").ok and not any("robots.txt" in c for c in tr.calls)


def test_per_domain_rate_limit_spaces_requests_and_crawl_delay_is_honoured():
    f, tr, clk = mk({(H, "/1"): (200, {}, "<p>a</p>"), (H, "/2"): (200, {}, "<p>b</p>")}, min_interval=2.0)
    f.fetch(f"https://{H}/1")
    t1 = clk.now()
    f.fetch(f"https://{H}/2")
    times = [r["t"] for r in f.request_log if r["url"].endswith(("/1", "/2"))]
    assert times[1] - times[0] >= 2.0 - 1e-9 and clk.slept
    f, _, clk = mk({(H, "/1"): (200, {}, "<p>a</p>"), (H, "/2"): (200, {}, "<p>b</p>")}, robots="User-agent: *\nCrawl-delay: 5\n", min_interval=0.5)
    f.fetch(f"https://{H}/1")
    f.fetch(f"https://{H}/2")
    t = [r["t"] for r in f.request_log if r["url"].endswith(("/1", "/2"))]
    assert t[1] - t[0] >= 5.0 - 1e-9


def test_bounded_retries_on_503_then_success_and_on_exhaustion():
    state = {"n": 0}

    def flaky(req):
        state["n"] += 1
        return httpx.Response(503, headers={"retry-after": "1"}) if state["n"] < 3 else httpx.Response(200, headers={"content-type": "text/html"}, text="<p>AI</p>")

    f, _, clk = mk({(H, "/f"): flaky}, retries=2)
    r = f.fetch(f"https://{H}/f")
    assert r.ok and r.attempts == 3 and state["n"] == 3 and clk.slept
    state["n"] = -100
    f, _, _ = mk({(H, "/f"): flaky}, retries=2)
    r = f.fetch(f"https://{H}/f")
    assert not r.ok and r.error == "http_503" and r.attempts == 3                    # bounded: 1 + 2 retries, never infinite


def test_retry_after_is_capped():
    def slow(req):
        return httpx.Response(429, headers={"retry-after": "99999"})
    f, _, clk = mk({(H, "/s"): slow}, retries=1, max_retry_after=3.0)
    f.fetch(f"https://{H}/s")
    assert max(clk.slept) <= 3.0


def test_non_retryable_status_is_not_retried():
    n = {"n": 0}

    def gone(req):
        n["n"] += 1
        return httpx.Response(404, text="x")
    f, _, _ = mk({(H, "/g"): gone}, retries=3)
    r = f.fetch(f"https://{H}/g")
    assert r.error == "http_404" and n["n"] == 1


def test_timeout_and_network_errors_are_reported_after_bounded_retries():
    def boom(req):
        raise httpx.ConnectTimeout("t")
    f, _, _ = mk({(H, "/t"): boom}, retries=1)
    r = f.fetch(f"https://{H}/t")
    assert r.error == "timeout" and r.attempts == 2

    def down(req):
        raise httpx.ConnectError("refused")
    f, _, _ = mk({(H, "/n"): down}, retries=0)
    assert f.fetch(f"https://{H}/n").error == "network"


def test_content_type_allow_list():
    f, _, _ = mk({(H, "/bin"): (200, {"content-type": "application/pdf"}, b"%PDF"), (H, "/img"): (200, {"content-type": "image/png"}, b"x")})
    assert f.fetch(f"https://{H}/bin").error == "bad_content_type"
    assert f.fetch(f"https://{H}/img").error == "bad_content_type"


def test_size_caps_by_header_and_by_streamed_body():
    f, _, _ = mk({(H, "/big"): (200, {"content-length": "5000000"}, b"x" * 10)}, max_bytes=1000)
    assert f.fetch(f"https://{H}/big").error == "too_large"
    f, _, _ = mk({(H, "/chunk"): (200, {}, b"<p>" + b"a" * 50_000 + b"</p>")}, max_bytes=10_000)
    r = f.fetch(f"https://{H}/chunk")
    assert r.error == "too_large" and r.text == ""


def test_declared_charset_is_decoded():
    f, _, _ = mk({(H, "/l1"): (200, {"content-type": "text/html; charset=iso-8859-1"}, "<p>caf\xe9 AI</p>".encode("latin-1"))})
    assert "café" in f.fetch(f"https://{H}/l1").text


def test_untrusted_content_is_returned_as_text_only():
    f, _, _ = mk({(H, "/x"): (200, {}, "<script>alert(1)</script><p>hi</p>")})
    r = f.fetch(f"https://{H}/x")
    assert isinstance(r.text, str) and r.ok


# --------------------------------------------------------------------------- fixtures
def test_fixture_fetcher_is_labelled_and_cannot_escape_its_directory():
    from liveosint_helpers import FIX
    ff = FixtureFetcher(FIX)
    r = ff.fetch("fixture://acme/subprocessors.html")
    assert r.ok and r.origin == "fixture" and "OpenAI" in r.text
    assert ff.fetch("fixture://../../etc/passwd").error.startswith("blocked")
    assert ff.fetch("https://www.acme-test.com/").error == "not_a_fixture_url"
    assert ff.fetch("fixture://acme/missing.html").error == "http_404"


def test_real_local_http_server_end_to_end(tmp_path):
    """Real sockets against 127.0.0.1 - allowed ONLY because allow_private is passed explicitly in the test."""
    import functools, http.server, threading
    (tmp_path / "p.html").write_text("<p>Our AI-powered assistant.</p>")
    (tmp_path / "robots.txt").write_text("User-agent: *\nDisallow: /secret\n")
    (tmp_path / "secret").write_text("no")
    h = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(tmp_path))
    h.log_message = lambda *a, **k: None
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), h)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        port = srv.server_address[1]
        with SafeFetcher(allow_private=True, min_interval=0) as f:
            assert f.fetch(f"http://127.0.0.1:{port}/p.html").ok
            assert f.fetch(f"http://127.0.0.1:{port}/secret").error == "robots_disallowed"
        with SafeFetcher(min_interval=0) as f:                                        # default: refused
            assert f.fetch(f"http://127.0.0.1:{port}/p.html").error.startswith("blocked:")
    finally:
        srv.shutdown()
        srv.server_close()
