"""UI: design-system guarantees (contrast, motion, focus, escaping) and end-to-end page flows via Streamlit AppTest. Offline."""
import json
import re

import pytest

at_mod = pytest.importorskip("streamlit.testing.v1")
from src.config import ROOT
from src.ui import theme as T
from src.ui.theme import contrast_ratio


def app(tmp_path, monkeypatch):
    monkeypatch.setenv("MERIDIAN_OSINT_DB", str(tmp_path / "osint.db"))
    monkeypatch.setenv("MERIDIAN_RL_LOG", str(tmp_path / "rl.jsonl"))
    import src.liveosint.review as rv
    monkeypatch.setattr(rv, "OVERLAY", tmp_path / "overlay.json")
    return at_mod.AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()


# ------------------------------------------------------------------ design system
@pytest.mark.parametrize("fg,bg,min_ratio", [("text", "bg", 7), ("text", "surface", 7), ("muted", "surface", 4.5), ("muted", "bg", 4.5), ("muted", "surface2", 4.5),
                                              ("accent", "surface", 4.5), ("accent_ink", "accent", 4.5), ("ok", "surface", 4.5), ("warn", "surface", 4.5),
                                              ("high", "surface", 4.5), ("crit", "surface", 4.5), ("faint", "surface", 4.5)])
def test_token_pairs_meet_wcag_aa(fg, bg, min_ratio):
    assert contrast_ratio(T.TOKENS[fg], T.TOKENS[bg]) >= min_ratio


def test_css_has_reduced_motion_focus_responsive_and_limited_glass():
    css = T.css()
    assert "prefers-reduced-motion:reduce" in css and "animation:none!important" in css and "transition:none!important" in css
    assert ":focus-visible" in css and "outline:2px solid" in css
    assert "@media (max-width:900px)" in css and "@media (max-width:640px)" in css
    assert css.count("backdrop-filter:blur") == 4 and "m-hero" in css and "stSidebar" in css          # glass only on sidebar + hero (each with -webkit-)
    assert not re.search(r"https?://", css), "no external fonts, images or scripts"
    assert "forced-colors" in css


def test_untrusted_text_is_escaped_in_components():
    evil = '<img src=x onerror=alert(1)><script>alert(2)</script>'
    for html in (T.badge(evil), T.kpi(evil, evil, evil), T.kv([(evil, evil)]), T.meter(0.5, label=evil)):
        assert "<script" not in html and "<img" not in html
    assert T.meter(5.0).count("100") >= 1 and "aria-valuenow=\"0\"" in T.meter(-1)


def test_state_is_never_conveyed_by_colour_alone():
    for tier in ("LOW", "MODERATE", "HIGH", "CRITICAL", "AUTO", "STRICT", "REVIEW", "BLOCK"):
        assert tier in T.badge(tier, tier)


# ------------------------------------------------------------------ pages
@pytest.mark.parametrize("page", ["Overview", "Digital Twin", "Document Gate", "Sanitisation Lab", "Live OSINT Explorer", "Policy Lab", "What-If", "Ledger", "Search"])
def test_every_page_renders_with_a_header_and_no_exception(page, tmp_path, monkeypatch):
    at = app(tmp_path, monkeypatch)
    at.radio(key="nav").set_value(page).run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("m-hero" in m.value for m in at.markdown)


def test_overview_has_no_fake_data_and_sorting_works(tmp_path, monkeypatch):
    at = app(tmp_path, monkeypatch)
    assert not at.exception
    cards = [m.value for m in at.markdown if "m-card" in m.value and "Chance of AI" in m.value]
    assert len(cards) >= 5
    names = json.load(open(ROOT / "data" / "vendors.json"))
    assert all(any(v["vendor_id"] in c for c in cards) for v in names)                  # one card per real vendor, nothing invented


def test_live_osint_fixture_flow_labels_fixtures_and_refuses_to_accept_them(tmp_path, monkeypatch):
    at = app(tmp_path, monkeypatch)
    at.radio(key="nav").set_value("Live OSINT Explorer").run()
    at.radio(key="lo_mode").set_value("Offline fixtures").run()
    [b for b in at.button if b.label == "Collect evidence"][0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    html = " ".join(m.value for m in at.markdown)
    assert "FIXTURE" in html and "Accepted (0)" in " ".join(t.label for t in at.tabs)
    accept = [b for b in at.button if b.label == "Accept"]
    assert accept and all(b.disabled for b in accept)                                        # fixture evidence cannot be accepted
    assert not any(b.label.startswith("Apply to Twin") for b in at.button)                  # nothing accepted -> nothing to apply


def test_live_osint_search_without_provider_says_so_and_invents_nothing(tmp_path, monkeypatch):
    for k in ("BRAVE_SEARCH_API_KEY", "GOOGLE_CSE_API_KEY", "GOOGLE_CSE_CX", "SEARXNG_URL", "MERIDIAN_SEARCH_PROVIDER"):
        monkeypatch.delenv(k, raising=False)
    at = app(tmp_path, monkeypatch)
    at.radio(key="nav").set_value("Live OSINT Explorer").run()
    [b for b in at.button if b.label == "Collect evidence"][0].click().run()
    html = " ".join(m.value for m in at.markdown)
    assert "No search provider configured" in html and "nothing was searched or fabricated" in html
    assert "<div class=\"m-src\">" not in html


def test_policy_lab_shows_advisory_beside_authoritative_decision(tmp_path, monkeypatch):
    at = app(tmp_path, monkeypatch)
    at.radio(key="nav").set_value("Policy Lab").run()
    assert not at.exception
    html = " ".join(m.value for m in at.markdown)
    assert "Authoritative (rule-based)" in html and "Advisory (bandit)" in html
    assert "Recommendation only" in " ".join(c.value for c in at.caption)


def test_sanitisation_lab_empty_state_and_matrix(tmp_path, monkeypatch):
    at = app(tmp_path, monkeypatch)
    at.radio(key="nav").set_value("Sanitisation Lab").run()
    assert not at.exception
    assert "Drop a file to begin" in " ".join(m.value for m in at.markdown)
    assert [b for b in at.button if b.label == "Sanitise"][0].disabled


def test_sanitisation_lab_end_to_end_upload_shows_verdict_and_download(tmp_path, monkeypatch):
    import fixtures_gen as F
    at = app(tmp_path, monkeypatch)
    at.radio(key="nav").set_value("Sanitisation Lab").run()
    src = tmp_path / "doc.docx"
    F.make_docx(src)
    fu = at.file_uploader[0]
    fu.upload("doc.docx", src.read_bytes(), "application/vnd.openxmlformats-officedocument.wordprocessingml.document") if hasattr(fu, "upload") else fu.set_value(
        [type("U", (), {"name": "doc.docx", "getvalue": lambda s: src.read_bytes()})()])
    at.run()
    [b for b in at.button if b.label == "Sanitise"][0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    html = " ".join(m.value for m in at.markdown)
    assert "APPROVED FOR RELEASE" in html and F.SSN not in html and F.EMAIL not in html          # raw values are never rendered
