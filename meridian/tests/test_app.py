"""Headless UI test: every page renders without exceptions."""
import pytest
at_mod = pytest.importorskip('streamlit.testing.v1')
from src.config import ROOT
PAGES = ['Overview', 'Digital Twin', 'Document Gate', 'Sanitisation Lab', 'Live OSINT Explorer', 'Policy Lab', 'What-If', 'Ledger', 'Search']

@pytest.mark.parametrize('page', PAGES)
def test_page_renders(page):
    at = at_mod.AppTest.from_file(str(ROOT / 'app.py'), default_timeout=120).run()
    at.radio(key='nav').set_value(page).run()
    assert not at.exception, [e.value for e in at.exception]
