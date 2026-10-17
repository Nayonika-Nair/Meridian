import sys; from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))


import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate_runtime_state(tmp_path, monkeypatch):
    """No test may write to the project's ledger/ folder: evidence DB and advisory log go to a temp dir."""
    monkeypatch.setenv("MERIDIAN_OSINT_DB", str(tmp_path / "_osint.db"))
    monkeypatch.setenv("MERIDIAN_RL_LOG", str(tmp_path / "_rl.jsonl"))
