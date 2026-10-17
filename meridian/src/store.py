"""Runtime state kept OUT of the supplied data files.

  ledger/evidence_overlay.json   analyst-added evidence (merged over data/evidence.json)
  ledger/approved_policies.json  self-healing policy changes a human approved

Nothing here is ever written by simulations; only explicit UI approvals call these writers.
"""
import json, time
from pathlib import Path
from .config import ROOT

OVERLAY = ROOT / 'ledger' / 'evidence_overlay.json'
POLICIES = ROOT / 'ledger' / 'approved_policies.json'


def _read(path, default):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _write(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, ensure_ascii=False))


def load_overlay(path=None): return _read(path or OVERLAY, {})


def add_evidence(vendor_id, row, path=None):
    ov = load_overlay(path)
    ov.setdefault(vendor_id, []).append(row)
    _write(path or OVERLAY, ov)


def load_policies(path=None): return _read(path or POLICIES, {})


def approved_policy(vendor_id, path=None):
    """The latest approved policy for a vendor, or None."""
    return load_policies(path).get(vendor_id)


def approve_policy(vendor_id, proposal, approver='analyst', path=None):
    pol = load_policies(path)
    pol[vendor_id] = {**proposal, 'approved_by': approver, 'approved_at': time.time()}
    _write(path or POLICIES, pol)
    return pol[vendor_id]


def revoke_policy(vendor_id, path=None):
    pol = load_policies(path)
    pol.pop(vendor_id, None)
    _write(path or POLICIES, pol)
