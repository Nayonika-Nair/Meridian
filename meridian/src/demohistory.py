"""Synthetic spend history for demos (privacy debt / time-to-failure need history to show anything).

Written to a SEPARATE database (ledger/demo_history.db), never the live ledger, and every row carries
meta.synthetic = true. It is illustrative only, not measured Meridian activity."""
import os, random, time
from pathlib import Path
from .config import ROOT
from .ledger import Ledger

DEMO_DB = str(ROOT / 'ledger' / 'demo_history.db')
LIVE_DB = str(ROOT / 'ledger' / 'ledger.db')
DAYS = 75
INTENSITY = {'V-001': 0.5, 'V-002': 1.7, 'V-003': 2.4, 'V-004': 0.6, 'V-005': 0.8, 'V-006': 1.6}   # relative daily burn
MONTHLY_GROWTH = 1.5                                                                           # burn grows over the period


def seed_demo_history(path=DEMO_DB, vendors=None, now=None, seed=7):
    """(Re)build the synthetic history. Deletes only the demo DB file."""
    for suffix in ('', '-wal', '-shm'):
        Path(path + suffix).unlink(missing_ok=True)
    from .demo import load
    vendors = vendors or load()[0]; now = time.time() if now is None else now; rng = random.Random(seed)
    led = Ledger(path); events = []
    for v in vendors:
        k = INTENSITY.get(v['vendor_id'], 1.0)
        for d in range(DAYS, 0, -1):
            growth = MONTHLY_GROWTH ** ((DAYS - d) / 30)
            for _ in range(rng.randint(1, 3)):
                events.append((now - d * 86400 + rng.uniform(0, 80000), v['vendor_name'], round(k * 1.4 * growth * rng.uniform(0.5, 1.5), 2), v['vendor_id']))
    events.sort()
    bal = {}
    for ts, name, cost, vid in events:
        before = bal.get(name, 1000.0); bal[name] = before - cost
        led.append(name, f'SYNTH-{vid}-{int(ts)}', cost, before, before - cost, {'synthetic': True, 'decision': 'AUTO'}, ts=ts)
    led.conn.close()
    return path


def open_ledger(mode):
    if mode == 'demo':
        if not os.path.exists(DEMO_DB):
            seed_demo_history()
        return Ledger(DEMO_DB)
    return Ledger(LIVE_DB)
