"""Tamper-evident SQLite ledger: every row's SHA-256 covers the previous row's hash.

`meta` (JSON: decision, receipt hash, synthetic flag, ...) is part of the hash when present,
so rows written by the original prototype (no meta) still verify.
"""
import sqlite3, hashlib, json, time
from pathlib import Path
from .config import ROOT

DEFAULT_DB = str(ROOT / 'ledger' / 'ledger.db')
COLS = ['id', 'destination', 'doc_id', 'cost', 'balance_before', 'balance_after', 'timestamp', 'prev_hash', 'row_hash', 'meta']


def _digest(prev, payload):
    return hashlib.sha256(prev.encode() + json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


class Ledger:
    def __init__(self, db_path=DEFAULT_DB):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.execute('CREATE TABLE IF NOT EXISTS ledger (id INTEGER PRIMARY KEY AUTOINCREMENT,destination TEXT,doc_id TEXT,'
                          'cost REAL,balance_before REAL,balance_after REAL,timestamp REAL,prev_hash TEXT,row_hash TEXT NOT NULL,meta TEXT)')
        if 'meta' not in [r[1] for r in self.conn.execute('PRAGMA table_info(ledger)')]:
            self.conn.execute('ALTER TABLE ledger ADD COLUMN meta TEXT')     # migrate prototype databases
        self.conn.commit()

    def last_hash(self):
        r = self.conn.execute('SELECT row_hash FROM ledger ORDER BY id DESC LIMIT 1').fetchone()
        return r[0] if r else 'GENESIS'

    def append(self, destination, doc_id, cost, before, after, meta=None, ts=None):
        ts = time.time() if ts is None else ts
        prev = self.last_hash()
        payload = {'destination': destination, 'doc_id': doc_id, 'cost': float(cost), 'balance_before': float(before),
                   'balance_after': float(after), 'timestamp': ts, 'prev_hash': prev}
        meta_s = None
        if meta is not None:
            meta_s = json.dumps(meta, sort_keys=True, separators=(',', ':'), default=str)
            payload['meta'] = meta_s
        h = _digest(prev, payload)
        self.conn.execute('INSERT INTO ledger(destination,doc_id,cost,balance_before,balance_after,timestamp,prev_hash,row_hash,meta) '
                          'VALUES (?,?,?,?,?,?,?,?,?)', (destination, doc_id, cost, before, after, ts, prev, h, meta_s))
        self.conn.commit()
        return h

    def rows(self):
        return self.conn.execute('SELECT ' + ','.join(COLS) + ' FROM ledger ORDER BY id DESC').fetchall()

    def records(self, destination=None):
        """Oldest-first list of dicts (meta decoded) for analytics: debt, forecast."""
        q = 'SELECT ' + ','.join(COLS) + ' FROM ledger' + (' WHERE destination=?' if destination else '') + ' ORDER BY id'
        out = []
        for r in self.conn.execute(q, (destination,) if destination else ()):
            d = dict(zip(COLS, r))
            d['meta'] = json.loads(d['meta']) if d['meta'] else {}
            out.append(d)
        return out

    def verify(self):
        prev = 'GENESIS'
        for r in self.conn.execute('SELECT destination,doc_id,cost,balance_before,balance_after,timestamp,prev_hash,row_hash,meta FROM ledger ORDER BY id').fetchall():
            p = {'destination': r[0], 'doc_id': r[1], 'cost': r[2], 'balance_before': r[3], 'balance_after': r[4],
                 'timestamp': r[5], 'prev_hash': r[6]}
            if r[8] is not None:
                p['meta'] = r[8]
            if r[6] != prev or r[7] != _digest(prev, p):
                return False
            prev = r[7]
        return True
