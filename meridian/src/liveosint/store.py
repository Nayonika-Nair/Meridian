"""SQLite store for collected evidence, analyst reviews (append-only log), run summaries and the fetch cache.
Connections are opened per call and always closed (Windows-safe)."""
from __future__ import annotations

import json
import os
import sqlite3
import time
from contextlib import closing
from pathlib import Path

from ..config import ROOT
from .fetcher import FetchResult

DEFAULT_DB = str(ROOT / "ledger" / "liveosint.db")
STATUSES = ("pending", "accepted", "rejected", "needs_review")
SCHEMA = """
CREATE TABLE IF NOT EXISTS evidence (evidence_id TEXT PRIMARY KEY, vendor_id TEXT NOT NULL, data TEXT NOT NULL,
  review_status TEXT NOT NULL DEFAULT 'pending', promoted_class TEXT, observation_ref TEXT, reviewer TEXT, reviewed_at REAL,
  review_note TEXT, accepted_age_days INTEGER, first_seen REAL, last_seen REAL, run_id TEXT);
CREATE INDEX IF NOT EXISTS ev_vendor ON evidence(vendor_id);
CREATE TABLE IF NOT EXISTS reviews (id INTEGER PRIMARY KEY AUTOINCREMENT, evidence_id TEXT, vendor_id TEXT, decision TEXT, reviewer TEXT,
  ts REAL, note TEXT, observation_ref TEXT, promoted_class TEXT);
CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, vendor_id TEXT, started REAL, finished REAL, mode TEXT, categories TEXT, summary TEXT);
CREATE TABLE IF NOT EXISTS cache (url TEXT PRIMARY KEY, final_url TEXT, fetched_at REAL, status INTEGER, content_type TEXT, sha256 TEXT, n_bytes INTEGER, text TEXT);
"""


class Store:
    def __init__(self, path: str | Path | None = None):
        self.path = str(path or os.environ.get("MERIDIAN_OSINT_DB") or DEFAULT_DB)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._c() as c:
            c.executescript(SCHEMA)

    def _c(self):
        con = sqlite3.connect(self.path, timeout=10)
        con.row_factory = sqlite3.Row
        return _Ctx(con)

    # ---------------------------------------------------------------- evidence
    def upsert_candidates(self, cands, run_id: str, now: float | None = None) -> dict:
        now = time.time() if now is None else now
        new = seen = 0
        with self._c() as c:
            for cand in cands:
                d = cand.to_dict() if hasattr(cand, "to_dict") else dict(cand)
                row = c.execute("SELECT 1 FROM evidence WHERE evidence_id=?", (d["evidence_id"],)).fetchone()
                if row:
                    c.execute("UPDATE evidence SET data=?, last_seen=?, run_id=? WHERE evidence_id=?", (json.dumps(d, sort_keys=True), now, run_id, d["evidence_id"]))
                    seen += 1
                else:
                    c.execute("INSERT INTO evidence(evidence_id,vendor_id,data,first_seen,last_seen,run_id) VALUES (?,?,?,?,?,?)",
                              (d["evidence_id"], d["vendor_id"], json.dumps(d, sort_keys=True), now, now, run_id))
                    new += 1
        return {"new": new, "already_known": seen}

    @staticmethod
    def _row(r) -> dict:
        d = json.loads(r["data"])
        d.update({"review_status": r["review_status"], "promoted_class": r["promoted_class"], "observation_ref": r["observation_ref"],
                  "reviewer": r["reviewer"], "reviewed_at": r["reviewed_at"], "review_note": r["review_note"],
                  "accepted_age_days": r["accepted_age_days"], "first_seen": r["first_seen"], "last_seen": r["last_seen"], "run_id": r["run_id"]})
        d["effective_class"] = r["promoted_class"] or d["evidence_class"]
        return d

    def evidence(self, vendor_id: str, status: str | None = None, run_id: str | None = None) -> list[dict]:
        q, a = "SELECT * FROM evidence WHERE vendor_id=?", [vendor_id]
        if status:
            q, a = q + " AND review_status=?", a + [status]
        if run_id:
            q, a = q + " AND run_id=?", a + [run_id]
        with self._c() as c:
            return [self._row(r) for r in c.execute(q + " ORDER BY evidence_id", a)]

    def get(self, evidence_id: str) -> dict | None:
        with self._c() as c:
            r = c.execute("SELECT * FROM evidence WHERE evidence_id=?", (evidence_id,)).fetchone()
        return self._row(r) if r else None

    def set_review(self, evidence_id, status, reviewer, note, observation_ref, promoted_class, age_days, now) -> None:
        with self._c() as c:
            c.execute("UPDATE evidence SET review_status=?, reviewer=?, reviewed_at=?, review_note=?, observation_ref=?, promoted_class=?, accepted_age_days=? WHERE evidence_id=?",
                      (status, reviewer, now, note, observation_ref, promoted_class, age_days, evidence_id))
            v = c.execute("SELECT vendor_id FROM evidence WHERE evidence_id=?", (evidence_id,)).fetchone()["vendor_id"]
            c.execute("INSERT INTO reviews(evidence_id,vendor_id,decision,reviewer,ts,note,observation_ref,promoted_class) VALUES (?,?,?,?,?,?,?,?)",
                      (evidence_id, v, status, reviewer, now, note, observation_ref, promoted_class))

    def review_log(self, vendor_id: str) -> list[dict]:
        with self._c() as c:
            return [dict(r) for r in c.execute("SELECT * FROM reviews WHERE vendor_id=? ORDER BY id", (vendor_id,))]

    # ---------------------------------------------------------------- runs
    def save_run(self, run_id, vendor_id, started, finished, mode, categories, summary):
        with self._c() as c:
            c.execute("INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?)", (run_id, vendor_id, started, finished, mode, json.dumps(categories), json.dumps(summary)))

    def runs(self, vendor_id: str) -> list[dict]:
        with self._c() as c:
            return [{**dict(r), "categories": json.loads(r["categories"]), "summary": json.loads(r["summary"])}
                    for r in c.execute("SELECT * FROM runs WHERE vendor_id=? ORDER BY started DESC", (vendor_id,))]

    # ---------------------------------------------------------------- cache (live fetches only; never fixtures)
    def cache_put(self, r: FetchResult) -> None:
        if not r.ok or r.origin != "live":
            return
        with self._c() as c:
            c.execute("INSERT OR REPLACE INTO cache VALUES (?,?,?,?,?,?,?,?)", (r.url, r.final_url, r.fetched_at, r.status, r.content_type, r.sha256, r.n_bytes, r.text))

    def cache_get(self, url: str, max_age_s: float, now: float | None = None, allow_stale: bool = False) -> FetchResult | None:
        """A cache hit is always labelled origin='cache' with its age. Entries older than max_age_s are returned only when
        allow_stale=True, and then carry stale=True so the UI cannot present them as current."""
        now = time.time() if now is None else now
        with self._c() as c:
            r = c.execute("SELECT * FROM cache WHERE url=?", (url,)).fetchone()
        if not r:
            return None
        age = max(0.0, now - r["fetched_at"])
        if age > max_age_s and not allow_stale:
            return None
        return FetchResult(url=r["url"], final_url=r["final_url"], ok=True, status=r["status"], content_type=r["content_type"], text=r["text"],
                           sha256=r["sha256"], n_bytes=r["n_bytes"], fetched_at=r["fetched_at"], origin="cache", age_s=age, stale=age > max_age_s)


class _Ctx:
    def __init__(self, con):
        self.con = con

    def __enter__(self):
        return self.con

    def __exit__(self, et, ev, tb):
        try:
            self.con.commit() if et is None else self.con.rollback()
        finally:
            self.con.close()
        return False
