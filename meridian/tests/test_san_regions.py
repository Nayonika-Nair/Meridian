"""A2 persistent regions and A8 grouping / duplicate suppression."""
import json
import sqlite3

import pytest

import fixtures_gen as F
from src.sanitize import Options, Policy, process_document
from src.sanitize.regions import (Region, RegionStore, group_regions, keyed_hash, make_region_id, merge_line_rects,
                                  text_shape)


def mk(rule, rect, start=None, end=None, unit="page:1", space="pdf_unrot_tl", action="REDACT", conf=0.9, rid=None):
    anchor = {} if start is None else {"start": start, "end": end}
    return Region(region_id=rid or make_region_id("d", unit, rule, str(rect)), document_id="d", unit_id=unit, space=space,
                  rects=[rect], anchor=anchor, category=rule.split(".")[0], rule_id=rule, action=action, confidence=conf,
                  severity="HIGH", raw="x")


# ------------------------------------------------------------------------------ A8 grouping
def test_exact_duplicates_are_suppressed_with_provenance():
    a = mk("EMAIL.ADDRESS", (10, 10, 100, 22), rid="R-a")
    b = mk("EMAIL.ADDRESS", (10.5, 10, 100, 22.2), rid="R-b", conf=0.95)
    out = group_regions([a, b])
    assert len(out) == 1 and out[0].merged_from == ["R-b"] and out[0].confidence == 0.95


def test_overlapping_boxes_of_different_rules_become_one_region_with_all_rules_listed():
    a = mk("GOV.US_SSN", (10, 10, 100, 22), rid="R-a")
    b = mk("GOV.TAX_ID_LABELLED", (12, 10, 98, 22), rid="R-b")
    out = group_regions([a, b])
    assert len(out) == 1 and out[0].rule_id == "GOV.US_SSN" and out[0].also_matched == ["GOV.TAX_ID_LABELLED"]


def test_stricter_action_wins_when_regions_merge():
    a = mk("EMAIL.ADDRESS", (10, 10, 100, 22), action="MASK")
    b = mk("PHONE.NUMBER", (20, 10, 90, 22), action="REDACT")
    assert group_regions([a, b])[0].action == "REDACT"


def test_nearby_but_unrelated_regions_are_not_merged():
    email = mk("EMAIL.ADDRESS", (10, 10, 100, 22), start=0, end=20)
    phone = mk("PHONE.NUMBER", (102, 10, 190, 22), start=21, end=35)          # 2pt away, same line, different rule
    other_email = mk("EMAIL.ADDRESS", (194, 10, 280, 22), start=200, end=220)   # same rule but text far apart
    out = group_regions([email, phone, other_email])
    assert len(out) == 3


def test_text_contiguous_same_rule_parts_on_one_line_are_grouped():
    a = mk("PERSON.NAME_LABELLED", (10, 10, 50, 22), start=6, end=10)
    b = mk("PERSON.NAME_LABELLED", (54, 10, 100, 22), start=11, end=16)
    out = group_regions([a, b])
    assert len(out) == 1 and out[0].anchor["start"] == 6 and out[0].anchor["end"] == 16


def test_regions_on_different_units_never_merge():
    a = mk("EMAIL.ADDRESS", (10, 10, 100, 22), unit="page:1")
    b = mk("EMAIL.ADDRESS", (10, 10, 100, 22), unit="page:2")
    assert len(group_regions([a, b])) == 2


def test_merge_line_rects_closes_gaps_only_on_the_same_line():
    rects = [(10, 10, 40, 22), (43, 10, 80, 22), (10, 40, 60, 52)]
    out = merge_line_rects(rects, gap_tol=5)
    assert (10, 10, 80, 22) in out and (10, 40, 60, 52) in out and len(out) == 2


# ------------------------------------------------------------------------------ A2 stable IDs + minimised metadata
def test_region_id_is_stable_and_scoped_to_the_document():
    a = make_region_id("sha-1", "page:1", "GOV.US_SSN", "a")
    assert a == make_region_id("sha-1", "page:1", "GOV.US_SSN", "a")
    assert a != make_region_id("sha-2", "page:1", "GOV.US_SSN", "a") and a.startswith("R-")


def test_keyed_hash_and_shape_do_not_expose_the_value():
    assert keyed_hash(b"k" * 32, "123-45-6789") == keyed_hash(b"k" * 32, "123 45 6789")      # normalised
    assert keyed_hash(b"k" * 32, "123-45-6789") != keyed_hash(b"z" * 32, "123-45-6789")      # salted
    assert text_shape("123-45-6789") == "ddd-dd-dddd" and "1" not in text_shape("Jane Doe 42")


def test_process_twice_gives_identical_region_ids_and_persists_without_raw_text(tmp_path):
    src = tmp_path / "in.pdf"
    F.make_native_pdf(src)
    store = tmp_path / "regions.db"
    ids = []
    for n in range(2):
        res = process_document(src, tmp_path / f"out{n}.pdf", Policy(ocr=False), Options(store_path=str(store)))
        ids.append(sorted(r["region_id"] for r in res["regions"] if r["source_method"] != "verification_rescan"))
    assert ids[0] == ids[1] and ids[0]
    rs = RegionStore(store)
    rows = rs.regions_for(res["input"]["sha256"])
    assert rows and {r["region_id"] for r in rows} >= set(ids[0])
    # every persisted region carries the A2 contract
    for r in rows:
        for k in ("document_id", "unit_id", "rects", "category", "rule_id", "confidence", "source_method", "action",
                  "verification_status", "text_hash", "text_shape"):
            assert k in r
        assert r["verification_status"] in ("verified", "review_required", "not_applicable", "failed")
    raw = open(store, "rb").read()
    for secret in (F.SSN.encode(), F.CARD.encode(), F.EMAIL.encode(), F.ACCT.encode(), F.NAME.encode()):
        assert secret not in raw
    # the result object itself never carries raw values either
    assert F.SSN not in json.dumps(res) and F.EMAIL not in json.dumps(res)


def test_region_store_updates_status_and_rejects_unknown_states(tmp_path):
    src = tmp_path / "in.pdf"
    F.make_native_pdf(src)
    store = RegionStore(tmp_path / "r.db")
    res = process_document(src, tmp_path / "o.pdf", Policy(ocr=False), Options(store_path=store.path))
    rid = res["regions"][0]["region_id"]
    store.update_status(rid, "review_required", "analyst flagged")
    assert store.get_region(rid)["verification_status"] == "review_required"
    with pytest.raises(ValueError):
        store.update_status(rid, "banana")
    assert store.runs() and store.runs()[0]["release_status"] == res["release"]["status"]


def test_sqlite_store_leaves_no_open_handles_or_sidecar_files(tmp_path):
    store = RegionStore(tmp_path / "x.db")
    store.salt
    import os
    # connections are opened per call and closed: the file can be deleted immediately (Windows-safe)
    os.remove(store.path)
    assert not os.path.exists(store.path)
    assert [p.name for p in tmp_path.iterdir() if "journal" in p.name or p.name.endswith(("-wal", "-shm"))] == []
