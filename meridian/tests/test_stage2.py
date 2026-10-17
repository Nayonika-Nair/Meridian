import copy, json, math, time, hashlib
import pytest, openpyxl
from src.config import ROOT, DATA
from src.demo import load, vendor_assessment
from src.ledger import Ledger
from src.redaction import load_policy
from src.engine import vendor_state, evaluate_document, commit
from src.debt import privacy_debt, monthly_series, time_to_failure, window_spend, debt_state, DAY
from src.simulate import run_scenario, fire_drill, KINDS
from src.healing import propose_policy
from src import store
from src.scenarios import demo_safe, demo_escalation, demo_hard_rule

VENDORS, EVID = load(); VM = {v['vendor_id']: v for v in VENDORS}; POL = load_policy()
NOW = 1_800_000_000.0

def recs(costs, start=NOW - 10 * DAY):
    return [{'cost': c, 'timestamp': start + i * DAY, 'destination': 'X', 'meta': {}} for i, c in enumerate(costs)]

# ---- source of truth: the workbook ----------------------------------------
def test_vendors_match_workbook():
    ws = openpyxl.load_workbook(DATA / 'Meridian_Vendor_Input.xlsx', data_only=True)['Vendor Inventory']
    rows = {r[1]: r for r in ws.iter_rows(values_only=True) if r[1] and str(r[1]).startswith('V-') and r[1] != 'V-000'}
    assert set(rows) == set(VM) and len(rows) == 6
    for vid, r in rows.items():
        v = VM[vid]; assert v['vendor_name'] == r[2] and v['dependency'] == r[8] and v['data_classification'] == r[9] and v['annual_volume'] == r[10]

# ---- privacy debt ---------------------------------------------------------
def test_privacy_debt_cumulative_and_tier():
    r = recs([100, 200, 300]); assert privacy_debt(r) == 600 and debt_state(r, 1000)['tier'] == 'MODERATE'
    assert debt_state(recs([100]), 1000)['tier'] == 'LOW' and debt_state(recs([2500]), 1000)['tier'] == 'CRITICAL'
def test_monthly_series_is_cumulative():
    s = monthly_series([{'cost': 120, 'timestamp': 1_735_689_600 + 1}, {'cost': 60, 'timestamp': 1_738_368_000 + 1}])  # Jan, Feb 2025
    assert s == [('2025-01', 120, 120), ('2025-02', 60, 180)]
def test_debt_reduces_allowance_via_engine():
    a = vendor_state(VM['V-004'], EVID['V-004'], records=[], trials=1000)['allowance']
    b = vendor_state(VM['V-004'], EVID['V-004'], records=recs([1500]), trials=1000, now=NOW)['allowance']
    assert b < a and b >= a * 0.6 - 1e-6

# ---- time to failure ------------------------------------------------------
def test_ttf_history_basis():
    t = time_to_failure(500, 1000, recs([10] * 8), NOW)
    assert t['basis'] == 'history' and not t['is_projection'] and 40 < t['days'] < 70
def test_ttf_projection_is_labelled_when_history_thin():
    t = time_to_failure(400, 1000, recs([10]), NOW, docs_per_day=5, avg_doc_cost=8)
    assert t['is_projection'] and 'PROJECTION' in t['note'] and math.isclose(t['days'], 10.0)
def test_ttf_volume_doubles_halves_runway():
    a = time_to_failure(400, 1000, [], NOW); b = time_to_failure(400, 1000, [], NOW, volume_multiplier=2)
    assert math.isclose(b['days'], a['days'] / 2) and b['forecast'][0][1] == 400 and b['forecast'][-1][1] <= 400

# ---- engine / gate --------------------------------------------------------
def test_engine_state_and_gate_ledger(tmp_path):
    led = Ledger(str(tmp_path / 'l.db')); v = VM['V-004']
    st = vendor_state(v, EVID['V-004'], led, trials=1000); assert st['state'] == 'AUTO' and st['balance'] == pytest.approx(st['allowance'])
    r = evaluate_document(st, 'customer_statement.pdf', POL['sample_documents']['customer_statement.pdf'], 'statement_summary')
    assert r['decision'] == 'AUTO' and r['spends']; commit(led, st, r, 'customer_statement.pdf'); assert led.verify()
    rec = led.records(v['vendor_name'])[0]; assert rec['meta']['decision'] == 'AUTO' and rec['meta']['receipt_hash'] == r['receipt']['receipt_hash']
    st2 = vendor_state(v, EVID['V-004'], led, trials=1000); assert st2['balance'] < st['balance'] and st2['debt'] > 0
def test_blocked_and_unapproved_review_do_not_spend(tmp_path):
    led = Ledger(str(tmp_path / 'l.db')); st = vendor_state(VM['V-004'], EVID['V-004'], led, trials=1000)
    r = evaluate_document(st, 'tax_form_1099.pdf', POL['sample_documents']['tax_form_1099.pdf'], 'tax_document'); assert r['decision'] == 'BLOCK' and not r['spends']
    commit(led, st, r, 'x'); assert privacy_debt(led.records()) == 0 and led.records()[0]['meta']['decision'] == 'BLOCK'
def test_policy_floor_forces_review():
    st = vendor_state(VM['V-004'], EVID['V-004'], records=[], trials=1000, policy={'require_review': True})
    r = evaluate_document(st, 'j', POL['sample_documents']['job_schedule_export.csv'], 'batch_scheduling', {'require_review': True}); assert r['decision'] == 'REVIEW'
    assert 'person must review' in ' '.join(r['why']['reasons'])
def test_dna_after_redaction_not_higher():
    st = vendor_state(VM['V-003'], EVID['V-003'], records=[], trials=1000)
    r = evaluate_document(st, 'c', POL['sample_documents']['customer_statement.pdf'], 'statement_summary')
    assert all(r['dna_after'][k] <= r['dna_before'][k] + 1e-9 for k in r['dna_before'])

# ---- what-if / fire drill never mutate production --------------------------
def _snapshot(led):
    return (led.rows(), [(p.name, p.read_bytes()) for p in (ROOT / 'ledger').glob('*.json')], (DATA / 'evidence.json').read_bytes())
def test_simulation_does_not_mutate_anything(tmp_path):
    led = Ledger(str(tmp_path / 'l.db')); led.append('X', 'd', 5, 10, 5)
    rows = copy.deepcopy(EVID['V-003']); doc = copy.deepcopy(POL['sample_documents']['customer_statement.pdf']); records = led.records()
    before = (copy.deepcopy(rows), copy.deepcopy(doc), copy.deepcopy(records), _snapshot(led))
    for s in [{'kind': k} for k in KINDS]: run_scenario(VM['V-003'], rows, s, doc, 'statement_mailing', records, trials=500)
    fire_drill(VM['V-003'], rows, doc, 'statement_mailing', records, trials=500)
    assert before == (rows, doc, records, _snapshot(led)) and led.verify()
def test_what_if_directions():
    rows, doc = EVID['V-003'], POL['sample_documents']['customer_statement.pdf']
    llm = run_scenario(VM['V-003'], rows, {'kind': 'add_llm'}, doc, 'statement_mailing', trials=2000)
    assert llm['simulated']['p_ai'] > llm['current']['p_ai'] and llm['budget_change_pct'] < 0 and llm['risk_increase'] > 0
    vol = run_scenario(VM['V-003'], rows, {'kind': 'volume', 'multiplier': 2}, doc, 'statement_mailing', trials=500)
    assert vol['risk_increase'] == pytest.approx(1.0, abs=1e-6)
    miss = run_scenario(VM['V-003'], rows, {'kind': 'detector_miss', 'entity_type': 'BANK'}, doc, 'statement_mailing', trials=500)
    assert miss['simulated']['doc_cost'] > miss['current']['doc_cost']
    rg = run_scenario(VM['V-003'], rows, {'kind': 'remove_gov_id'}, doc, 'statement_mailing', trials=500); assert rg['simulated']['doc_cost'] <= rg['current']['doc_cost']
    po = run_scenario(VM['V-003'], rows, {'kind': 'raise_p_ai', 'value': 0.58}, doc, 'statement_mailing', trials=500); assert po['simulated']['p_ai'] == 0.58
def test_stale_evidence_flags_false_assurance():
    r = run_scenario(VM['V-005'], EVID['V-005'], {'kind': 'stale'}, POL['sample_documents']['rtp_payment_file.xml'], 'payment_routing', trials=1500)
    assert r['false_assurance'] and r['simulated']['p_ai'] < r['current']['p_ai']
def test_fire_drill_covers_six_failures():
    out = fire_drill(VM['V-001'], EVID['V-001'], POL['sample_documents']['job_schedule_export.csv'], 'batch_scheduling', trials=500)
    assert len(out) == 6 and all(o['mitigation'] for o in out)

# ---- self-healing policy --------------------------------------------------
def test_proposal_only_tightens_and_is_not_applied(tmp_path):
    b = vendor_state(VM['V-003'], EVID['V-003'], records=[], trials=2000)
    rows = copy.deepcopy(EVID['V-003']); from src.simulate import _row
    rows.append(_row(VM['V-003']['vendor_name'], 'LLM', 'ai_provider', 'Technical proof (docs/SDK/job posts)', 'x'))
    a = vendor_state(VM['V-003'], rows, records=[], trials=2000); p = propose_policy(b, a)
    assert p['direction'] == 'tighten' and p['require_review'] and {'BANK', 'GOV_ID'} <= set(p['extra_redact']) and p['status'] == 'PROPOSED' and p['budget_change_pct'] < 0
    assert store.approved_policy('V-003', tmp_path / 'none.json') is None
    assert propose_policy(a, b)['direction'] == 'relax' and propose_policy(a, b)['actions'] == []
def test_approval_roundtrip(tmp_path):
    path = tmp_path / 'p.json'; store.approve_policy('V-003', {'require_review': True, 'extra_redact': ['BANK']}, 'me', path)
    assert store.approved_policy('V-003', path)['approved_by'] == 'me'; store.revoke_policy('V-003', path); assert store.approved_policy('V-003', path) is None

# ---- the three demos -------------------------------------------------------
def test_demo1_safe():
    d = demo_safe(); assert d['result']['decision'] == 'AUTO' and d['result']['remaining_after'] > 0.9 and d['ledger_ok']
def test_demo2_escalation():
    d = demo_escalation()
    assert d['after']['p_ai'] > d['before']['p_ai'] and d['after']['allowance'] < d['before']['allowance']
    assert d['result_before']['decision'] == 'AUTO' and d['result_after']['decision'] == 'REVIEW'
    act = lambda r: {x['field']: x['action'] for x in r['plan']['rows']}['bank_account']
    assert act(d['result_before']) == 'KEEP' and act(d['result_after']) in ('MASK', 'REDACT')
def test_demo3_hard_rule():
    d = demo_hard_rule(); r = d['result']
    assert r['decision'] == 'BLOCK' and r['remaining_after'] >= 0.95 and 'government ID' in r['why']['policy']
def test_demos_leave_production_untouched():
    before = [(p.name, p.read_bytes()) for p in (ROOT / 'ledger').glob('*.json')]
    demo_safe(); demo_escalation(); demo_hard_rule()
    assert before == [(p.name, p.read_bytes()) for p in (ROOT / 'ledger').glob('*.json')]
