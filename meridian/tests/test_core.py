import math,sqlite3,copy,pytest
import networkx as nx
from src.twin import (noisy_or,decayed_probability,BASE_PROB,build_vendor_graph,simulate,blast_radius,SOURCE,FLOW_PROVEN)
from src.budget import (allowance,doc_cost,decision,destination_multiplier,history_factor,allowance_breakdown,adaptive_allowance)
from src.demo import load,vendor_assessment
from src.ledger import Ledger
from src.dna import data_dna
from src.redaction import load_policy,plan_redaction,minimum_necessary
from src.receipt import make_receipt,render_text
from src.explain import decision_reasons

VENDORS,EVID=load(); VM={v['vendor_id']:v for v in VENDORS}

# ---- digital twin ---------------------------------------------------------
def test_evidence_mapping(): assert [BASE_PROB[k] for k in BASE_PROB]==[0.10,0.40,0.70,0.95]
def test_noisy_or():
    assert noisy_or([])==0 and math.isclose(noisy_or([0.5,0.5]),0.75) and math.isclose(noisy_or([0.1,0.4]),1-0.9*0.6)
def test_time_decay():
    assert math.isclose(decayed_probability('Confirmed',0),0.95)
    assert math.isclose(decayed_probability('Confirmed',365),0.95*math.exp(-1))
    assert decayed_probability('Confirmed',10)>decayed_probability('Confirmed',400)
def test_monte_carlo_matches_analytic():
    G=nx.DiGraph(); G.add_edge('s','a',p=0.6); G.add_edge('a','t',p=0.5); G.add_edge('s','t',p=0.2)
    exact=1-(1-0.3)*(1-0.2)   # P(path s-a-t) = .3, direct = .2, independent
    assert abs(simulate(G,'s',['t'],40000)-exact)<0.01
def test_baseline_p_ai_unchanged():
    # numbers produced by the original prototype (5000 trials, seed 42)
    a=vendor_assessment(VM['V-004'],EVID['V-004'],1000,5000); assert abs(a['p_ai']-0.091)<0.005 and abs(a['allowance']-909.0)<8
    b=vendor_assessment(VM['V-005'],EVID['V-005'],1000,5000); assert b['p_ai']>0.95
def test_unknown_nodes_do_not_change_p_ai():
    a=vendor_assessment(VM['V-001'],EVID['V-001'],1000,5000); assert a['blast']['unknown_destinations']==1
    Gc=a['core_graph']; assert not any(d['kind']=='unknown' for _,d in Gc.nodes(data=True))
def test_flow_discount_only_hits_ai_use_only_edges():
    rows=copy.deepcopy(EVID['V-005']); full=vendor_assessment(VM['V-005'],rows,1000,3000)['p_ai']
    disc=vendor_assessment(VM['V-005'],rows,1000,3000,flow_discount=0.5)['p_ai']; assert disc<full
    for r in rows: r['claim_type']=FLOW_PROVEN
    assert abs(vendor_assessment(VM['V-005'],rows,1000,3000,flow_discount=0.5)['p_ai']-full)<0.02
def test_evidence_records_complete():
    for rows in EVID.values():
        for e in rows:
            for k in ('source','url','date','evidence_type','excerpt','confidence'): assert e.get(k) not in (None,''),k

# ---- budget ---------------------------------------------------------------
def test_allowance_formula(): assert math.isclose(allowance(1000,0.4,0.75),450)
def test_adaptive_equals_original_without_history():
    assert math.isclose(adaptive_allowance(1000,0.3,0.75,0.0),allowance(1000,0.3,0.75))
def test_history_lowers_allowance_with_floor():
    assert history_factor(0)==1 and history_factor(1e9)==pytest.approx(0.6) and adaptive_allowance(1000,0,1,1000)<1000
def test_breakdown_is_transparent():
    b=allowance_breakdown(1000,0.42,'High',0.75,500); assert math.isclose(b['allowance'],1000*0.58*0.75*b['history_factor']); assert len(b['lines'])==4
def test_doc_cost_and_multiplier():
    ents=[{'type':'EMAIL','confidence':0.5},{'type':'GOV_ID','confidence':0.9}]
    assert math.isclose(doc_cost(ents,0.2),(2*0.5+10*0.1)*1.2); assert destination_multiplier(0.34)==1.34
def test_brake_logic():
    assert [decision(b,1000,10) for b in (800,300,100,40)]==['AUTO','STRICT','REVIEW','BLOCK']
def test_gov_id_hard_block_even_with_95pct_budget():
    assert decision(950,1000,0,True)=='BLOCK'
    w=decision_reasons('BLOCK',950,1000,0,True,0.1); assert 'government ID' in w['policy'] and 'government ID' in w['reasons'][0]

# ---- ledger ---------------------------------------------------------------
def test_ledger_chain_and_tamper(tmp_path):
    db=str(tmp_path/'l.db'); l=Ledger(db); l.append('A','d1',1,10,9); l.append('A','d2',2,9,7,meta={'decision':'AUTO'}); assert l.verify()
    l.conn.execute("UPDATE ledger SET cost=0 WHERE id=1"); l.conn.commit(); assert not l.verify()
def test_ledger_meta_tamper(tmp_path):
    l=Ledger(str(tmp_path/'l.db')); l.append('A','d',1,10,9,meta={'decision':'BLOCK'}); assert l.verify()
    l.conn.execute("UPDATE ledger SET meta='{\"decision\":\"AUTO\"}'"); l.conn.commit(); assert not l.verify()

# ---- DNA / redaction ------------------------------------------------------
POL=load_policy(); STMT=POL['sample_documents']['customer_statement.pdf']
def test_dna_axes_and_monotonic():
    d=data_dna(STMT['entities']); assert set(d)=={'Identity','Financial','Contact','Operational','Regulatory','AI sensitivity'} and all(0<=v<=1 for v in d.values())
    assert data_dna([])['Identity']==0 and data_dna([{'type':'GOV_ID'}])['Identity']==1.0
    assert data_dna([{'type':'EMAIL'}])['Contact']<data_dna([{'type':'EMAIL'},{'type':'EMAIL'}])['Contact']
def test_minimum_necessary():
    m=minimum_necessary('statement_summary'); assert set(m['required'])=={'txn_date','txn_amount','txn_type'}
    assert {'gov_id','full_name','bank_account','address'}<=set(m['not_required'])
def test_destination_aware_redaction_differs_by_risk():
    doc=POL['sample_documents']['rtp_payment_file.xml']
    lo=plan_redaction(doc,'payment_routing',0.05); hi=plan_redaction(doc,'payment_routing',0.80)
    act=lambda p,f:[r['action'] for r in p['rows'] if r['field']==f][0]
    assert act(lo,'bank_account')=='KEEP' and act(hi,'bank_account')=='MASK' and hi['cost']<lo['cost']*10 and lo['tier']=='LOW' and hi['tier']=='CRITICAL'
    assert hi['masked']['BANK']==2 and lo['kept']['BANK']==2
def test_not_required_always_redacted_and_gov_id_never_clear():
    for p in (0.0,0.5,0.99):
        pl=plan_redaction(STMT,'statement_summary',p); assert all(r['action']=='REDACT' for r in pl['rows'] if r['entity'])
        tx=plan_redaction(POL['sample_documents']['tax_form_1099.pdf'],'tax_document',p); assert tx['unresolved_gov_id']
def test_low_detector_confidence_on_gov_id_is_unresolved():
    d={'fields_present':['gov_id'],'entities':[{'type':'GOV_ID','confidence':0.80}]}; assert plan_redaction(d,'statement_summary',0.1)['unresolved_gov_id']
def test_redaction_uses_original_doc_cost_for_redacted_entities():
    pl=plan_redaction(STMT,'statement_summary',0.2); assert math.isclose(pl['cost'],doc_cost(STMT['entities'],0.2))

# ---- blast radius ---------------------------------------------------------
def test_blast_radius_counts():
    a=vendor_assessment(VM['V-005'],EVID['V-005'],1000,2000); b=a['blast']
    assert b['direct_vendors']==1 and b['external_ai_providers']==1 and b['ai_systems']==2 and b['max_depth']==2
def test_blast_radius_grows_with_subprocessor():
    rows=copy.deepcopy(EVID['V-004']); base=vendor_assessment(VM['V-004'],rows,1000,1000)['blast']
    rows.append({'parent':VM['V-004']['vendor_name'],'target':'Hypothetical Hosting Co','tag':'Corroborated (2+ independent sources)','age_days':1,'target_kind':'service'})
    rows.append({'parent':'Hypothetical Hosting Co','target':'Hypothetical LLM API','tag':'Technical proof (docs/SDK/job posts)','age_days':1,'target_kind':'ai'})
    b=vendor_assessment(VM['V-004'],rows,1000,1000)['blast']
    assert b['organizations']>base['organizations'] and b['max_depth']==3 and b['subprocessors']==1

# ---- receipt --------------------------------------------------------------
def test_receipt_contains_decision_and_reasons():
    pl=plan_redaction(STMT,'statement_summary',0.2); w=decision_reasons('AUTO',400,1000,pl['cost'],False,0.2)
    r=make_receipt('BNY','customer_statement.pdf','t',pl,0.2,400,400-pl['cost'],'AUTO',w); t=render_text(r)
    assert 'PRIVACY RECEIPT' in t and 'AUTO' in t and 'estimate from public pages' in t and r['receipt_hash'] in t
