"""The three competition demos, runnable headlessly or from the UI.

They run against a TEMPORARY ledger and policy file, so they never touch production state.
Evidence added in Demo 2 is a labelled simulation input, not real OSINT.
"""
import copy, tempfile, os
from pathlib import Path
from .config import BASE
from .demo import load
from .ledger import Ledger
from .redaction import load_policy
from .engine import vendor_state, evaluate_document, commit
from .healing import propose_policy
from . import store
from .simulate import _row


def _ctx():
    vendors, ev = load()
    return {v['vendor_id']: v for v in vendors}, ev, load_policy()


def demo_safe(trials=3000):
    """Low-risk vendor + low-sensitivity document + high remaining budget -> AUTO."""
    vm, ev, pol = _ctx(); v = vm['V-001']
    with tempfile.TemporaryDirectory() as d:
        led = Ledger(os.path.join(d, 'l.db'))
        st = vendor_state(v, ev['V-001'], led, BASE, trials)
        r = evaluate_document(st, 'job_schedule_export.csv', pol['sample_documents']['job_schedule_export.csv'], 'batch_scheduling')
        commit(led, st, r, 'job_schedule_export.csv', synthetic=True)
        return {'name': 'Demo 1 - Safe', 'vendor': v['vendor_name'], 'state': st, 'result': r, 'ledger_ok': led.verify()}


def demo_escalation(trials=3000):
    """New AI subprocessor -> P_ai up -> budget down -> stricter redaction -> AUTO becomes REVIEW."""
    vm, ev, pol = _ctx(); v = vm['V-006']; doc_name = 'rtp_payment_file.xml'; doc = pol['sample_documents'][doc_name]
    task = 'payment_routing'
    with tempfile.TemporaryDirectory() as d:
        led = Ledger(os.path.join(d, 'l.db')); ppath = Path(d) / 'policies.json'
        before = vendor_state(v, ev['V-006'], led, BASE, trials)
        r0 = evaluate_document(before, doc_name, doc, task)
        new_rows = copy.deepcopy(ev['V-006']) + [
            _row(v['vendor_name'], 'New AI subprocessor (demo input)', 'ai_provider', 'Technical proof (docs/SDK/job posts)',
                 'DEMO INPUT: vendor adds an AI subprocessor. Simulated, not real evidence.')]
        after = vendor_state(v, new_rows, led, BASE, trials)
        proposal = propose_policy(before, after)
        approved = store.approve_policy(v['vendor_id'], proposal, 'demo-approver', ppath)
        after = vendor_state(v, new_rows, led, BASE, trials, policy=approved)
        r1 = evaluate_document(after, doc_name, doc, task, approved)
        return {'name': 'Demo 2 - Risk escalation', 'vendor': v['vendor_name'], 'before': before, 'after': after,
                'result_before': r0, 'result_after': r1, 'proposal': proposal, 'policy': approved}


def demo_hard_rule(trials=3000):
    """Unresolved GOV_ID -> BLOCK even with >=95% of the budget remaining."""
    vm, ev, pol = _ctx(); v = vm['V-004']
    with tempfile.TemporaryDirectory() as d:
        led = Ledger(os.path.join(d, 'l.db'))
        st = vendor_state(v, ev['V-004'], led, BASE, trials)
        r = evaluate_document(st, 'tax_form_1099.pdf', pol['sample_documents']['tax_form_1099.pdf'], 'tax_document')
        return {'name': 'Demo 3 - Hard security rule', 'vendor': v['vendor_name'], 'state': st, 'result': r}
