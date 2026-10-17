"""Meridian - vendor AI search and privacy control.

Search public pages -> grade claims -> Digital Twin -> send / block a document.
Run:  streamlit run app.py
"""
import math, os, shutil, sys, tempfile
from pathlib import Path
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from src.config import BASE, WEIGHTS, AI_USE_FLOW_DISCOUNT
from src.demo import load
from src.redaction import load_policy, minimum_necessary
from src.engine import vendor_state, evaluate_document, commit, describe_policy
from src.receipt import render_text
from src.simulate import run_scenario, KINDS
from src.demohistory import open_ledger, seed_demo_history, DEMO_DB
from src.ledger import Ledger
from src.twin import BASE_PROB, FLOW_PROVEN
from src.osint import investigate, to_evidence_rows, vendor_domain
from src import store, viz
from src.ui import theme as T
from src.views import sanitise_lab, live_osint, policy_lab

st.set_page_config(page_title='Meridian', page_icon='◐', layout='wide', initial_sidebar_state='expanded')
T.inject()

TC = viz.TIER_COLOR
STATE_COLOR = {k: T.TONE[k] for k in ('AUTO', 'STRICT', 'REVIEW', 'BLOCK')}
ACTION_ICON = {'KEEP': 'ALLOWED', 'MASK': 'MASKED', 'REDACT': 'REDACTED'}
DEFAULT_TASK = {'customer_statement.pdf': 'statement_summary', 'rtp_payment_file.xml': 'payment_routing',
                'tax_form_1099.pdf': 'tax_document', 'job_schedule_export.csv': 'batch_scheduling'}
PAGES = ['Overview', 'Digital Twin', 'Document Gate', 'Sanitisation Lab', 'Live OSINT Explorer', 'Policy Lab', 'What-If', 'Ledger', 'Search']

POL = load_policy()
vendors, evidence = load(include_overlay=True)
VM = {v['vendor_id']: v for v in vendors}

with st.sidebar:
    T.brand()
    page = st.radio('Go to', PAGES, key='nav', label_visibility='collapsed')
    with st.expander('Settings'):
        base = st.number_input('Base budget', 100, 10000, int(BASE), 100)
        trials = st.number_input('Simulation runs', 1000, 50000, 5000, 1000)
        disc = st.slider('Discount AI-use-only evidence', 0.1, 1.0, float(AI_USE_FLOW_DISCOUNT), 0.05,
                         help='1.0 keeps the original model. Lower it when a page only shows that AI exists, not that your data reaches it.')
        docs_day = st.number_input('Documents per day (forecast only)', 1, 1000, 10)
        mode_label = st.radio('Ledger', ['Live', 'Demo data'])
        mode = 'demo' if mode_label.startswith('Demo') else 'live'
        if mode == 'demo' and st.button('Rebuild demo data'):
            seed_demo_history(); st.rerun()
    st.caption('Chance of AI (P_ai) is an estimate from public pages, not a measured fact.')

LEDGER = open_ledger(mode)
APPROVED = store.load_policies()
if mode == 'demo':
    st.warning('This is demo data, not real Meridian activity.')


def get_state(v, rows=None, **kw):
    return vendor_state(v, rows if rows is not None else evidence.get(v['vendor_id'], []), LEDGER, base, int(trials), disc,
                        policy=APPROVED.get(v['vendor_id']), docs_per_day=docs_day, **kw)


def pill(tier, text=None):
    return T.badge(text or tier, tier)


def fmt_days(t):
    if math.isinf(t['days']): return '> 90 days'
    return f"{t['days']:.1f} days" + (' (forecast)' if t['is_projection'] else '')


def vendor_pick(key, label='Vendor'):
    return st.selectbox(label, list(VM), key=key, format_func=lambda x: f"{x} — {VM[x]['vendor_name']}")


def doc_picker(key):
    names = list(POL['sample_documents']) + ['Custom document']
    name = st.selectbox('Document', names, key=f'{key}_doc')
    tasks = list(POL['tasks'])
    if name != 'Custom document':
        task = st.selectbox('Task', tasks, index=tasks.index(DEFAULT_TASK[name]), key=f'{key}_task', format_func=lambda t: POL['tasks'][t]['label'])
        return name, POL['sample_documents'][name], task
    ed = st.data_editor(pd.DataFrame([{'type': 'PERSON', 'confidence': 0.99}, {'type': 'BANK', 'confidence': 0.7}]), num_rows='dynamic', key=f'{key}_ents',
                        column_config={'type': st.column_config.SelectboxColumn(options=list(WEIGHTS)), 'confidence': st.column_config.NumberColumn(min_value=0.0, max_value=1.0)})
    ents = [{'type': r['type'], 'confidence': float(r['confidence'])} for r in ed.to_dict('records') if r.get('type')]
    extra = st.multiselect('Fields in the document', [f for f, d in POL['fields'].items() if not d['entity']], ['txn_date', 'txn_amount'], key=f'{key}_fields')
    task = st.selectbox('Task', tasks, key=f'{key}_task', format_func=lambda t: POL['tasks'][t]['label'])
    return 'custom_document', {'fields_present': extra, 'entities': ents}, task


def banner(decision, lines, sub=''):
    T.verdict(decision, sub, decision)
    for x in lines: T.callout(x, decision)


def page_search():
    T.page_header('Site Search', 'Quick check of the vendor’s own public site: read the pages, then grade the claims. Other domains are ignored. For responsible, reviewable collection use the Live OSINT Explorer.', 'Legacy single-site search')
    vid = vendor_pick('osint_vendor')
    v = VM[vid]
    st.caption(f"Only pages on {vendor_domain(v)}. Chat widgets, job boards, and other sites are skipped.")
    if st.button('Search public pages', type='primary'):
        with st.spinner(f'Searching {vendor_domain(v)}…'):
            st.session_state['osint_result'] = {'vid': vid, **investigate(v)}
    hit = st.session_state.get('osint_result')
    if not hit or hit.get('vid') != vid:
        st.info('Pick a vendor and search. We look for trust, privacy, sub-processor, DPA, docs, and careers pages, then grade what we find.')
        return
    g, pages, ledger = hit['grade'], hit['pages'], hit['ledger']
    k = st.columns(4)
    k[0].metric('Grade', g['verdict'])
    k[1].metric('Providers named', ', '.join(g['providers']) or 'None')
    k[2].metric('Pages read', f"{g['pages_ok']} / {g['pages_tried']}")
    k[3].metric('Page types covered', f"{g['coverage']:.0%}")
    st.write(g['why'])
    if g.get('contradictions'):
        st.error('Contradiction: the same provider is both named and denied. Read the quotes.')
    st.subheader('Pages we opened')
    st.dataframe(pd.DataFrame([{'Page type': p['source_type'], 'URL': p['url'], 'Read': 'Yes' if p['ok'] else 'No'} for p in pages]),
                 hide_index=True, width='stretch')
    st.subheader('Claim list')
    st.caption('Each row is a quote. Reliability and how specific the claim is are shown separately. We do not multiply them.')
    if ledger:
        st.dataframe(pd.DataFrame([{
            'Quote': r['claim'],
            'Providers': ', '.join(r['providers']) or '—',
            'How reliable': r['reliability_label'],
            'How specific': r['specificity_label'],
            'Tag': r['tag'],
            'Sources that agree': r['corroboration'],
            'What it shows': 'Your data may reach it' if r['scope'] == 'customer_data' else 'AI exists (not the data path)',
            'URL': r['url'],
            'When': r['retrieved_at'],
        } for r in ledger]), hide_index=True, width='stretch')
    else:
        st.write('No AI claims on the pages we could read.')
    st.subheader('Limits')
    for line in g['limits']:
        st.markdown(f'- {line}')
    if hit['signals'] and st.button('Save as unverified evidence'):
        for row in to_evidence_rows(v, hit['signals']):
            store.add_evidence(vid, row)
        st.success('Saved. A person must still check each quote. Refresh Overview to see it in the twin.')


def page_overview():
    T.page_header('Fleet overview', 'How exposed is each vendor right now? Chance of AI, remaining leak budget, and whether a document would be sent.', 'Command center')
    S = [get_state(v) for v in vendors]
    cnt = {t: sum(s['overall_tier'] == t for s in S) for t in viz.TIER_COLOR}
    T.kpi_row([(t.title(), cnt[t], 'vendors at this overall tier', t) for t in cnt])
    order = st.segmented_control('Sort', ['Highest risk', 'Least budget left', 'Name'], default='Highest risk', key='ov_sort') or 'Highest risk'
    rank = {'CRITICAL': 0, 'HIGH': 1, 'MODERATE': 2, 'LOW': 3}
    S = sorted(S, key=(lambda s: (rank[s['overall_tier']], -s['p_ai'])) if order == 'Highest risk' else (lambda s: s['remaining_frac']) if order == 'Least budget left' else (lambda s: s['name']))
    grid = st.columns(3)
    for i, s in enumerate(S):
        v = VM[s['vendor_id']]; rem = max(0, min(1, s['remaining_frac'])); b = s['blast']
        nm = s['name'] if len(s['name']) <= 26 else s['name'][:25].rstrip(' ,') + '…'
        inner = (f"<div style='display:flex;justify-content:space-between;align-items:center;gap:8px'><h4>{T.esc(nm)}</h4>{pill(s['overall_tier'])}</div>"
                 f"<div class='m-sub2'>{T.esc(s['vendor_id'])} · {T.esc(v['category'])} · {T.esc(v['criticality'])} criticality</div>"
                 + T.kv([('Chance of AI (estimate)', T.raw(f"{s['p_ai']:.0%} &nbsp;{T.badge(s['p_ai_tier'], s['p_ai_tier'])}")),
                         ('Budget left', f"{s['balance']:,.0f} / {s['allowance']:,.0f}")])
                 + T.meter(rem, s['state'], 'Budget remaining')
                 + T.kv([('Past exposure', f"{s['debt']:,.0f} · {s['debt_tier']}"), ('Who could see this data', f"{b['organizations']} orgs · {b['ai_systems']} AI"),
                         ('Budget runs out', fmt_days(s['ttf'])), ('Send decision', T.raw(T.badge(s['state'], s['state'])))]))
        grid[i % 3].markdown(T.card(inner, s['overall_tier']), unsafe_allow_html=True)
    st.caption('Overall rating is the worst of AI chance, past exposure, and budget. These are policy cut-offs, not facts.')
    with st.expander('Table'):
        st.dataframe(pd.DataFrame([{'Vendor': s['name'], 'Overall': s['overall_tier'], 'P_ai': f"{s['p_ai']:.1%}",
                                    'Budget': round(s['allowance']), 'Left': round(s['balance']),
                                    'Past exposure': round(s['debt']), 'Orgs': s['blast']['organizations'],
                                    'Runs out': fmt_days(s['ttf']), 'State': s['state']} for s in S]),
                     width='stretch', hide_index=True)


def page_twin():
    T.page_header('Digital Twin', 'Where Meridian data could travel: vendor, sub-processors and AI systems, with the evidence behind every link.', 'Exposure graph')
    s = get_state(VM[vendor_pick('tw')])
    T.kpi_row([('Chance of AI (estimate)', f"{s['p_ai']:.1%}", s['p_ai_tier'], s['p_ai_tier']), ('Most likely path', f"{s['path_prob']:.1%}", 'to an AI system', 'accent'),
               ('Leak budget', f"{s['allowance']:,.0f}", 'allowance this window', 'accent'), ('Orgs that could see this', s['blast']['organizations'], 'incl. AI providers', 'info')])
    main, rail = st.columns([2.4, 1], gap='large')
    b = s['blast']
    with main:
        st.plotly_chart(viz.twin_figure(s['graph'], s['path']), width='stretch')
        st.caption('Numbers on arrows are estimates. Amber is the most likely path to AI. Dotted means we do not know the destination.')
        st.write('**Most likely path:**', ' → '.join(s['path']) if s['path'] else 'No AI path in current evidence')
    with rail:
        T.render(T.card('<h4>Blast radius</h4><div class="m-sub2">Who could receive this data</div>' + T.kv([
            ('Organisations', b['organizations']), ('Direct vendors', b['direct_vendors']), ('Sub-processors', b['subprocessors']), ('AI systems', b['ai_systems']),
            ('External AI providers', b['external_ai_providers']), ('Unknown destinations', b['unknown_destinations'])]), lift=False))
        st.plotly_chart(viz.blast_fig(b), width='stretch')
    st.subheader('Evidence')
    T.callout('“They use AI” is not the same as “our data reaches that AI.” Unverified rows still need a person to check the quote.', 'warn')
    rows = evidence.get(s['vendor_id'], [])
    st.dataframe(pd.DataFrame([{'Target': r['target'], 'Kind': r.get('target_kind'), 'Strength': r['tag'],
                                'Claim': 'Data path shown' if r.get('claim_type') == FLOW_PROVEN else 'AI use only',
                                'Source': r['source'], 'URL': r['url'], 'Date': r['date'],
                                'Quote': r['excerpt'], 'Checked': r.get('verified', False)} for r in rows]),
                 width='stretch', hide_index=True)


def page_gate():
    T.page_header('Document Gate', 'Decide what a document may carry to a vendor: minimum necessary fields, destination-aware masking, then the leak-budget brake.', 'Send / block')
    v = VM[vendor_pick('gt')]; s = get_state(v); pol = APPROVED.get(v['vendor_id'])
    name, doc, task = doc_picker('gate')
    T.render('<div class="m-card">' + T.kv([('Send to', s['name']), ('Chance of AI (estimate)', f"{s['p_ai']:.1%}"), ('Budget', f"{s['balance']:,.0f} / {s['allowance']:,.0f}")]) + '</div>')
    if pol:
        st.info('A stricter rule is already in force: ' + ' · '.join(describe_policy(pol)))
    ev = evaluate_document(s, name, doc, task, pol, 0.0, 0.0, 0); plan = ev['plan']
    mn = minimum_necessary(task); lab = mn['labels']
    st.subheader('What this task needs')
    m = st.columns(3)
    m[0].markdown('**Need**\n\n' + ('\n\n'.join(f'- {lab[f]}' for f in mn['required']) or '_none_'))
    m[1].markdown('**Nice to have**\n\n' + ('\n\n'.join(f'- {lab[f]}' for f in mn['optional']) or '_none_'))
    m[2].markdown('**Drop**\n\n' + '\n\n'.join(f'- {lab[f]}' for f in mn['not_required'][:7]))
    st.subheader('What we would send')
    k = st.columns(4)
    k[0].metric('Fields', len(plan['rows']))
    k[1].metric('Clear', sum(r['action'] == 'KEEP' for r in plan['rows']))
    k[2].metric('Masked', sum(r['action'] == 'MASK' for r in plan['rows']))
    k[3].metric('Removed', sum(r['action'] == 'REDACT' for r in plan['rows']))
    st.dataframe(pd.DataFrame([{'Field': r['label'], 'Action': ACTION_ICON[r['action']], 'Why': r['reason']} for r in plan['rows']]),
                 hide_index=True, width='stretch')
    cc = st.columns([1.1, 1])
    with cc[0]:
        st.subheader('Decision')
        banner(ev['decision'], ev['why']['reasons'], ev['why']['policy'])
        st.write(f"Cost **{ev['cost']:.1f}** · budget **{s['balance']:,.0f} → {ev['balance_after']:,.0f}**")
        if ev['decision'] == 'BLOCK':
            st.error('Blocked. Nothing is spent. The attempt is logged.')
    with cc[1]:
        st.plotly_chart(viz.dna_radar(ev['dna_before'], ev['dna_after'], 'Before vs after we strip fields'), width='stretch')
    if st.button('Record this decision', type='primary'):
        commit(LEDGER, s, ev, name, synthetic=(mode == 'demo')); st.success('Saved to the ledger.'); st.rerun()
    if ev['decision'] == 'REVIEW' and st.button('I reviewed this. Record it.'):
        ev2 = evaluate_document(s, name, doc, task, pol, 0.0, 0.0, 0, human_approved=True)
        commit(LEDGER, s, ev2, name, synthetic=(mode == 'demo')); st.success('Review recorded.'); st.rerun()
    txt = render_text(ev['receipt'])
    with st.expander('Receipt'):
        st.code(txt, language=None)
        st.download_button('Download receipt', txt, f'receipt_{name}.txt')


def scenario_form(key):
    kind = st.selectbox('What if', list(KINDS), key=f'{key}_k', format_func=KINDS.get); scn = {'kind': kind}
    tags = list(BASE_PROB)
    if kind in ('add_llm', 'add_subprocessor', 'ai_reachable'):
        scn['tag'] = st.selectbox('How strong is the new link', tags, 2, key=f'{key}_t')
    if kind == 'add_subprocessor':
        scn['with_ai'] = st.checkbox('That sub-processor runs AI', True, key=f'{key}_w')
    if kind == 'raise_p_ai':
        scn['value'] = st.slider('New chance of AI', 0.0, 1.0, 0.6, 0.01, key=f'{key}_p')
    if kind == 'detector_miss':
        scn['entity_type'] = st.selectbox('Missed field type', list(WEIGHTS), 1, key=f'{key}_e'); scn['count'] = st.number_input('How many', 1, 20, 1, key=f'{key}_c')
    if kind == 'volume':
        scn['multiplier'] = st.slider('Volume multiplier', 0.25, 10.0, 2.0, 0.25, key=f'{key}_m')
    if kind == 'ocr_drop':
        scn['penalty'] = st.slider('Read error penalty', 0.0, 20.0, 5.0, key=f'{key}_o'); scn['regions'] = st.number_input('Unreadable regions', 0, 10, 2, key=f'{key}_r')
    if kind == 'stale':
        scn['extra_days'] = st.slider('Evidence ages by (days)', 30, 1000, 400, key=f'{key}_a')
    return scn


def page_whatif():
    T.page_header('What-If', 'Change one assumption and see how the chance of AI, the budget and the decision move.', 'Simulation')
    T.callout('This is a copy. The live ledger and evidence are not changed.', 'ok')
    kw = dict(base=base, trials=int(trials), flow_discount=disc, docs_per_day=docs_day)
    c = st.columns(3)
    with c[0]: v = VM[vendor_pick('wi')]
    with c[1]: name, doc, task = doc_picker('wi')
    with c[2]: scn = scenario_form('wi')
    r = run_scenario(v, evidence.get(v['vendor_id'], []), scn, doc, task, LEDGER.records(v['vendor_name']), policy=APPROVED.get(v['vendor_id']), **kw)
    a, b = r['current'], r['simulated']
    h = st.columns(2)
    for col, ttl, x in ((h[0], 'Now', a), (h[1], 'If - ' + r['label'], b)):
        col.markdown(T.card(f"<h4>{T.esc(ttl)}</h4>" + T.kv([('Chance of AI', f"{x['p_ai']:.0%}"), ('Budget', f"{x['allowance']:,.0f}"), ('Cost', f"{x['doc_cost']:.1f}"), ('Decision', T.raw(T.badge(x['decision'], x['decision'])))]), 'accent' if x is a else 'info', lift=False), unsafe_allow_html=True)
    k = st.columns(4)
    k[0].metric('Chance of AI', f"{b['p_ai']:.0%}", f"{(b['p_ai'] - a['p_ai']) * 100:+.1f} pts", delta_color='inverse')
    k[1].metric('Budget', f"{b['allowance']:,.0f}", f"{r['budget_change_pct']:+.0%}")
    ri = r['risk_increase']; k[2].metric('Risk change', 'inf' if math.isinf(ri) else f'{ri:+.0%}')
    k[3].metric('Decision', b['decision'], 'changed' if r['decision_changed'] else 'same', delta_color='off')
    st.markdown(f"**What to do:** {r['mitigation']}")
    if r['false_assurance']:
        T.callout('Old evidence can make the model look safer without any real change. Check the pages again.', 'warn')
    for x in b['reasons']:
        st.markdown(f'- {x}')


def page_ledger():
    T.page_header('Ledger', 'Every decision is chained to the one before it. Editing history breaks the chain.', 'Audit trail')
    ok = LEDGER.verify()
    T.callout('Hash chain is intact.' if ok else 'Hash chain is broken.', 'ok' if ok else 'crit')
    recs = LEDGER.records()
    st.caption(f'{len(recs)} entries · {mode_label}')
    if recs:
        df = pd.DataFrame([{'id': r['id'], 'time': pd.to_datetime(r['timestamp'], unit='s'), 'destination': r['destination'],
                            'document': r['doc_id'], 'decision': r['meta'].get('decision', ''),
                            'cost': round(r['cost'], 2), 'after': round(r['balance_after'], 1),
                            'demo': r['meta'].get('synthetic', False), 'hash': r['row_hash'][:12]} for r in reversed(recs)])
        st.dataframe(df.head(500), hide_index=True, width='stretch')
    st.caption('The tamper check edits a throwaway copy. The real ledger is not touched.')
    if st.button('Test tamper detection') and recs:
        with tempfile.TemporaryDirectory() as d:
            src = DEMO_DB if mode == 'demo' else str(Path(LEDGER.conn.execute('PRAGMA database_list').fetchone()[2]))
            cp = os.path.join(d, 'copy.db'); shutil.copy(src, cp); t = Ledger(cp)
            t.conn.execute('UPDATE ledger SET cost = cost * 0.5 WHERE id = (SELECT MIN(id) FROM ledger WHERE cost > 0)'); t.conn.commit()
            st.write('Copy after edit:', 'intact' if t.verify() else 'broken — change caught')
            t.conn.close()


{
    'Overview': page_overview,
    'Digital Twin': page_twin,
    'Document Gate': page_gate,
    'Sanitisation Lab': lambda: sanitise_lab.render(VM, get_state, POL, evaluate_document),
    'Live OSINT Explorer': lambda: live_osint.render(VM, evidence, LEDGER, base, int(trials), disc),
    'Policy Lab': lambda: policy_lab.render(VM, get_state, POL, evaluate_document),
    'What-If': page_whatif,
    'Ledger': page_ledger,
    'Search': page_search,
}[page]()
st.divider()
st.caption('These numbers are estimates from public evidence. They are not proof.')
