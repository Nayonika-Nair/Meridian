"""Human-readable privacy receipt for every document decision."""
import hashlib,json,time

def _fmt(c): return ', '.join(f'{n} {t}' for t,n in sorted(c.items())) or 'none'

def make_receipt(destination,doc_name,task,plan,p_ai,balance_before,balance_after,decision,why,ts=None,approved_policy=None):
    ts=time.time() if ts is None else ts
    r={'destination':destination,'document':doc_name,'task':task,'timestamp':ts,
       'detected':plan['detected'],'redacted':plan['redacted'],'masked':plan['masked'],'sent_in_clear':plan['kept'],
       'allowed_fields':[x['field'] for x in plan['rows'] if x['action']=='KEEP'],
       'masked_fields':[x['field'] for x in plan['rows'] if x['action']=='MASK'],
       'redacted_fields':[x['field'] for x in plan['rows'] if x['action']=='REDACT'],
       'residual_risk':plan['residual'],'destination_risk':plan['tier'],'p_ai':p_ai,'multiplier':plan['multiplier'],
       'cost':plan['cost'],'balance_before':balance_before,'balance_after':balance_after,
       'decision':decision,'reasons':why['reasons'],'policy':why['policy'],
       'approved_policy_in_force':approved_policy or [],
        'honesty':'Chance of AI is an estimate from public pages, not a measured fact.'}
    r['receipt_hash']=hashlib.sha256(json.dumps({k:v for k,v in r.items()},sort_keys=True,default=str).encode()).hexdigest()
    return r

def render_text(r):
    L=['PRIVACY RECEIPT','='*40,f"Destination:  {r['destination']}",f"Document:     {r['document']}",f"Task:         {r['task']}",'',
       f"Detected:     {_fmt(r['detected'])}",f"Redacted:     {_fmt(r['redacted'])}",f"Masked:       {_fmt(r['masked'])}",f"Sent clear:   {_fmt(r['sent_in_clear'])}",'',
       f"Fields allowed:  {', '.join(r['allowed_fields']) or 'none'}",f"Fields masked:   {', '.join(r['masked_fields']) or 'none'}",f"Fields redacted: {', '.join(r['redacted_fields']) or 'none'}",'',
       f"Residual risk:     {r['residual_risk']:.1f} points",f"Destination risk:  {r['multiplier']:.2f}x  ({r['destination_risk']}, P_ai {r['p_ai']:.1%})",
       f"Document cost:     {r['cost']:.1f} points",f"Budget:            {r['balance_before']:.0f} -> {r['balance_after']:.0f}",'',
       f"Decision: {r['decision']}",'Reason:']+[f'  - {x}' for x in r['reasons']]+[f"Policy: {r['policy']}",'',r['honesty'],f"Receipt hash: {r['receipt_hash']}"]
    return '\n'.join(L)
