"""Destination-aware redaction + minimum-necessary optimiser.

Policy in plain words:
  * A field the task does not need is removed for every destination (data minimisation).
  * A needed field is sent in the clear to low-risk destinations, but is MASKED when the
    destination's P_ai tier is high and the field is sensitive.
  * Optional fields are kept only while the destination is low risk.
  * Government IDs are never sent in the clear.
The residual cost still goes through budget.doc_cost(), so the original PII scoring is intact."""
import json
from .config import (DATA, WEIGHTS, MASK_WEIGHT_MIN, OPTIONAL_DROP_WEIGHT, MASK_RESIDUAL, HARD_BLOCK_TYPES, p_ai_tier)
from .budget import doc_cost, residual_risk

def load_policy(): return json.loads((DATA/'policy.json').read_text())
ENTITY_FIELD=lambda pol:{v['entity']:k for k,v in pol['fields'].items() if v.get('entity')}

def minimum_necessary(task_key,policy=None):
    pol=policy or load_policy(); t=pol['tasks'][task_key]; allf=list(pol['fields'])
    req=t['required']; opt=t['optional']; no=[f for f in allf if f not in req and f not in opt]
    return {'task':t['label'],'required':req,'optional':opt,'not_required':no,
            'labels':{k:v['label'] for k,v in pol['fields'].items()}}

def field_action(field,task,tier,policy,extra_redact=()):
    """-> (action, status, reason). action in KEEP / MASK / REDACT."""
    f=policy['fields'][field]; ent=f.get('entity'); w=WEIGHTS.get(ent,0) if ent else 0
    status='required' if field in task['required'] else 'optional' if field in task['optional'] else 'not required'
    if ent in extra_redact: return 'REDACT',status,f'Approved self-healing policy redacts {ent}.'
    if status=='not required': return 'REDACT',status,'Not necessary for this task (data minimisation).'
    if status=='optional':
        lim=OPTIONAL_DROP_WEIGHT[tier]
        if lim is not None and w>=lim: return 'REDACT',status,f'Optional field dropped: destination risk is {tier}.'
        return 'KEEP',status,'Optional field kept: destination risk is low enough.'
    # required
    if ent=='GOV_ID': return 'MASK',status,'Required, but government IDs are never sent in the clear.'
    lim=MASK_WEIGHT_MIN[tier]
    if lim is not None and w>=lim: return 'MASK',status,f'Required, but sensitive (weight {w}) and destination risk is {tier}: masked / tokenised.'
    return 'KEEP',status,'Required for the task; destination risk is acceptable.'

def plan_redaction(document,task_key,p_ai,policy=None,extra_redact=(),undetected_risk=0.0,ocr_penalty=0.0,unknown_regions=0):
    pol=policy or load_policy(); task=pol['tasks'][task_key]; tier=p_ai_tier(p_ai); e2f=ENTITY_FIELD(pol)
    fields=list(dict.fromkeys(document['fields_present']+[e2f.get(e['type']) for e in document['entities'] if e2f.get(e['type'])]))
    rows=[]; actions={}
    for f in fields:
        a,st,why=field_action(f,task,tier,pol,extra_redact); actions[f]=a
        ent=pol['fields'][f].get('entity'); n=sum(1 for e in document['entities'] if e['type']==ent) if ent else 0
        rows.append({'field':f,'label':pol['fields'][f]['label'],'entity':ent,'count':n,'status':st,'action':a,'reason':why})
    eff=[]; unresolved=False
    for e in document['entities']:
        f=e2f.get(e['type']); a=actions.get(f,'REDACT'); conf=float(e['confidence'])
        eff_conf={'REDACT':conf,'MASK':max(0.0,1-MASK_RESIDUAL),'KEEP':0.0}[a]
        eff.append({**e,'action':a,'effective_confidence':eff_conf,'exposure':1-eff_conf})
        if e['type'] in HARD_BLOCK_TYPES and (a!='REDACT' or conf<0.95): unresolved=True
    cost=doc_cost(eff and [{'type':x['type'],'confidence':x['effective_confidence']} for x in eff],p_ai,undetected_risk,ocr_penalty,unknown_regions)
    resid=residual_risk([{'type':x['type'],'confidence':x['effective_confidence']} for x in eff],undetected_risk,ocr_penalty,unknown_regions)
    naive=doc_cost([{'type':x['type'],'confidence':x['confidence']} for x in document['entities']],p_ai,undetected_risk,ocr_penalty,unknown_regions)
    return {'task':task['label'],'tier':tier,'multiplier':1+p_ai,'rows':rows,'entities':eff,'unresolved_gov_id':unresolved,
            'residual':resid,'cost':cost,'cost_if_everything_redacted_perfectly_by_detector':naive,
            'detected':_count(document['entities']),
            'redacted':_count([x for x in eff if x['action']=='REDACT']),
            'masked':_count([x for x in eff if x['action']=='MASK']),
            'kept':_count([x for x in eff if x['action']=='KEEP'])}

def _count(ents):
    out={}
    for e in ents: out[e['type']]=out.get(e['type'],0)+1
    return out
