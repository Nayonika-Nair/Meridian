"""Data DNA: a six-axis privacy profile derived from detected entities."""
from .config import WEIGHTS

CATEGORIES=['Identity','Financial','Contact','Operational','Regulatory','AI sensitivity']
AI_AFFINITY=0.9   # how damaging any detected entity is if an AI system ingests it
# how strongly each entity type contributes to each axis (0..1) -- configurable policy
AFFINITY={
 'GOV_ID':{'Identity':1.0,'Regulatory':1.0},
 'BANK':{'Financial':1.0,'Regulatory':0.7},
 'PERSON':{'Identity':0.8,'Contact':0.3},
 'EMAIL':{'Contact':1.0,'Identity':0.3},
 'PHONE':{'Contact':1.0,'Identity':0.2},
 'ADDRESS':{'Contact':0.8,'Identity':0.4,'Regulatory':0.3},
 'EMP_ID':{'Operational':1.0,'Identity':0.5},
}

def data_dna(entities,weights=None,exposure_key=None):
    """score_axis = 1 - prod(1 - affinity*weight/10)  (saturating, noisy-OR style), per entity occurrence.
    If exposure_key is given, each entity's contribution is scaled by entity[exposure_key] (0..1)."""
    w=weights or WEIGHTS; keep={c:1.0 for c in CATEGORIES}
    for e in entities:
        t=e.get('type'); ex=float(e.get(exposure_key,1.0)) if exposure_key else 1.0; sev=min(1.0,w.get(t,1)/10.0)*ex
        for c,a in AFFINITY.get(t,{}).items(): keep[c]*=1-min(1.0,a*sev)
        keep['AI sensitivity']*=1-min(1.0,AI_AFFINITY*sev)
    return {c:1-keep[c] for c in CATEGORIES}
