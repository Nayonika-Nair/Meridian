"""Self-healing privacy policy.

New evidence -> Digital Twin changes -> P_ai changes -> budget changes -> policy tightens.

SAFETY: this module only PROPOSES. Nothing is applied until a human approves it through store.approve_policy().
Only tightening is proposed automatically; relaxations are flagged for manual review.
"""
from .config import WEIGHTS, SEVERITY, EMOJI

DELTA_TRIGGER = 0.15          # a P_ai jump this large tightens policy even inside the same tier


def propose_policy(before, after):
    """before / after are engine.vendor_state() dicts for the same vendor (same ledger, different evidence)."""
    p0, p1 = before['p_ai'], after['p_ai']
    t0, t1 = before['p_ai_tier'], after['p_ai_tier']
    delta = p1 - p0
    cut = 1 - after['allowance'] / before['allowance'] if before['allowance'] else 0.0
    base = {'vendor_id': after['vendor_id'], 'vendor': after['name'], 'p_ai_before': p0, 'p_ai_after': p1,
            'tier_before': t0, 'tier_after': t1, 'budget_before': before['allowance'], 'budget_after': after['allowance'],
            'budget_change_pct': -cut, 'status': 'PROPOSED',
            'honesty': 'P_ai is a model-based estimate; this proposal is a recommendation, not an applied change.'}
    tightening = SEVERITY.index(t1) > SEVERITY.index(t0) or delta >= DELTA_TRIGGER
    if not tightening:
        direction = 'relax' if delta < -0.05 else 'none'
        return {**base, 'direction': direction, 'extra_redact': [], 'require_review': False, 'actions': [],
                'summary': ('Risk decreased. No automatic change; relaxing policy is a manual decision.'
                            if direction == 'relax' else 'No material risk change. No policy change proposed.')}
    high_floor = 4 if t1 == 'CRITICAL' else 10                 # weight at/above which an entity is always redacted
    redact = sorted([e for e, w in WEIGHTS.items() if w >= high_floor], key=lambda e: -WEIGHTS[e])
    review = SEVERITY.index(t1) >= SEVERITY.index('HIGH') or delta >= DELTA_TRIGGER
    actions = [f'Always redact {e} for this destination' for e in redact]
    if review:
        actions.append('Require human review before release')
    if cut > 0:
        actions.append(f'Leak budget falls {cut:.0%} ({before["allowance"]:,.0f} -> {after["allowance"]:,.0f}) automatically via P_ai')
    return {**base, 'direction': 'tighten', 'extra_redact': redact, 'require_review': review, 'actions': actions,
            'summary': f'P_ai {p0:.0%} -> {p1:.0%} ({EMOJI[t0]} {t0} -> {EMOJI[t1]} {t1}).'}
