"""Explainability: every decision answers WHY."""
from .config import BRAKE, p_ai_tier

def decision_reasons(action,balance,initial,cost,unresolved_id,p_ai,forced_review=False):
    rem=((balance-cost)/initial) if initial else 0.0
    reasons=[]
    if unresolved_id:
        reasons.append('A government ID is still in the file and was not clearly removed.')
    reasons.append(f'Budget left after this file = {max(rem,0):.1%} (AUTO > {BRAKE["AUTO"]:.0%}, STRICT > {BRAKE["STRICT"]:.0%}, REVIEW > {BRAKE["REVIEW"]:.0%}).')
    reasons.append(f'Chance this destination uses AI = {p_ai:.1%} ({p_ai_tier(p_ai)}). This is an estimate.')
    if forced_review: reasons.append('An approved rule says a person must review sends to this vendor.')
    if cost>balance: reasons.append(f'This file costs {cost:.1f} but only {balance:.1f} budget is left.')
    if unresolved_id: policy='A leftover government ID always blocks the send.'
    elif action=='BLOCK': policy='Budget is gone. Sends are blocked.'
    elif action=='REVIEW': policy='Budget is almost gone. A person must approve.'
    elif action=='STRICT': policy='Budget is low. Strip more fields and check the file.'
    else: policy='Within budget and no hard rule was hit.'
    return {'action':action,'reasons':reasons,'policy':policy}
