"""Privacy debt (history) and privacy time-to-failure (forecast).

Budget answers  : "Can I send THIS document?"
Privacy debt    : "How much exposure have we accumulated historically?"  (cumulative ledger spend)
Time-to-failure : "At this burn rate, when does the budget run out?"
"""
import math, time, datetime
from .config import BRAKE, WINDOW_DAYS, BASE, debt_tier

DAY = 86400.0
DEFAULT_DOCS_PER_DAY = 10          # used ONLY for projections when history is thin
DEFAULT_AVG_DOC_COST = 8.0
MIN_HISTORY_DAYS = 3


def privacy_debt(records):
    """Cumulative exposure points ever spent toward this destination."""
    return float(sum(r['cost'] for r in records))


def debt_state(records, base=BASE):
    d = privacy_debt(records)
    return {'debt': d, 'tier': debt_tier(d, base), 'ratio_to_base': d / base if base else 0.0}


def monthly_series(records):
    """[(YYYY-MM, spend_in_month, cumulative_debt)] oldest first."""
    by = {}
    for r in records:
        k = datetime.datetime.fromtimestamp(r['timestamp'], datetime.timezone.utc).strftime('%Y-%m')
        by[k] = by.get(k, 0.0) + r['cost']
    out, cum = [], 0.0
    for k in sorted(by):
        cum += by[k]
        out.append((k, by[k], cum))
    return out


def window_spend(records, now=None, window_days=WINDOW_DAYS):
    now = time.time() if now is None else now
    return float(sum(r['cost'] for r in records if r['timestamp'] >= now - window_days * DAY))


def time_to_failure(balance, allowance, records, now=None, docs_per_day=None, avg_doc_cost=None,
                    volume_multiplier=1.0, lookback_days=14):
    """Estimate when the budget is exhausted.
    basis='history'    -> daily burn measured from the ledger (>= MIN_HISTORY_DAYS distinct days)
    basis='projection' -> burn assumed from docs/day x avg cost (clearly labelled; not measured)."""
    now = time.time() if now is None else now
    spend = [r for r in records if r['cost'] > 0 and r['timestamp'] >= now - lookback_days * DAY]
    days_seen = {int(r['timestamp'] // DAY) for r in spend}
    if len(days_seen) >= MIN_HISTORY_DAYS:
        span = min(lookback_days, max(1.0, (now - min(r['timestamp'] for r in spend)) / DAY))
        daily, basis = sum(r['cost'] for r in spend) / span, 'history'
        note = f'Measured from {len(spend)} ledger entries over {span:.0f} days.'
    else:
        daily = (docs_per_day or DEFAULT_DOCS_PER_DAY) * (avg_doc_cost or DEFAULT_AVG_DOC_COST)
        basis = 'projection'
        note = (f'PROJECTION: only {len(days_seen)} day(s) of history (need {MIN_HISTORY_DAYS}); '
                f'assumes {docs_per_day or DEFAULT_DOCS_PER_DAY} docs/day x {avg_doc_cost or DEFAULT_AVG_DOC_COST:.1f} pts.')
    daily *= volume_multiplier
    to_zero = balance / daily if daily > 0 else math.inf
    floor = BRAKE['REVIEW'] * allowance
    to_block = max(0.0, (balance - floor) / daily) if daily > 0 else math.inf
    to_strict = max(0.0, (balance - BRAKE['AUTO'] * allowance) / daily) if daily > 0 else math.inf
    horizon = 30 if math.isinf(to_zero) else int(min(90, max(7, math.ceil(to_zero * 1.25))))
    forecast = [(d, max(0.0, balance - daily * d)) for d in range(horizon + 1)]
    return {'days': to_zero, 'days_to_block': to_block, 'days_to_leave_auto': to_strict, 'daily_burn': daily,
            'basis': basis, 'is_projection': basis == 'projection', 'note': note, 'forecast': forecast}
