"""Central, analyst-editable policy defaults for Meridian.

Everything here is a *policy default*, not a measured fact. The UI shows the
values that are in force so no calculation is hidden.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

BASE = 1000.0                       # base allowance per vendor per window
WINDOW_DAYS = 30                    # budget window used to derive the live balance

# --- Data sensitivity weights (points per fully-exposed entity) -------------
WEIGHTS = {"GOV_ID": 10, "BANK": 10, "PERSON": 3, "EMAIL": 2, "PHONE": 2, "EMP_ID": 4, "ADDRESS": 3,
           # extended by the document sanitisation engine (src/sanitize): same scale, same meaning
           "CARD": 10, "SECRET": 10, "DOB": 4, "FACE": 5}
# Entity types whose leftover presence hard-blocks a send regardless of budget (mandatory rules)
HARD_BLOCK_TYPES = ("GOV_ID", "CARD", "SECRET")

# --- Criticality factors (also stored per vendor in vendors.json) -----------
CRITICALITY = {"Critical": 0.50, "High": 0.75, "Medium": 1.00, "Low": 1.25}

# --- Brake thresholds: remaining fraction of the initial allowance ----------
BRAKE = {"AUTO": 0.50, "STRICT": 0.20, "REVIEW": 0.05}

# --- Historical exposure -> allowance (Feature 1) ---------------------------
HISTORY_MAX_CUT = 0.40              # privacy debt can shave at most 40% off the allowance
DEBT_CRITICAL_MULT = 2.0            # debt >= 2 x BASE is CRITICAL

# --- Privacy-debt tiers (multiples of BASE) ---------------------------------
DEBT_TIERS = [(0.5, "LOW"), (1.0, "MODERATE"), (2.0, "HIGH"), (float("inf"), "CRITICAL")]

# --- P_ai tiers (destination risk) ------------------------------------------
P_AI_TIERS = [(0.25, "LOW"), (0.50, "MODERATE"), (0.75, "HIGH"), (1.01, "CRITICAL")]

# --- Evidence honesty --------------------------------------------------------
# 1.0 keeps the original behaviour. Lower it (e.g. 0.5) to discount edges whose
# evidence only shows *AI use*, not that Meridian data reaches the AI.
AI_USE_FLOW_DISCOUNT = 1.0
UNKNOWN_PROVIDER_PRIOR = 0.5        # modeling assumption for unresolved model providers

# --- Destination-aware redaction --------------------------------------------
# entity weight at/above which a *required* field is masked instead of sent clear
MASK_WEIGHT_MIN = {"LOW": None, "MODERATE": None, "HIGH": 8, "CRITICAL": 3}
# optional fields with weight >= this are dropped at the given tier
OPTIONAL_DROP_WEIGHT = {"LOW": None, "MODERATE": 3, "HIGH": 0, "CRITICAL": 0}
MASK_RESIDUAL = 0.30                # a masked / tokenised value still carries 30% of its weight

SEVERITY = ["LOW", "MODERATE", "HIGH", "CRITICAL"]
EMOJI = {"LOW": "🟢", "MODERATE": "🟡", "HIGH": "🟠", "CRITICAL": "🔴"}

def tier_from(value, tiers):
    for limit, name in tiers:
        if value < limit:
            return name
    return tiers[-1][1]

def p_ai_tier(p): return tier_from(p, P_AI_TIERS)
def debt_tier(debt, base=BASE): return tier_from(debt / base if base else 0, DEBT_TIERS)
def worst(*tiers): return max(tiers, key=SEVERITY.index)
