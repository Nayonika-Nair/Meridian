"""Bridge between sanitisation results and the Document Gate / Leak Budget.

``document_from_result`` turns the REGIONS DETECTED in a sanitisation run into the
``{'fields_present', 'entities'}`` document shape the existing redaction planner understands, so the same
file that was inspected by the engine can be priced and gated by the unchanged budget logic.

Confidence semantics follow the existing model ("chance the entity is removed"):
  * verified acting region          -> the detector confidence is kept (capped at 0.99)
  * failed / unverified acting      -> 0.0 (treated as still present)
"""
from __future__ import annotations

from ..config import WEIGHTS

ENTITY_TO_FIELD = {"PERSON": "full_name", "GOV_ID": "gov_id", "BANK": "bank_account", "ADDRESS": "address",
                   "EMAIL": "email", "PHONE": "phone", "EMP_ID": "employee_id"}


def document_from_result(result: dict, *, after_sanitisation: bool = False) -> dict:
    """Entities of a run. ``after_sanitisation=True`` reflects what is left once verified actions are applied."""
    ents, fields = [], []
    for r in result.get("regions", []):
        if r.get("source_method") == "verification_rescan":
            continue
        etype = r.get("entity_type")
        if etype not in WEIGHTS:
            continue
        conf = float(r.get("confidence", 0.0))
        if after_sanitisation:
            conf = min(0.99, conf) if (r.get("verification_status") == "verified" and r.get("action") in ("REDACT", "MASK", "GENERALISE")) else 0.0
        ents.append({"type": etype, "confidence": round(conf, 4)})
        f = ENTITY_TO_FIELD.get(etype)
        if f and f not in fields:
            fields.append(f)
    return {"fields_present": fields, "entities": ents}
