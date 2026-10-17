"""Meridian document sanitisation engine.

    from src.sanitize import process_document, Policy, Options
    result = process_document("in.pdf", "out.pdf", Policy(), Options(store_path="ledger/regions.db"))
    result["release"]["status"]   # APPROVED_FOR_RELEASE | REVIEW_REQUIRED | BLOCKED | REJECTED_INPUT | FAILED

See engine.py for the result schema, rules.py for the rule registry, capabilities.py for the
format-capability registry and verify.py for the independent output checks.
"""
from .capabilities import CAPABILITIES, capability_matrix
from .engine import (APPROVED, BLOCKED, ENGINE_VERSION, FAILED, REJECTED, REVIEW, ProcessingResult, process_document,
                     verify_document)
from .policy import Limits, Options, Policy
from .rules import REGISTRY_VERSION, Rule, RuleError, build_registry, builtin_rules, check_pattern_safety

__all__ = ["process_document", "verify_document", "Policy", "Options", "Limits", "ProcessingResult", "APPROVED", "REVIEW", "BLOCKED",
           "REJECTED", "FAILED", "ENGINE_VERSION", "REGISTRY_VERSION", "Rule", "RuleError", "build_registry",
           "builtin_rules", "check_pattern_safety", "CAPABILITIES", "capability_matrix"]
