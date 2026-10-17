"""A9 - rule registry, precedence, hard blocks, allow/deny lists, regex safety, bounded evaluation."""
import time

import pytest

import fixtures_gen as F
from src.sanitize.detect import DetectContext, resolve_action, scan_text, make_replacement
from src.sanitize.policy import Policy
from src.sanitize.rules import (MAX_MATCHES_PER_RULE, REGISTRY_VERSION, Rule, RuleError, build_registry, builtin_rules,
                                check_pattern_safety, evaluate_rule, luhn, iban_mod97, aba_checksum)
from src.sanitize.safety import Deadline, DeadlineExceeded


def ctx(policy=None, **kw):
    policy = policy or Policy(**kw)
    reg = build_registry(policy)
    return DetectContext(registry=reg, policy=policy, doc_id="d" * 64, salt=b"s" * 32, warnings=[])


def found(text, c=None, fmt="pdf"):
    c = c or ctx()
    return {f.rule.id: f for f in scan_text(c, text, fmt)}


# ------------------------------------------------------------------------------ registry shape
def test_registry_is_versioned_with_stable_unique_ids_and_required_categories():
    rules = builtin_rules()
    ids = [r.id for r in rules]
    assert len(ids) == len(set(ids)) and REGISTRY_VERSION
    cats = {r.category for r in rules}
    assert {"GOV_ID", "BANK", "CARD", "PERSON", "EMAIL", "PHONE", "ADDRESS", "DOB", "EMP_ID", "SECRET", "FACE"} <= cats
    for r in rules:
        assert r.severity and r.allowed_actions and r.formats and isinstance(r.precedence, int)
        assert r.verification in ("text", "ocr", "text+ocr") and 0 < r.min_confidence <= 1
        assert r.default_action in r.allowed_actions
    assert any(r.category == "CUSTOM" for r in build_registry(Policy(denylist=["x"])).rules)


def test_describe_exposes_every_contract_field():
    row = build_registry(Policy()).describe()[0]
    for k in ("id", "category", "severity", "enabled", "min_confidence", "allowed_actions", "formats", "precedence", "verification", "hard_block"):
        assert k in row


def test_hard_block_rules_only_allow_redact():
    for r in builtin_rules():
        if r.hard_block:
            assert r.allowed_actions == ("REDACT",)
    with pytest.raises(RuleError):
        Rule(id="X", category="C", entity_type="C", label="x", pattern="a", hard_block=True, allowed_actions=("REDACT", "MASK"))


# ------------------------------------------------------------------------------ detection + validators
@pytest.mark.parametrize("text,rule", [
    (f"SSN {F.SSN}", "GOV.US_SSN"), (f"card {F.CARD}", "CARD.PAN"), (f"Account number: {F.ACCT}", "BANK.ACCOUNT_LABELLED"),
    (f"mail {F.EMAIL}", "EMAIL.ADDRESS"), (f"call {F.PHONE}", "PHONE.NUMBER"), (f"Name: {F.NAME}", "PERSON.NAME_LABELLED"),
    (f"key {F.AWS}", "SECRET.PROVIDER_KEY"), ("DOB: 04/12/1985", "DOB.LABELLED"), ("badge EMP-48213", "EMPID.LABELLED"),
    ("Ship to 221 Baker Street, Springfield, IL 62704", "ADDRESS.POSTAL"),
    ("IBAN GB82 WEST 1234 5698 7654 32", "BANK.IBAN"), ("routing number 021000021", "BANK.ROUTING_US"),
    ("api_key = Zx9fQ2LmN8vB3kTq7RwY", "SECRET.ASSIGNED"),
])
def test_each_category_detected(text, rule):
    assert rule in found(text)


def test_validators_reject_look_alikes():
    assert luhn("4111 1111 1111 1111") and not luhn("4111 1111 1111 1112")
    assert iban_mod97("GB82 WEST 1234 5698 7654 32") and not iban_mod97("GB82 WEST 1234 5698 7654 33")
    assert aba_checksum("021000021") and not aba_checksum("021000022")
    assert "CARD.PAN" not in found("order 4111 1111 1111 1112")
    assert "GOV.US_SSN" not in found("000-12-3456 666-12-3456 900-12-3456")


def test_label_driven_rules_do_not_claim_bare_numbers_or_names():
    f = found(f"Invoice #{F.ACCT} issued to {F.NAME}")
    assert "BANK.ACCOUNT_LABELLED" not in f and "PERSON.NAME_LABELLED" not in f


# ------------------------------------------------------------------------------ precedence and overlaps
def test_overlapping_findings_fold_into_one_region_with_deterministic_winner():
    text = f"SSN: {F.SSN}"
    fs = scan_text(ctx(), text, "pdf")
    assert len(fs) == 1
    f = fs[0]
    assert f.rule.id == "GOV.US_SSN" and "GOV.TAX_ID_LABELLED" in f.also_matched      # lower precedence number wins
    assert text[f.start:f.end] == F.SSN
    # deterministic: shuffle evaluation order via a second context
    assert [(x.rule.id, x.start, x.end) for x in scan_text(ctx(), text, "pdf")] == [(f.rule.id, f.start, f.end)]


def test_overlap_union_never_leaves_part_of_a_sensitive_span():
    text = "Contact jane.doe@example.com now"
    f = scan_text(ctx(), text, "pdf")[0]
    assert text[f.start:f.end] == "jane.doe@example.com"


# ------------------------------------------------------------------------------ mandatory restrictions
def test_permissive_policy_cannot_disable_or_weaken_hard_blocks():
    p = Policy(rule_toggles={"GOV.US_SSN": False, "CARD.PAN": False}, confidence_overrides={"GOV.US_SSN": 0.0},
               category_actions={"GOV_ID": "MASK"}, rule_actions={"CARD.PAN": "GENERALISE"}, allowlist=[F.SSN, F.CARD],
               review_low_confidence=False)
    c = ctx(p)
    fs = {f.rule.id: f for f in scan_text(c, f"{F.SSN} and {F.CARD}", "pdf")}
    assert fs["GOV.US_SSN"].action == "REDACT" and fs["CARD.PAN"].action == "REDACT"
    joined = " | ".join(c.warnings + c.registry.warnings)
    assert "policy_override_rejected:GOV.US_SSN" in joined and "allowlist_ignored_hard_block" in joined
    assert "policy_override_rejected" in joined


def test_allowlist_suppresses_non_mandatory_values_only():
    c = ctx(Policy(allowlist=["support@example.com"]))
    fs = scan_text(c, "mail support@example.com and jane.doe@example.com", "pdf")
    actions = {f.raw: f.action for f in fs}
    assert actions["support@example.com"] == "ALLOWLISTED" and actions["jane.doe@example.com"] == "MASK"


def test_denylist_always_redacts_literal():
    c = ctx(Policy(denylist=["Project Nightingale"]))
    fs = scan_text(c, "the project nightingale roadmap", "pdf")
    assert fs and fs[0].rule.id == "CUSTOM.DENYLIST" and fs[0].action == "REDACT"


def test_rule_toggle_disables_optional_rules():
    assert "EMAIL.ADDRESS" not in found(f"mail {F.EMAIL}", ctx(Policy(rule_toggles={"EMAIL.ADDRESS": False})))


def test_action_validation_and_destination_escalation():
    reg = build_registry(Policy())
    email, bank = reg.get("EMAIL.ADDRESS"), reg.get("BANK.ACCOUNT_LABELLED")
    assert resolve_action(email, Policy()) == "MASK"
    assert resolve_action(email, Policy(category_actions={"EMAIL": "GENERALISE"})) == "GENERALISE"
    c = ctx()
    assert resolve_action(bank, Policy(category_actions={"BANK": "GENERALISE"}), c) == "REDACT"      # not allowed -> strictest allowed
    assert any(w.startswith("action_not_allowed") for w in c.warnings)
    assert resolve_action(bank, Policy(destination_risk="CRITICAL")) == "REDACT"
    assert resolve_action(email, Policy(rule_actions={"EMAIL.ADDRESS": "BOGUS"}), c) == "MASK"


def test_low_confidence_goes_to_review_unless_mandatory():
    c = ctx(Policy(confidence_overrides={"EMAIL.ADDRESS": 0.999}))
    f = scan_text(c, f"mail {F.EMAIL}", "pdf")[0]
    assert f.action == "FLAG_REVIEW" and f.status == "review_required"
    c2 = ctx(Policy(confidence_overrides={"EMAIL.ADDRESS": 0.999}, review_low_confidence=False))
    assert scan_text(c2, f"mail {F.EMAIL}", "pdf")[0].action == "NONE"
    # mandatory rules fail closed: even a hypothetical low base confidence still acts
    import dataclasses
    reg = build_registry(Policy())
    weak = dataclasses.replace(reg.get("GOV.US_SSN"), base_confidence=0.2)
    assert len(evaluate_rule(weak, F.SSN)) == 1


def test_replacement_text_masks_without_revealing_more_than_a_third():
    assert make_replacement("MASK", "BANK", "98765432") == "******32"
    assert make_replacement("MASK", "BANK", "12345") == "****5"                   # never more than a third is kept
    assert make_replacement("MASK", "BANK", "12") == "**"                         # very short values keep nothing
    assert make_replacement("MASK", "EMAIL", "jane.doe@example.com") == "j***@example.com"
    assert make_replacement("GENERALISE", "DOB", "04/12/1985") == "1985"
    assert make_replacement("REDACT", "GOV_ID", F.SSN) == "[REDACTED:ID]"


# ------------------------------------------------------------------------------ regex safety + bounded evaluation
@pytest.mark.parametrize("bad", [r"(a+)+$", r"(a*)*b", r"(\w+\s?)+$", r"(.*a){12}", r"(x+x+)+y", r"(a)\1", "[", "", "a" * 400])
def test_unsafe_or_invalid_patterns_are_rejected(bad):
    with pytest.raises(RuleError):
        check_pattern_safety(bad)


def test_custom_rule_applied_and_unsafe_custom_rule_rejected_with_warning():
    p = Policy(custom_rules=[{"id": "badge", "pattern": r"BADGE-\d{6}", "severity": "HIGH", "default_action": "REDACT"},
                             {"id": "evil", "pattern": r"(a+)+$"}])
    reg = build_registry(p)
    assert reg.get("CUSTOM.badge") and reg.get("CUSTOM.evil") is None
    assert any(w.startswith("custom_rule_rejected:CUSTOM.evil") for w in reg.warnings)
    c = DetectContext(registry=reg, policy=p, doc_id="d" * 64, salt=b"s" * 32)
    assert any(f.rule.id == "CUSTOM.badge" for f in scan_text(c, "id BADGE-123456", "pdf"))


def test_custom_rules_can_never_be_hard_blocks_or_outrank_mandatory_precedence_by_accident():
    p = Policy(custom_rules=[{"id": "x", "pattern": r"\d{3}-\d{2}-\d{4}", "precedence": 1, "allowed_actions": ["MASK"], "default_action": "MASK"}])
    c = DetectContext(registry=build_registry(p), policy=p, doc_id="d" * 64, salt=b"s" * 32)
    f = scan_text(c, f"ref {F.SSN}", "pdf")
    # the custom rule matches first (precedence 1) but the mandatory rule's span is merged in and its REDACT wins
    assert f[0].action == "REDACT" and f[0].hard_block


def test_evaluation_is_bounded_in_matches_and_time():
    rule = build_registry(Policy()).get("EMAIL.ADDRESS")
    text = " ".join(f"u{i}@example.com" for i in range(MAX_MATCHES_PER_RULE + 500))
    assert len(evaluate_rule(rule, text)) <= MAX_MATCHES_PER_RULE
    big = "a" * 50_000 + F.SSN
    t0 = time.monotonic()
    evaluate_rule(build_registry(Policy()).get("GOV.US_SSN"), big)
    assert time.monotonic() - t0 < 3
    dl = Deadline(0.0)
    time.sleep(0.01)
    with pytest.raises(DeadlineExceeded):
        evaluate_rule(rule, "x@y.com", deadline=dl)


def test_chunked_evaluation_finds_matches_across_chunk_boundaries():
    from src.sanitize import rules
    pad = "x " * (rules.CHUNK // 2 - 3)
    text = pad + F.EMAIL + " tail"
    got = evaluate_rule(build_registry(Policy()).get("EMAIL.ADDRESS"), text)
    assert any(m.value == F.EMAIL for m in got)
