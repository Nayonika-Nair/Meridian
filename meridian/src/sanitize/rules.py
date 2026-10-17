"""Central, versioned privacy-rule registry (A9) and bounded rule evaluation.

Design rules
------------
* Every rule has a stable ID, category, severity, confidence threshold, allowed actions,
  supported formats, a precedence number (lower wins) and verification requirements.
* Rules marked ``hard_block`` are MANDATORY: policy cannot disable them, allow-list them,
  lower their action below REDACT, or relax their confidence threshold (they fail closed).
* Custom patterns are statically screened (length, nested quantifiers, back-references)
  and every evaluation is bounded (text chunking, match caps, wall-clock deadline).
  If the optional ``regex`` package is installed its per-call timeout is also used.
* Precedence is deterministic: (precedence, -severity, -span length, start, rule id).
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field, replace
from typing import Callable, Iterable, Iterator

try:  # optional hard regex timeout
    import regex as _regex  # type: ignore
except Exception:  # pragma: no cover - depends on environment
    _regex = None

REGISTRY_VERSION = "2026.10.1"

SEVERITY = ("LOW", "MODERATE", "HIGH", "CRITICAL")
ACTIONS = ("FLAG_REVIEW", "GENERALISE", "MASK", "REDACT")          # ascending strictness
STRICTNESS = {a: i for i, a in enumerate(ACTIONS)}
FORMATS = ("pdf", "docx", "pptx", "image")

MAX_PATTERN_LEN = 300
MAX_UNBOUNDED_REPEATS = 8
CHUNK = 20_000
CHUNK_OVERLAP = 256
MAX_MATCHES_PER_RULE = 2_000
MAX_TEXT_PER_UNIT = 400_000


class RuleError(ValueError):
    """Raised for unsafe or malformed rule definitions."""


# --------------------------------------------------------------------------- validators
def _digits(s: str) -> str:
    return re.sub(r"\D", "", s)


def luhn(value: str) -> float | None:
    d = _digits(value)
    if not 13 <= len(d) <= 19 or len(set(d)) == 1:
        return None
    total, alt = 0, False
    for ch in reversed(d):
        n = int(ch)
        if alt:
            n *= 2
            n -= 9 if n > 9 else 0
        total += n
        alt = not alt
    return 1.0 if total % 10 == 0 else None


def card_number(value: str) -> float | None:
    """Luhn + a plausible issuer prefix / length. Cuts coincidental Luhn hits on timestamps and order numbers."""
    if luhn(value) is None:
        return None
    d = _digits(value)
    n = len(d)
    p2, p3, p4, p6 = int(d[:2]), int(d[:3]), int(d[:4]), int(d[:6])
    ok = (
        (d[0] == "4" and n in (13, 16, 19)) or
        (51 <= p2 <= 55 and n == 16) or (2221 <= p4 <= 2720 and n == 16) or
        (p2 in (34, 37) and n == 15) or
        (d[:4] == "6011" and n in (16, 19)) or (p2 == 65 and n in (16, 19)) or (644 <= p3 <= 649 and n in (16, 19)) or
        (3528 <= p4 <= 3589 and n in (16, 19)) or
        (p2 in (36, 38, 39) and n in (14, 16)) or (300 <= p3 <= 305 and n == 14) or
        (p2 == 62 and 16 <= n <= 19) or (p2 in (50, 56, 57, 58, 63, 67) and 12 <= n <= 19) or
        (p6 >= 0 and d[:2] == "60" and n in (16, 19))
    )
    return 1.0 if ok else None


def iban_mod97(value: str) -> float | None:
    s = re.sub(r"\s", "", value).upper()
    if not 15 <= len(s) <= 34:
        return None
    rearr = s[4:] + s[:4]
    num = "".join(str(int(c, 36)) for c in rearr)
    return 1.0 if int(num) % 97 == 1 else None


def aba_checksum(value: str) -> float | None:
    d = _digits(value)
    if len(d) != 9:
        return None
    w = (3, 7, 1, 3, 7, 1, 3, 7, 1)
    return 1.0 if sum(int(a) * b for a, b in zip(d, w)) % 10 == 0 else None


def not_trivial_digits(value: str) -> float | None:
    d = _digits(value)
    if len(d) < 6 or len(set(d)) < 3:
        return None
    return 1.0


def plausible_phone(value: str) -> float | None:
    d = _digits(value)
    return 1.0 if 10 <= len(d) <= 15 and len(set(d)) > 2 else None


def plausible_dob(value: str) -> float | None:
    m = re.search(r"(\d{4})", value)
    if m:
        y = int(m.group(1))
        return 1.0 if 1900 <= y <= time.gmtime().tm_year else None
    return 0.8          # two-digit years: plausible but ambiguous


def ssn_shape(value: str) -> float | None:
    d = _digits(value)
    if len(d) != 9 or d[:3] in ("000", "666") or d[:3] >= "900" or d[3:5] == "00" or d[5:] == "0000":
        return None
    return 1.0


def secret_entropy(value: str) -> float | None:
    v = value.strip("\"' ")
    if len(v) < 16 or len(set(v)) < 6:
        return None
    return 1.0


VALIDATORS: dict[str, Callable[[str], float | None]] = {
    "luhn": luhn, "card": card_number, "iban": iban_mod97, "aba": aba_checksum, "digits": not_trivial_digits,
    "phone": plausible_phone, "dob": plausible_dob, "ssn": ssn_shape, "secret": secret_entropy,
}


# --------------------------------------------------------------------------- rule model
@dataclass(frozen=True)
class Rule:
    id: str
    category: str                   # GOV_ID, BANK, CARD, PERSON, EMAIL, PHONE, ADDRESS, DOB, EMP_ID, SECRET, CUSTOM, FACE
    entity_type: str                # maps to Meridian's WEIGHTS vocabulary (or an extension of it)
    label: str
    kind: str = "regex"             # regex | face | literal
    pattern: str = ""
    group: int = 0                  # capture group that holds the sensitive value (0 = whole match)
    flags: int = re.IGNORECASE
    validator: str | None = None
    base_confidence: float = 0.9
    severity: str = "HIGH"
    min_confidence: float = 0.6
    allowed_actions: tuple[str, ...] = ("REDACT", "MASK")
    default_action: str = "REDACT"
    formats: tuple[str, ...] = FORMATS
    precedence: int = 100
    verification: str = "text+ocr"  # text | ocr | text+ocr
    hard_block: bool = False
    enabled: bool = True
    version: int = 1
    description: str = ""

    def __post_init__(self):
        if self.severity not in SEVERITY:
            raise RuleError(f"{self.id}: bad severity {self.severity}")
        bad = [a for a in self.allowed_actions if a not in STRICTNESS]
        if bad or self.default_action not in self.allowed_actions:
            raise RuleError(f"{self.id}: invalid actions {bad or self.default_action}")
        if self.hard_block and self.allowed_actions != ("REDACT",):
            raise RuleError(f"{self.id}: hard-block rules may only allow REDACT")
        if self.kind == "regex":
            check_pattern_safety(self.pattern)


# --------------------------------------------------------------------------- regex safety
def _sre_parse():
    try:
        import re._parser as p  # py>=3.11
        return p
    except Exception:           # pragma: no cover
        import sre_parse as p   # type: ignore
        return p


def check_pattern_safety(pattern: str, *, max_len: int = MAX_PATTERN_LEN) -> None:
    """Static screen for catastrophic-backtracking shapes. Raises RuleError.

    This is a heuristic gate, not a proof. Evaluation is additionally bounded by
    chunking, match caps and a deadline (and the optional ``regex`` timeout).
    """
    if not isinstance(pattern, str) or not pattern:
        raise RuleError("empty pattern")
    if len(pattern) > max_len:
        raise RuleError(f"pattern longer than {max_len} characters")
    try:
        re.compile(pattern)
    except re.error as exc:
        raise RuleError(f"invalid regex: {exc}") from exc
    parse = _sre_parse()
    try:
        tree = parse.parse(pattern)
    except Exception as exc:    # pragma: no cover
        raise RuleError(f"unparseable regex: {exc}") from exc
    MAXREPEAT = parse.MAXREPEAT
    unbounded = 0

    def walk(items, outer_unbounded: bool, outer_wide: bool):
        """outer_unbounded: inside a repeat with no upper bound; outer_wide: inside a repeat allowing > 5 iterations."""
        nonlocal unbounded
        for op, av in items:
            name = str(op)
            if name in ("MAX_REPEAT", "MIN_REPEAT"):
                lo, hi, sub = av
                is_unb = hi == MAXREPEAT
                if is_unb:
                    unbounded += 1
                    if outer_unbounded or outer_wide:
                        raise RuleError("nested unbounded quantifier (catastrophic backtracking risk)")
                if outer_unbounded and _contains_repeat(sub):
                    raise RuleError("nested quantifier inside an unbounded repeat (catastrophic backtracking risk)")
                if hi > 5 and _contains_repeat(sub):
                    raise RuleError("nested quantifier inside a wide repeat (catastrophic backtracking risk)")
                walk(sub, outer_unbounded or is_unb, outer_wide or hi > 5)
            elif name == "GROUPREF":
                raise RuleError("back-references are not allowed in rules")
            elif name == "SUBPATTERN":
                walk(av[-1], outer_unbounded, outer_wide)
            elif name == "BRANCH":
                for branch in av[1]:
                    walk(branch, outer_unbounded, outer_wide)
            elif name in ("ASSERT", "ASSERT_NOT"):
                walk(av[1], outer_unbounded, outer_wide)

    def _contains_repeat(items) -> bool:
        for op, av in items:
            name = str(op)
            if name in ("MAX_REPEAT", "MIN_REPEAT") and av[1] > 1 and av[0] != av[1]:
                return True
            if name in ("MAX_REPEAT", "MIN_REPEAT") and _contains_repeat(av[2]):
                return True
            if name == "SUBPATTERN" and _contains_repeat(av[-1]):
                return True
            if name == "BRANCH" and any(_contains_repeat(b) for b in av[1]):
                return True
        return False

    walk(tree, False, False)
    if unbounded > MAX_UNBOUNDED_REPEATS:
        raise RuleError(f"too many unbounded quantifiers ({unbounded} > {MAX_UNBOUNDED_REPEATS})")


# --------------------------------------------------------------------------- built-in registry
def _r(**kw) -> Rule:
    return Rule(**kw)


def builtin_rules() -> list[Rule]:
    S = ("pdf", "docx", "pptx", "image")
    return [
        # -- Government identifiers (mandatory hard blocks) --
        _r(id="GOV.US_SSN", category="GOV_ID", entity_type="GOV_ID", label="US Social Security number",
           pattern=r"(?<![\d-])(?!000|666|9\d\d)\d{3}[- ](?!00)\d{2}[- ](?!0000)\d{4}(?![\d-])", validator="ssn",
           base_confidence=0.97, severity="CRITICAL", min_confidence=0.7, allowed_actions=("REDACT",),
           default_action="REDACT", precedence=10, hard_block=True,
           description="Formatted US SSN (dashes or spaces)."),
        _r(id="GOV.TAX_ID_LABELLED", category="GOV_ID", entity_type="GOV_ID", label="Labelled tax / national / passport ID",
           pattern=r"\b(?:ssn|tin|ein|tax\s*id|national\s*id|passport(?:\s*(?:no\.?|number))?|nino|sin)\b\s*[:#\-]?\s*([A-Z0-9][A-Z0-9\-]{4,18}[A-Z0-9])",
           group=1, validator="digits", base_confidence=0.92, severity="CRITICAL", min_confidence=0.7,
           allowed_actions=("REDACT",), default_action="REDACT", precedence=11, hard_block=True,
           description="Identifier introduced by an explicit label such as 'Passport No'."),
        # -- Payment cards (mandatory) --
        _r(id="CARD.PAN", category="CARD", entity_type="CARD", label="Payment card number (Luhn-valid, known issuer prefix)",
           pattern=r"(?<![\d-])(?:\d[ -]?){12,18}\d(?![\d-])", validator="card", base_confidence=0.96,
           severity="CRITICAL", min_confidence=0.8, allowed_actions=("REDACT",), default_action="REDACT",
           precedence=12, hard_block=True, description="13-19 digit PAN passing the Luhn check."),
        # -- Secrets (mandatory) --
        _r(id="SECRET.PROVIDER_KEY", category="SECRET", entity_type="SECRET", label="Provider API key / token",
           pattern=r"\b(?:AKIA[0-9A-Z]{16}|ASIA[0-9A-Z]{16}|sk-[A-Za-z0-9_\-]{20,}|ghp_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{40,}|xox[baprs]-[A-Za-z0-9\-]{10,}|AIza[0-9A-Za-z_\-]{35})\b",
           flags=0, base_confidence=0.98, severity="CRITICAL", min_confidence=0.8, allowed_actions=("REDACT",),
           default_action="REDACT", precedence=13, hard_block=True, description="Well-known key prefixes."),
        _r(id="SECRET.PRIVATE_KEY", category="SECRET", entity_type="SECRET", label="PEM private key block",
           pattern=r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY(?: BLOCK)?-----[\s\S]{0,6000}?-----END (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY(?: BLOCK)?-----",
           flags=0, base_confidence=0.99, severity="CRITICAL", min_confidence=0.8, allowed_actions=("REDACT",),
           default_action="REDACT", precedence=13, hard_block=True, formats=("docx", "pptx", "pdf")),
        _r(id="SECRET.JWT", category="SECRET", entity_type="SECRET", label="JSON Web Token",
           pattern=r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\b", flags=0,
           base_confidence=0.95, severity="CRITICAL", min_confidence=0.8, allowed_actions=("REDACT",),
           default_action="REDACT", precedence=14, hard_block=True),
        _r(id="SECRET.ASSIGNED", category="SECRET", entity_type="SECRET", label="Credential assigned to a secret-like name",
           pattern=r"\b(?:api[_\- ]?key|secret|token|passwd|password|client[_\- ]secret|access[_\- ]key)\b\s*[:=]\s*[\"']?([A-Za-z0-9_\-./+=!@#$%^&*]{12,})",
           group=1, validator="secret", base_confidence=0.9, severity="CRITICAL", min_confidence=0.7,
           allowed_actions=("REDACT",), default_action="REDACT", precedence=15, hard_block=True),
        # -- Bank identifiers --
        _r(id="BANK.IBAN", category="BANK", entity_type="BANK", label="IBAN (mod-97 valid)",
           pattern=r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,4})?\b", flags=0, validator="iban",
           base_confidence=0.95, severity="HIGH", min_confidence=0.7, allowed_actions=("REDACT", "MASK"),
           default_action="MASK", precedence=20),
        _r(id="BANK.ROUTING_US", category="BANK", entity_type="BANK", label="US ABA routing number",
           pattern=r"\b(?:routing|aba|rtn)\b[^\d\n]{0,20}(\d{9})\b", group=1, validator="aba",
           base_confidence=0.93, severity="HIGH", min_confidence=0.7, allowed_actions=("REDACT", "MASK"),
           default_action="MASK", precedence=21),
        _r(id="BANK.ACCOUNT_LABELLED", category="BANK", entity_type="BANK", label="Labelled bank account number",
           pattern=r"\b(?:account|acct|a/c)\b\s*(?:no\.?|number|num|#)?\s*[:#\-]?\s*(\d{6,17})\b", group=1,
           validator="digits", base_confidence=0.9, severity="HIGH", min_confidence=0.65,
           allowed_actions=("REDACT", "MASK"), default_action="MASK", precedence=22),
        # -- Contact data --
        _r(id="EMAIL.ADDRESS", category="EMAIL", entity_type="EMAIL", label="Email address",
           pattern=r"(?<![\w.+-])[A-Za-z0-9._%+\-]{1,64}@(?:[A-Za-z0-9\-]{1,63}\.){1,5}[A-Za-z]{2,24}(?![\w-])",
           base_confidence=0.97, severity="MODERATE", min_confidence=0.7,
           allowed_actions=("REDACT", "MASK", "GENERALISE"), default_action="MASK", precedence=40),
        _r(id="PHONE.NUMBER", category="PHONE", entity_type="PHONE", label="Phone number",
           pattern=r"(?<![\w-])(?:\+?\d{1,3}[\s.\-]?)?(?:\(\d{3}\)\s?|\d{3}[\s.\-])\d{3}[\s.\-]\d{4}(?![\w-])",
           validator="phone", base_confidence=0.88, severity="MODERATE", min_confidence=0.6,
           allowed_actions=("REDACT", "MASK", "GENERALISE"), default_action="MASK", precedence=41),
        _r(id="ADDRESS.POSTAL", category="ADDRESS", entity_type="ADDRESS", label="Postal address",
           pattern=r"\b\d{1,6}[ \t]+(?:[A-Z][A-Za-z.']+[ \t]+){1,3}(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Way|Court|Ct|Place|Pl)\b\.?(?:,?[ \t]+[A-Z][A-Za-z.]+(?:[ \t][A-Z][A-Za-z.]+)?,?[ \t]+[A-Z]{2}[ \t]+\d{5}(?:-\d{4})?)?",
           flags=0, base_confidence=0.82, severity="MODERATE", min_confidence=0.6,
           allowed_actions=("REDACT", "MASK", "GENERALISE"), default_action="REDACT", precedence=42),
        # -- Person data --
        _r(id="PERSON.NAME_LABELLED", category="PERSON", entity_type="PERSON", label="Name introduced by a label",
           pattern=r"\b(?i:full\s+name|name|customer|account\s+holder|employee|beneficiary|payee|patient|applicant)\b[ \t]*[:\-][ \t]*([A-Z][a-z'\-]+(?:[ \t]+[A-Z][a-z'\-]+){1,3})\b",
           flags=0, group=1, base_confidence=0.85, severity="MODERATE", min_confidence=0.6,
           allowed_actions=("REDACT", "MASK"), default_action="REDACT", precedence=50,
           description="Label-driven: bare names without a label are NOT detected (no NER model is bundled)."),
        _r(id="DOB.LABELLED", category="DOB", entity_type="DOB", label="Date of birth",
           pattern=r"\b(?:dob|d\.o\.b\.?|date\s+of\s+birth|born(?:\s+on)?)\b\s*[:\-]?\s*(\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}|\d{4}-\d{2}-\d{2})\b",
           group=1, validator="dob", base_confidence=0.93, severity="HIGH", min_confidence=0.7,
           allowed_actions=("REDACT", "GENERALISE"), default_action="GENERALISE", precedence=30),
        _r(id="EMPID.LABELLED", category="EMP_ID", entity_type="EMP_ID", label="Employee ID",
           pattern=r"\b(?:EMP|EID|STAFF)[-_ ]?\d{4,8}\b", flags=0, base_confidence=0.9, severity="MODERATE",
           min_confidence=0.6, allowed_actions=("REDACT", "MASK"), default_action="MASK", precedence=45),
        # -- Face regions (image-borne) --
        _r(id="FACE.REGION", category="FACE", entity_type="FACE", label="Detected face region (no identity recognition)",
           kind="face", base_confidence=0.8, severity="HIGH", min_confidence=0.5,
           allowed_actions=("REDACT",), default_action="REDACT", precedence=60, enabled=False,
           verification="ocr", formats=S, description="Opt-in local detector; detection only, never identification."),
    ]


class RuleRegistry:
    """Resolved, ordered set of rules after applying a policy."""

    def __init__(self, rules: Iterable[Rule], version: str = REGISTRY_VERSION, warnings: list[str] | None = None):
        self.rules = sorted(rules, key=lambda r: (r.precedence, -SEVERITY.index(r.severity), r.id))
        self.version = version
        self.warnings = warnings or []
        self._by_id = {r.id: r for r in self.rules}

    def get(self, rule_id: str) -> Rule | None:
        return self._by_id.get(rule_id)

    def for_format(self, fmt: str) -> list[Rule]:
        return [r for r in self.rules if r.enabled and fmt in r.formats]

    def ids(self) -> list[str]:
        return [r.id for r in self.rules]

    def describe(self) -> list[dict]:
        return [{"id": r.id, "category": r.category, "severity": r.severity, "enabled": r.enabled,
                 "min_confidence": r.min_confidence, "allowed_actions": list(r.allowed_actions),
                 "default_action": r.default_action, "formats": list(r.formats), "precedence": r.precedence,
                 "verification": r.verification, "hard_block": r.hard_block, "kind": r.kind,
                 "version": r.version, "label": r.label} for r in self.rules]


def build_registry(policy=None) -> RuleRegistry:
    """Apply a Policy to the built-in rules. Mandatory rules cannot be weakened."""
    warnings: list[str] = []
    rules = {r.id: r for r in builtin_rules()}
    if policy is not None:
        for rid, enabled in (policy.rule_toggles or {}).items():
            r = rules.get(rid)
            if r is None:
                warnings.append(f"rule_toggle_unknown:{rid}")
            elif r.hard_block and not enabled:
                warnings.append(f"policy_override_rejected:{rid}:mandatory rule cannot be disabled")
            else:
                rules[rid] = replace(r, enabled=bool(enabled))
        for rid, thr in (policy.confidence_overrides or {}).items():
            r = rules.get(rid)
            if r is None:
                warnings.append(f"confidence_override_unknown:{rid}")
            elif r.hard_block:
                warnings.append(f"policy_override_rejected:{rid}:mandatory rule threshold is fixed (fails closed)")
            else:
                rules[rid] = replace(r, min_confidence=min(1.0, max(0.0, float(thr))))
        if policy.faces:
            rules["FACE.REGION"] = replace(rules["FACE.REGION"], enabled=True)
        for i, spec in enumerate(policy.custom_rules or []):
            rid = spec.get("id") or f"CUSTOM.{i + 1:03d}"
            if not rid.startswith("CUSTOM."):
                rid = "CUSTOM." + rid
            try:
                actions = tuple(spec.get("allowed_actions", ("REDACT", "MASK")))
                rules[rid] = Rule(
                    id=rid, category=spec.get("category", "CUSTOM"), entity_type=spec.get("entity_type", "CUSTOM"),
                    label=spec.get("label", rid), pattern=spec["pattern"], group=int(spec.get("group", 0)),
                    flags=re.IGNORECASE if spec.get("ignore_case", True) else 0,
                    base_confidence=float(spec.get("confidence", 0.8)), severity=spec.get("severity", "HIGH"),
                    min_confidence=float(spec.get("min_confidence", 0.6)), allowed_actions=actions,
                    default_action=spec.get("default_action", actions[0] if "REDACT" not in actions else "REDACT"),
                    precedence=int(spec.get("precedence", 70)), formats=tuple(spec.get("formats", FORMATS)),
                    hard_block=False, description="Custom analyst pattern")
            except (RuleError, KeyError, TypeError, ValueError) as exc:
                warnings.append(f"custom_rule_rejected:{rid}:{exc}")
        if policy.denylist:
            rules["CUSTOM.DENYLIST"] = Rule(
                id="CUSTOM.DENYLIST", category="CUSTOM", entity_type="CUSTOM", label="Deny-listed literal",
                kind="literal", base_confidence=1.0, severity="HIGH", min_confidence=0.5,
                allowed_actions=("REDACT", "MASK"), default_action="REDACT", precedence=5, flags=re.IGNORECASE)
    return RuleRegistry(rules.values(), warnings=warnings)


# --------------------------------------------------------------------------- bounded evaluation
@dataclass
class RawMatch:
    start: int
    end: int
    confidence: float
    value: str = field(repr=False)


class EvaluationBudgetExceeded(RuntimeError):
    pass


def _chunks(text: str) -> Iterator[tuple[int, str]]:
    n = len(text)
    if n <= CHUNK:
        yield 0, text
        return
    pos = 0
    while pos < n:
        end = min(n, pos + CHUNK)
        yield pos, text[pos:end]
        if end == n:
            break
        pos = end - CHUNK_OVERLAP


def evaluate_rule(rule: Rule, text: str, *, deadline=None, literals: Iterable[str] = ()) -> list[RawMatch]:
    """Run one rule over one text unit with bounds. Returns validated matches (confidence set)."""
    if len(text) > MAX_TEXT_PER_UNIT:
        text = text[:MAX_TEXT_PER_UNIT]
    out: dict[tuple[int, int], RawMatch] = {}
    if rule.kind == "literal":
        low = text.lower()
        for lit in literals:
            lit_l = lit.lower()
            if not lit_l:
                continue
            i = low.find(lit_l)
            while i != -1 and len(out) < MAX_MATCHES_PER_RULE:
                out[(i, i + len(lit_l))] = RawMatch(i, i + len(lit_l), rule.base_confidence, text[i:i + len(lit_l)])
                i = low.find(lit_l, i + 1)
        return sorted(out.values(), key=lambda m: m.start)
    if rule.kind != "regex":
        return []
    flags = rule.flags
    use_regex = _regex is not None
    pat = _regex.compile(rule.pattern, flags) if use_regex else re.compile(rule.pattern, flags)
    validator = VALIDATORS.get(rule.validator) if rule.validator else None
    for offset, chunk in _chunks(text):
        if deadline is not None:
            deadline.check(f"rule {rule.id}")
        it = pat.finditer(chunk, timeout=0.75) if use_regex else pat.finditer(chunk)
        try:
            for m in it:
                g = rule.group
                if g and m.group(g) is None:
                    continue
                s, e = (m.start(g), m.end(g)) if g else (m.start(), m.end())
                if s == e:
                    continue
                val = chunk[s:e]
                conf = rule.base_confidence
                if validator is not None:
                    adj = validator(val)
                    if adj is None:
                        continue
                    conf = min(1.0, conf * adj)
                key = (offset + s, offset + e)
                if key not in out:
                    out[key] = RawMatch(offset + s, offset + e, conf, val)
                if len(out) >= MAX_MATCHES_PER_RULE:
                    break
        except TimeoutError as exc:  # regex-package timeout
            raise EvaluationBudgetExceeded(f"{rule.id} timed out") from exc
        if len(out) >= MAX_MATCHES_PER_RULE:
            break
    return sorted(out.values(), key=lambda m: (m.start, m.end))
