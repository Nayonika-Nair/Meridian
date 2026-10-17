"""Evidence extraction with provenance. Page content is untrusted text: parsed with BeautifulSoup, never executed.

Evidence classes (strongest last). Only a human can create the last two, by citing an authorised observation:
  marketing_claim                 AI wording on a marketing page / vague wording
  technical_indication            provider/capability/endpoint named in docs, jobs, code
  corroborated_claim              the same AI use asserted by 2+ distinct domains, at least one independent of the vendor
  explicit_processing_disclosure  vendor legal/trust text ties AI processing to customer data or lists the AI provider as a sub-processor
  observed_flow / verified_flow   analyst-supplied observation of Meridian data reaching the AI (never auto-generated)
AI use alone never reaches the flow classes.
"""
from __future__ import annotations

import hashlib
import re
import time
from dataclasses import asdict, dataclass, field
from urllib.parse import urldefrag, urlsplit

from bs4 import BeautifulSoup, Comment

EXTRACTOR_VERSION = "2026.10.1"
CLASSES = ["marketing_claim", "technical_indication", "corroborated_claim", "explicit_processing_disclosure", "observed_flow", "verified_flow"]
AUTO_CLASSES = CLASSES[:4]
FLOW_CLASSES = ("observed_flow", "verified_flow")

PROVIDERS = {"Azure OpenAI": r"azure openai", "OpenAI": r"\bopenai\b|\bchatgpt\b|\bgpt-?(?:3|4|5|4o)\b",
             "Anthropic": r"\banthropic\b|\bclaude\b", "Google AI": r"vertex ai|\bgemini\b",
             "AWS Bedrock": r"\bbedrock\b", "Cohere": r"\bcohere\b", "Mistral": r"\bmistral\b",
             "Hugging Face": r"hugging ?face", "Meta Llama": r"\bllama\b"}
ENDPOINTS = {"OpenAI": r"api\.openai\.com", "Azure OpenAI": r"\.openai\.azure\.com", "Anthropic": r"api\.anthropic\.com",
             "Google AI": r"generativelanguage\.googleapis\.com", "AWS Bedrock": r"bedrock-runtime\."}
CAPABILITY = re.compile(r"\b(llm|large language model|machine learning|generative ai|genai|neural network|copilot|summari[sz]\w+|embedding\w*|"
                        r"fine-?tun\w+|retrieval-augmented|ml engineer)\b", re.I)
VAGUE = re.compile(r"\bAI\b|artificial intelligence|ai-(?:powered|driven)", re.I)
NEGATION = re.compile(r"\b(do not|does not|don't|doesn't|never|no longer|not use[sd]?|without|will not|won't|not train)\b", re.I)
CUSTOMER_DATA = re.compile(r"customer data|your data|personal (?:data|information)|content you submit|customer content", re.I)
PATH_TYPE = [(r"sub-?process", "subprocessor_list"), (r"/dpa|data-processing", "dpa"), (r"privacy", "privacy_policy"),
             (r"trust|security", "trust_center"), (r"changelog|release-notes", "changelog"), (r"career|/jobs", "job_posting"),
             (r"/docs|developer", "api_docs"), (r"blog|news|press", "blog")]
TIER = {"subprocessor_list": 1, "dpa": 1, "privacy_policy": 1, "trust_center": 1, "api_docs": 1, "changelog": 1,
        "job_posting": 2, "js_source": 2, "homepage": 3, "blog": 3, "third_party": 3}
LEGAL = ("subprocessor_list", "dpa", "privacy_policy", "trust_center")
MAX_EXCERPT, MAX_SENTENCES = 400, 3000
_CTRL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f‪-‮⁦-⁩]")      # control + bidi override characters


def registrable(host: str) -> str:
    h = host.lower().removeprefix("www.")
    parts = h.split(".")
    if len(parts) >= 3 and parts[-2] in ("co", "com", "org", "net", "gov", "ac") and len(parts[-1]) == 2:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:]) if len(parts) >= 2 else h


def vendor_domain(vendor: dict) -> str:
    host = re.sub(r"^https?://", "", vendor["website"].strip().lower()).split("/")[0]
    return registrable(host)


def source_type_for(url: str, independent: bool) -> str:
    if independent:
        return "third_party"
    path = urlsplit(url).path.lower()
    for pat, st in PATH_TYPE:
        if re.search(pat, path):
            return st
    return "homepage"


def clean(s: str) -> str:
    return re.sub(r"\s+", " ", _CTRL.sub(" ", s)).strip()


def parse_html(raw: str):
    """-> (title, visible_text_blocks, script_blob). Scripts/styles/comments/hidden inputs never leak into text."""
    soup = BeautifulSoup(raw, "lxml")
    scripts = " ".join([s.get("src", "") for s in soup.find_all("script")] + [s.get_text(" ") for s in soup.find_all("script")][:20])
    for t in soup(["script", "style", "noscript", "template", "iframe", "svg"]):
        t.decompose()
    for c in soup.find_all(string=lambda x: isinstance(x, Comment)):
        c.extract()
    title = clean(soup.title.get_text(" ")) if soup.title else ""
    blocks = [clean(x) for x in soup.get_text("\n").split("\n")]
    return title, [b for b in blocks if b], scripts


def split_sentences(blocks: list[str]) -> list[str]:
    out = []
    for b in blocks:
        out += [s.strip() for s in re.split(r"(?<=[.!?])\s+", b) if s.strip()]
        if len(out) > MAX_SENTENCES:
            break
    return out[:MAX_SENTENCES]


def providers_in(s: str) -> list[str]:
    found = [p for p in PROVIDERS if re.search(PROVIDERS[p], s, re.I)]
    return [p for p in found if not (p == "OpenAI" and "Azure OpenAI" in found)]


def normalise_for_id(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def make_evidence_id(vendor_id: str, url: str, excerpt: str) -> str:
    return "EV-" + hashlib.sha256(f"{vendor_id}|{urldefrag(url)[0]}|{normalise_for_id(excerpt)}".encode()).hexdigest()[:20]


@dataclass
class Candidate:
    evidence_id: str
    vendor_id: str
    url: str
    final_url: str
    domain: str
    independent: bool
    category: str
    source_type: str
    reliability_tier: int
    specificity: str                    # provider | capability | vague | endpoint
    providers: list
    excerpt: str
    negated: bool
    customer_data: bool
    evidence_class: str
    page_sha256: str                    # SHA-256 of the extracted visible text of the page
    excerpt_sha256: str
    retrieved_at: float
    origin: str                         # live | cache | fixture | manual
    http_status: int | None
    robots: str
    extractor_version: str = EXTRACTOR_VERSION
    also_seen_at: list = field(default_factory=list)
    stale: bool = False
    age_s: float = 0.0

    def to_dict(self):
        return asdict(self)


def extract(vendor: dict, fetch, category: str, *, independent: bool | None = None, origin: str | None = None, now: float | None = None) -> list[Candidate]:
    """FetchResult -> candidates. Never fabricates: nothing is returned for pages without AI wording."""
    if not fetch.ok:
        return []
    final = fetch.final_url or fetch.url
    host = urlsplit(final).hostname or ""
    vdom = vendor_domain(vendor)
    if independent is None:
        independent = registrable(host) != vdom and not fetch.url.startswith("fixture://")
    if fetch.url.startswith("fixture://"):
        independent = bool(independent)
    stype = source_type_for(final, independent) if not final.startswith("fixture://") else source_type_for("/" + final.split("/", 3)[-1], independent)
    tier = TIER.get(stype, 3)
    text_like = fetch.content_type in ("text/html", "application/xhtml+xml") or "<html" in fetch.text[:2000].lower() or "<body" in fetch.text[:2000].lower()
    if text_like:
        title, blocks, scripts = parse_html(fetch.text)
    else:
        title, blocks, scripts = "", [clean(x) for x in fetch.text.splitlines() if x.strip()], ""
    page_sha = hashlib.sha256("\n".join(blocks).encode()).hexdigest()
    now = time.time() if now is None else now
    org = origin or fetch.origin or "live"
    out = []

    def mk(excerpt, spec, provs, negated, cust, cls, st, t):
        excerpt = excerpt[:MAX_EXCERPT]
        out.append(Candidate(make_evidence_id(vendor["vendor_id"], final, excerpt), vendor["vendor_id"], fetch.url, final, registrable(host) or "fixture",
                             bool(independent), category, st, t, spec, provs, excerpt, negated, cust, cls, page_sha,
                             hashlib.sha256(normalise_for_id(excerpt).encode()).hexdigest(), fetch.fetched_at or now, org,
                             fetch.status, fetch.robots, age_s=getattr(fetch, "age_s", 0.0), stale=getattr(fetch, "stale", False)))

    for s in split_sentences(blocks):
        provs = providers_in(s)
        spec = "provider" if provs else "capability" if CAPABILITY.search(s) else "vague" if VAGUE.search(s) else None
        if not spec:
            continue
        neg, cust = bool(NEGATION.search(s)), bool(CUSTOMER_DATA.search(s))
        if neg:
            cls = "marketing_claim"                                         # negative statements are recorded but never scored as adoption
        elif stype in LEGAL and ((cust and spec != "vague") or (stype in ("subprocessor_list", "dpa") and provs)):
            cls = "explicit_processing_disclosure"
        elif tier <= 2 and spec in ("provider", "capability"):
            cls = "technical_indication"
        else:
            cls = "marketing_claim"
        mk(s, spec, provs, neg, cust, cls, stype, tier)
    for p, pat in ENDPOINTS.items():
        m = re.search(pat, scripts + " " + fetch.text)
        if m:
            mk(f"Page source references the {p} API endpoint ({m.group(0)}).", "endpoint", [p], False, False, "technical_indication", "js_source", 2)
    return out


def dedupe(cands: list[Candidate]) -> list[Candidate]:
    """Collapse identical evidence_ids and boilerplate repeated across URLs (same normalised excerpt): keep the most
    reliable occurrence, remember where else it appeared."""
    best: dict[tuple, Candidate] = {}
    for c in cands:
        key = (c.vendor_id, c.excerpt_sha256)
        cur = best.get(key)
        if cur is None:
            best[key] = c
            continue
        keep, drop = (c, cur) if (c.reliability_tier, c.url) < (cur.reliability_tier, cur.url) else (cur, c)
        keep.also_seen_at = sorted(set(keep.also_seen_at + drop.also_seen_at + ([drop.final_url] if drop.final_url != keep.final_url else [])))
        best[key] = keep
    return list(best.values())


def corroborate(cands: list[Candidate]) -> list[Candidate]:
    """Promote marketing/technical items to corroborated_claim when 2+ distinct domains (>=1 independent of the vendor)
    assert the same AI use (same named provider; or, without a provider, the same specificity). Fixtures never corroborate."""
    groups: dict[tuple, list[Candidate]] = {}
    for c in cands:
        if c.negated or c.origin == "fixture":
            continue
        key = (c.vendor_id, c.providers[0] if c.providers else "(unnamed AI)")
        groups.setdefault(key, []).append(c)
    for items in groups.values():
        doms = {i.domain for i in items}
        if len(doms) >= 2 and any(i.independent for i in items):
            for i in items:
                if i.evidence_class in ("marketing_claim", "technical_indication"):
                    i.evidence_class = "corroborated_claim"
    return cands
