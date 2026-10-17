"""OSINT: search the vendor's own site, read pages, then grade claims.

Public pages only. Extra domains are dropped. Every claim keeps a verbatim
excerpt, URL, time, and a SHA-256 of the page text.

Grade on two axes (do not multiply them):
  reliability  1 = vendor legal/technical docs (vendor-written, not independent)
               2 = jobs / page code
               3 = homepage / blog / press
  specificity  provider named | capability | vague "AI"

Tags: Confirmed, Probable, Marketing, Needs review.
Also: Refuted, No evidence, Contradiction.
No public evidence is not proof of no AI.
"""
import hashlib
import html as htmlmod
import re
import urllib.request
import urllib.robotparser
from datetime import date
from urllib.parse import parse_qs, unquote, urljoin, urlparse, urlencode

TIER = {'subprocessor_list': 1, 'dpa': 1, 'privacy_policy': 1, 'trust_center': 1, 'api_docs': 1, 'changelog': 1,
        'job_posting': 2, 'js_source': 2, 'github': 2, 'marketplace': 2, 'case_study': 2,
        'homepage': 3, 'blog': 3, 'press': 3}
GRID = {1: {'provider': 'Confirmed', 'capability': 'Confirmed', 'vague': 'Probable'},
        2: {'provider': 'Probable', 'capability': 'Probable', 'vague': 'Marketing'},
        3: {'provider': 'Probable', 'capability': 'Marketing', 'vague': 'Marketing'}}
CHECKLIST = {'trust_center': ['/trust', '/security', '/trust-center', '/trust-centre'],
             'subprocessor_list': ['/subprocessors', '/sub-processors', '/legal/subprocessors', '/legal/sub-processors'],
             'privacy_policy': ['/privacy', '/privacy-policy', '/legal/privacy'],
             'dpa': ['/dpa', '/legal/dpa', '/data-processing'],
             'api_docs': ['/docs', '/developers'], 'changelog': ['/changelog', '/release-notes'],
             'job_posting': ['/careers', '/jobs'], 'homepage': ['/']}
RELIABILITY_LABEL = {
    1: 'Vendor legal or technical docs (the vendor wrote this)',
    2: 'Jobs or page code',
    3: 'Homepage, blog, or press',
}
SPEC_LABEL = {
    'provider': 'Names a provider or model',
    'capability': 'Names a capability (LLM, summarisation, and so on)',
    'vague': 'Vague "AI" wording',
}
PATH_TYPE = [
    (r'sub-?process', 'subprocessor_list'), (r'/dpa|data-processing', 'dpa'),
    (r'privacy', 'privacy_policy'), (r'trust|security', 'trust_center'),
    (r'changelog|release-notes', 'changelog'), (r'career|/jobs', 'job_posting'),
    (r'/docs|developer', 'api_docs'), (r'blog|news|press', 'blog'),
]
INTERESTING = re.compile(
    r'sub-?process|privacy|dpa|trust|security|changelog|developer|/docs|career|/jobs|'
    r'\bai\b|llm|openai|anthropic|machine-learning|machine_learning|third-part', re.I)
SKIP_HOST_BITS = (
    'linkedin.', 'twitter.', 'facebook.', 'youtube.', 'wikipedia.', 'crunchbase.',
    'github.', 'google.', 'g2.com', 'capterra.', 'reddit.', 'medium.com',
    'intercom.', 'drift.com', 'zendesk.', 'hotjar.', 'googletagmanager.',
)

PROVIDERS = {'Azure OpenAI': r'azure openai', 'OpenAI': r'\bopenai\b|\bchatgpt\b|\bgpt-?(?:3|4|5|4o)\b',
             'Anthropic': r'\banthropic\b|\bclaude\b', 'Google AI': r'vertex ai|\bgemini\b',
             'AWS Bedrock': r'\bbedrock\b', 'Cohere': r'\bcohere\b', 'Mistral': r'\bmistral\b',
             'Hugging Face': r'hugging ?face', 'Meta Llama': r'\bllama\b'}
ENDPOINTS = {'OpenAI': r'api\.openai\.com', 'Azure OpenAI': r'\.openai\.azure\.com',
             'Anthropic': r'api\.anthropic\.com', 'Google AI': r'generativelanguage\.googleapis\.com',
             'AWS Bedrock': r'bedrock-runtime\.'}
CAPABILITY = re.compile(r'\b(llm|large language model|machine learning|generative ai|genai|neural network|'
                        r'copilot|summari[sz]\w+|embedding\w*|fine-?tun\w+|retrieval-augmented|ml engineer)\b', re.I)
VAGUE = re.compile(r'\bAI\b|artificial intelligence|\bintelligent\b|ai-(?:powered|driven)', re.I)
NEGATION = re.compile(r"\b(do not|does not|don't|doesn't|never|no longer|not use|without)\b", re.I)
CUSTOMER_DATA = re.compile(r'customer data|your data|personal (?:data|information)|content you submit', re.I)
_ORDERED = ['Azure OpenAI'] + [p for p in PROVIDERS if p != 'Azure OpenAI']
NEEDS_REVIEW = 'Needs review'
UA = 'MeridianTPRM-OSINT/1.0'
LIMITS = [
    'A legal page is written by the vendor. It is accountable, not an independent check.',
    'Naming a provider shows that AI exists somewhere, not that this product sends your data to it.',
    'No public page is not proof the vendor does not use AI. The real product may sit behind a login.',
    'We only read the vendor’s own public site. Other domains are ignored.',
]


def vendor_domain(vendor):
    host = vendor['website'].replace('https://', '').replace('http://', '').split('/')[0].lower()
    return host.removeprefix('www.')


def vendor_origin(vendor):
    return 'https://' + vendor['website'].removeprefix('https://').removeprefix('http://').rstrip('/')


def same_site(url, domain):
    host = urlparse(url).netloc.lower().removeprefix('www.')
    d = domain.lower().removeprefix('www.')
    if any(bit in host for bit in SKIP_HOST_BITS):
        return False
    return host == d or host.endswith('.' + d)


def source_type_from_url(url):
    path = urlparse(url).path.lower()
    for pat, stype in PATH_TYPE:
        if re.search(pat, path):
            return stype
    return 'homepage'


def _text(raw):
    raw = re.sub(r'(?is)<(script|style).*?</\1>', ' ', raw)
    return re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', ' ', raw)).strip()


def _providers(sentence):
    found = [p for p in _ORDERED if re.search(PROVIDERS[p], sentence, re.I)]
    return [p for p in found if not (p == 'OpenAI' and 'Azure OpenAI' in found)]


def classify(raw, source_type, url, retrieved=None):
    """Page text/HTML -> list of tagged, traceable signals."""
    tier, retrieved = TIER[source_type], retrieved or date.today().isoformat()
    text = _text(raw)
    sha = hashlib.sha256(text.encode()).hexdigest()
    base = {'source_type': source_type, 'tier': tier, 'url': url, 'retrieved_at': retrieved, 'snapshot_sha256': sha,
            'reliability': RELIABILITY_LABEL[tier]}
    out = []
    for s in filter(None, (x.strip() for x in re.split(r'(?<=[.!?])\s+', text))):
        provs = _providers(s)
        spec = 'provider' if provs else 'capability' if CAPABILITY.search(s) else 'vague' if VAGUE.search(s) else None
        if not spec:
            continue
        tag = NEEDS_REVIEW if NEGATION.search(s) else GRID[tier][spec]
        flow = bool(provs) and (source_type in ('subprocessor_list', 'dpa') or bool(CUSTOMER_DATA.search(s)))
        out.append({**base, 'excerpt': s[:240], 'specificity': spec, 'specificity_label': SPEC_LABEL[spec],
                    'providers': provs, 'tag': tag, 'data_flow': flow,
                    'scope': 'customer_data' if flow else 'existence_only'})
    for p, pat in ENDPOINTS.items():
        if re.search(pat, raw):
            out.append({**base, 'source_type': 'js_source', 'tier': 2, 'reliability': RELIABILITY_LABEL[2],
                        'excerpt': f'Page source references {p} API endpoint',
                        'specificity': 'provider', 'specificity_label': SPEC_LABEL['provider'],
                        'providers': [p], 'tag': GRID[2]['provider'], 'data_flow': False, 'scope': 'existence_only'})
    return out


def _contradictions(signals):
    pos, neg = set(), set()
    for s in signals:
        names = set(s['providers'])
        if s['tag'] == NEEDS_REVIEW:
            neg |= names
        else:
            pos |= names
    return sorted(pos & neg)


def verdict(signals):
    """Roll signals up to one explainable vendor verdict."""
    usable = [s for s in signals if s['tag'] != NEEDS_REVIEW]
    confirmed = [s for s in usable if s['tag'] == 'Confirmed']
    probable = [s for s in usable if s['tag'] == 'Probable']
    provs = sorted({p for s in usable for p in s['providers']})
    review = sum(s['tag'] == NEEDS_REVIEW for s in signals)
    contra = _contradictions(signals)
    kinds = {s['source_type'] for s in probable}
    if contra:
        v, why = 'Contradiction', 'The same provider is both named and denied. A person must read the quotes.'
    elif confirmed:
        v, why = ('Confirmed AI use',
                  f"{len(confirmed)} signal(s) in the vendor's own legal or technical pages. "
                  'That is the vendor speaking, not an independent check.')
    elif len(kinds) >= 2:
        v, why = 'Likely AI use', 'Probable signals from 2+ different page types'
    elif usable:
        v, why = 'Unproven', 'Only marketing-grade or single-source evidence'
    else:
        v, why = 'No public evidence', 'Nothing found; this is NOT proof the vendor does not use AI'
    return {'verdict': v, 'why': why, 'providers': provs, 'signals': len(signals),
            'needs_review': review, 'data_flow_signals': sum(s['data_flow'] for s in usable),
            'contradictions': contra, 'corroboration': len(kinds),
            'limits': LIMITS}


def claim_ledger(signals):
    """Verbatim claims with two-axis tags and a corroboration count. Nothing is multiplied."""
    groups = {}
    for s in signals:
        key = (tuple(s['providers']), s['specificity'], s['tag'] == NEEDS_REVIEW)
        g = groups.setdefault(key, {'claims': [], 'source_types': set(), 'urls': set()})
        g['claims'].append(s)
        g['source_types'].add(s['source_type'])
        g['urls'].add(s['url'])
    rows = []
    for (provs, spec, negated), g in groups.items():
        best = g['claims'][0]
        for c in g['claims'][1:]:
            if c['tier'] < best['tier']:
                best = c
        rows.append({
            'claim': best['excerpt'],
            'providers': list(provs),
            'reliability': best['tier'],
            'reliability_label': best.get('reliability', RELIABILITY_LABEL[best['tier']]),
            'specificity': spec,
            'specificity_label': SPEC_LABEL[spec],
            'tag': NEEDS_REVIEW if negated else GRID[best['tier']][spec],
            'corroboration': len(g['source_types']),
            'urls': sorted(g['urls']),
            'url': best['url'],
            'retrieved_at': best['retrieved_at'],
            'snapshot_sha256': best['snapshot_sha256'],
            'source_type': best['source_type'],
            'scope': 'customer_data' if any(c.get('data_flow') for c in g['claims']) else 'existence_only',
            'count': len(g['claims']),
        })
    order = {NEEDS_REVIEW: 0, 'Confirmed': 1, 'Probable': 2, 'Marketing': 3}
    rows.sort(key=lambda r: (order.get(r['tag'], 9), r['reliability'], -r['corroboration']))
    return rows


def grade(signals, pages=None):
    """Human-facing grade: two axes, corroboration, coverage, limits."""
    v = verdict(signals)
    types_hit = {s['source_type'] for s in signals} | {p.get('source_type') for p in (pages or []) if p.get('ok')}
    wanted = set(CHECKLIST)
    coverage = round(len(types_hit & wanted) / len(wanted), 2)
    ledger = claim_ledger(signals)
    return {
        **v,
        'status': v['verdict'],
        'ledger': ledger,
        'coverage': coverage,
        'pages_ok': sum(1 for p in (pages or []) if p.get('ok')),
        'pages_tried': len(pages or []),
        'existence_only': sum(1 for r in ledger if r['scope'] == 'existence_only' and r['tag'] != NEEDS_REVIEW),
        'customer_data': sum(1 for r in ledger if r['scope'] == 'customer_data'),
    }


_TWIN_TAG = {'Confirmed': 'Confirmed', 'Marketing': 'Marketing claim only'}
_CONF = {'Confirmed': 'High', 'Probable': 'Medium', 'Marketing': 'Low'}


def to_evidence_rows(vendor, signals):
    """Signals -> rows the Digital Twin already understands. Probable maps to the twin's 'Technical proof'
    only for Tier-2 evidence, to 'Corroborated' when 2+ source types agree, else conservatively to Marketing."""
    vn, groups = vendor['vendor_name'], {}
    for s in (x for x in signals if x['tag'] != NEEDS_REVIEW):
        for t in s['providers'] or ['Internal AI service']:
            groups.setdefault(t, []).append(s)
    rows = []
    for target, sigs in groups.items():
        best = min(sigs, key=lambda s: ['Confirmed', 'Probable', 'Marketing'].index(s['tag']))
        kinds = {s['source_type'] for s in sigs if s['tag'] == 'Probable'}
        if best['tag'] == 'Probable':
            tag = ('Corroborated (2+ independent sources)' if len(kinds) >= 2
                   else 'Technical proof (docs/SDK/job posts)' if best['tier'] == 2 else 'Marketing claim only')
        else:
            tag = _TWIN_TAG[best['tag']]
        rows.append({'parent': vn, 'target': target, 'target_kind': 'ai' if target == 'Internal AI service' else 'ai_provider',
                     'tag': tag, 'osint_tag': best['tag'], 'source': best['source_type'], 'source_type': best['source_type'],
                     'url': best['url'], 'excerpt': best['excerpt'], 'age_days': 0, 'date': best['retrieved_at'],
                     'date_basis': 'Retrieved by collector', 'snapshot_sha256': best['snapshot_sha256'],
                     'claim_type': 'data_flow_proven' if any(s['data_flow'] for s in sigs) else 'ai_use_only',
                     'evidence_type': f"Tier {best['tier']} {best['source_type']}", 'confidence': _CONF[best['tag']],
                     'verified': False})
    return rows


def fetch(url, timeout=10, max_bytes=500_000):
    """Polite fetch: honours robots.txt, identifies itself, returns text or None. Never logs in."""
    try:
        u = urlparse(url)
        rp = urllib.robotparser.RobotFileParser(f'{u.scheme}://{u.netloc}/robots.txt')
        rp.read()
        if not rp.can_fetch(UA, url):
            return None
        with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': UA}), timeout=timeout) as r:
            if not r.headers.get_content_type().startswith('text/'):
                return None
            return r.read(max_bytes).decode('utf-8', 'ignore')
    except Exception:
        return None


def extract_links(raw, base_url, domain):
    """Same-site links that look like legal, trust, docs, or AI pages. Extra domains are dropped."""
    out, seen = [], set()
    for href in re.findall(r'''href=["']([^"'#]+)''', raw, re.I):
        href = htmlmod.unescape(href.split('?')[0].strip())
        if href.startswith(('mailto:', 'javascript:', 'tel:')):
            continue
        url = urljoin(base_url, href)
        if url in seen or not same_site(url, domain):
            continue
        if INTERESTING.search(url) or INTERESTING.search(href):
            seen.add(url)
            out.append(url)
    return out


def web_search(query, max_results=8, timeout=10):
    """Passive HTML search. Fail soft. Caller must still filter to the vendor domain."""
    urls = []
    try:
        q = urlencode({'q': query})
        req = urllib.request.Request(
            'https://lite.duckduckgo.com/lite/?' + q,
            headers={'User-Agent': UA},
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read(200_000).decode('utf-8', 'ignore')
        for m in re.finditer(r'href=["\']([^"\']+)["\']', body, re.I):
            href = htmlmod.unescape(m.group(1))
            if 'uddg=' in href:
                href = unquote(parse_qs(urlparse(href).query).get('uddg', [href])[0])
            if href.startswith('http') and 'duckduckgo.com' not in href:
                urls.append(href.split('?')[0])
            if len(urls) >= max_results:
                break
    except Exception:
        return []
    seen, out = set(), []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out[:max_results]


def search_queries(domain):
    return [
        f'site:{domain} (subprocessors OR "sub-processors")',
        f'site:{domain} ("trust center" OR "trust centre" OR DPA)',
        f'site:{domain} (OpenAI OR Anthropic OR "Azure OpenAI" OR LLM OR "machine learning")',
        f'site:{domain} (privacy AND (AI OR "artificial intelligence"))',
    ]


def discover_urls(vendor, fetcher=fetch, searcher=None, max_pages=12):
    """Find pages on this vendor's own site only: checklist + on-site links + optional web search."""
    domain, origin = vendor_domain(vendor), vendor_origin(vendor)
    ranked = []

    def add(url, stype, rank):
        if not same_site(url, domain):
            return
        cleaned = url.split('#')[0]
        parsed = urlparse(cleaned)
        if parsed.path in ('', '/'):
            cleaned = f'{parsed.scheme}://{parsed.netloc}/'
        else:
            cleaned = cleaned.rstrip('/')
        ranked.append((rank, cleaned, stype))

    for i, (stype, paths) in enumerate(CHECKLIST.items()):
        add(origin + paths[0], stype, i)
        for path in paths[1:]:
            add(origin + path, stype, 50 + i)

    home = origin + '/'
    raw_home = fetcher(home)
    if raw_home:
        for link in extract_links(raw_home, home, domain):
            add(link, source_type_from_url(link), 20)

    if searcher is not False:
        fn = web_search if searcher is None else searcher
        try:
            for q in search_queries(domain):
                hits = fn(q) or []
                for hit in hits:
                    add(hit, source_type_from_url(hit), 30)
        except Exception:
            pass

    seen, out = set(), []
    for rank, url, stype in sorted(ranked, key=lambda x: x[0]):
        key = url.lower().rstrip('/')
        if key in seen:
            continue
        seen.add(key)
        out.append((url, stype))
        if len(out) >= max_pages:
            break
    return out


def investigate(vendor, fetcher=fetch, searcher=None):
    """Search, fetch, read, then grade. `searcher=False` skips the web search (tests / checklist only)."""
    urls = discover_urls(vendor, fetcher=fetcher, searcher=searcher)
    pages, signals, seen = [], [], set()
    for url, stype in urls:
        raw = fetcher(url)
        page = {'url': url, 'source_type': stype, 'ok': bool(raw)}
        pages.append(page)
        if not raw:
            continue
        for s in classify(raw, stype, url):
            key = (s['url'], s['excerpt'])
            if key not in seen:
                seen.add(key)
                signals.append(s)
    g = grade(signals, pages)
    return {'pages': pages, 'signals': signals, 'ledger': g['ledger'], 'grade': g, 'verdict': g}


def collect(vendor, fetcher=fetch, searcher=False):
    """Run the per-vendor checklist (and optional search). `fetcher` is injectable so tests need no network."""
    return investigate(vendor, fetcher=fetcher, searcher=searcher)['signals']


def audit_evidence(evidence):
    """Flag stored evidence rows that would not survive an analyst's review."""
    issues = []
    for vid, rows in evidence.items():
        for r in rows:
            et = r.get('evidence_type', '')
            if 'Modeling assumption' in et:
                continue
            probs = []
            if not r.get('verified'):
                probs.append('unverified')
            if 'placeholder' in et.lower() or 'needs analyst' in r.get('source', '').lower():
                probs.append('placeholder, not real evidence')
            if r.get('tag') == 'Confirmed' and not et.startswith('Tier 1'):
                probs.append(f'tagged Confirmed but evidence type is "{et}", not a Tier-1 document')
            if not r.get('snapshot_sha256'):
                probs.append('no snapshot hash')
            if probs:
                issues.append({'vendor_id': vid, 'target': r['target'], 'problems': probs})
    return issues
