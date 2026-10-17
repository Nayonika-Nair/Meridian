from src.osint import (classify, verdict, to_evidence_rows, collect, audit_evidence, NEEDS_REVIEW,
                       same_site, extract_links, discover_urls, investigate, claim_ledger, grade)
from src.demo import load, vendor_assessment

V = {'vendor_id': 'X-1', 'vendor_name': 'ExampleVendor', 'website': 'www.example.test', 'criticality': 'High', 'criticality_factor': 0.75}
SUB = '<p>Our sub-processors include Azure OpenAI for text summarization of support tickets.</p>'
HYPE = '<h1>Our AI-powered platform will transform your business.</h1>'
JOB = '<p>We are hiring an ML engineer to build LLM features.</p>'
JS = '<script>fetch("https://api.openai.com/v1/chat/completions")</script><p>Welcome</p>'
HOME = '<a href="/privacy">Privacy</a> <a href="/subprocessors">Subs</a> <a href="https://other.com/ai">skip</a> <p>Welcome</p>'


def test_subprocessor_naming_provider_is_confirmed_and_data_flow():
    s = classify(SUB, 'subprocessor_list', 'https://example.test/subprocessors')
    assert s[0]['tag'] == 'Confirmed' and s[0]['providers'] == ['Azure OpenAI'] and s[0]['data_flow']
    assert verdict(s)['verdict'] == 'Confirmed AI use'


def test_marketing_hype_alone_is_never_confirmed():
    s = classify(HYPE, 'homepage', 'https://example.test/')
    assert {x['tag'] for x in s} == {'Marketing'} and verdict(s)['verdict'] == 'Unproven'


def test_two_independent_probable_sources_make_likely():
    s = classify(JOB, 'job_posting', 'u1') + classify(JS, 'homepage', 'u2')
    assert verdict(s)['verdict'] == 'Likely AI use'


def test_single_job_post_is_only_unproven():
    assert verdict(classify(JOB, 'job_posting', 'u'))['verdict'] == 'Unproven'


def test_negated_sentence_goes_to_review_not_confirmed():
    s = classify('We do not use OpenAI to train on customer data.', 'privacy_policy', 'u')
    assert s[0]['tag'] == NEEDS_REVIEW and verdict(s)['verdict'] == 'No public evidence'


def test_empty_is_no_public_evidence_not_no_ai():
    v = verdict([]); assert v['verdict'] == 'No public evidence' and 'NOT proof' in v['why']


def test_azure_openai_not_double_counted_as_openai():
    assert classify(SUB, 'dpa', 'u')[0]['providers'] == ['Azure OpenAI']


def test_traceability_fields_present_and_hash_stable():
    a = classify(SUB, 'dpa', 'u', '2026-10-12')[0]; b = classify(SUB, 'dpa', 'u', '2026-10-12')[0]
    assert a['url'] == 'u' and a['retrieved_at'] == '2026-10-12' and a['snapshot_sha256'] == b['snapshot_sha256']


def test_collect_uses_injected_fetcher_and_dedupes():
    pages = {'https://www.example.test/subprocessors': SUB, 'https://www.example.test/': HYPE}
    s = collect(V, fetcher=pages.get)
    assert verdict(s)['verdict'] == 'Confirmed AI use' and len(s) == len({(x['url'], x['excerpt']) for x in s})


def test_rows_feed_the_existing_twin():
    rows = to_evidence_rows(V, classify(SUB, 'subprocessor_list', 'u'))
    assert rows[0]['tag'] == 'Confirmed' and rows[0]['claim_type'] == 'data_flow_proven' and rows[0]['verified'] is False
    assert vendor_assessment(V, rows, 1000, 3000)['p_ai'] > 0.9


def test_hype_rows_stay_low():
    rows = to_evidence_rows(V, classify(HYPE, 'homepage', 'u'))
    assert vendor_assessment(V, rows, 1000, 3000)['p_ai'] < 0.2


def test_audit_flags_shipped_placeholders_and_non_tier1_confirmed():
    _, ev = load(); issues = audit_evidence(ev)
    assert any('placeholder' in p for i in issues for p in i['problems'])
    bny = [i for i in issues if i['vendor_id'] == 'V-005' and 'Confirmed' in ' '.join(i['problems'])]
    assert bny


def test_extra_domains_are_dropped():
    assert same_site('https://www.example.test/privacy', 'example.test')
    assert not same_site('https://other.com/ai', 'example.test')
    assert not same_site('https://linkedin.com/company/example', 'example.test')
    links = extract_links(HOME, 'https://www.example.test/', 'example.test')
    assert any('subprocessors' in u or 'privacy' in u for u in links)
    assert not any('other.com' in u for u in links)


def test_search_then_grade_ignores_offsite_hits():
    pages = {
        'https://www.example.test': HOME,
        'https://www.example.test/': HOME,
        'https://www.example.test/subprocessors': SUB,
        'https://www.example.test/privacy': 'We do not use Azure OpenAI.',
    }

    def searcher(_q):
        return ['https://www.example.test/subprocessors', 'https://news.example.net/ai', 'https://linkedin.com/x']

    found = discover_urls(V, fetcher=pages.get, searcher=searcher)
    urls = [u for u, _ in found]
    assert any('subprocessors' in u for u in urls)
    assert not any('news.example.net' in u or 'linkedin.com' in u for u in urls)
    result = investigate(V, fetcher=pages.get, searcher=searcher)
    g = result['grade']
    assert g['verdict'] == 'Contradiction'
    assert 'Azure OpenAI' in g['providers']
    assert result['ledger']
    row = result['ledger'][0]
    assert 'reliability' in row and 'specificity' in row and 'corroboration' in row
    assert 'score' not in row
    assert g['coverage'] > 0


def test_claim_ledger_keeps_two_axes():
    s = classify(SUB, 'subprocessor_list', 'u') + classify(HYPE, 'homepage', 'u2')
    rows = claim_ledger(s)
    assert rows[0]['tag'] == 'Confirmed' and rows[0]['reliability'] == 1
    assert rows[0]['specificity'] == 'provider' and rows[0]['scope'] == 'customer_data'
    g = grade(s)
    assert g['customer_data'] >= 1 and g['existence_only'] >= 1
