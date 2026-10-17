"""Search a vendor's own public pages, then grade what was found.

  python scripts/collect.py                 # dry run: one verdict per vendor, writes nothing
  python scripts/collect.py --vendor V-005  # one vendor
  python scripts/collect.py --save          # write the findings log and add rows to the evidence overlay
  python scripts/collect.py --no-search     # checklist paths only (no web search)

Saved rows are marked verified=False. A person must confirm them before relying on them.
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.demo import load
from src.osint import investigate, to_evidence_rows, audit_evidence
from src.config import ROOT
from src import store

ap = argparse.ArgumentParser()
ap.add_argument('--vendor')
ap.add_argument('--save', action='store_true')
ap.add_argument('--no-search', action='store_true')
args = ap.parse_args()

vendors, evidence = load()
log = {}
for v in vendors:
    if args.vendor and v['vendor_id'] != args.vendor:
        continue
    result = investigate(v, searcher=False if args.no_search else None)
    g = result['grade']
    log[v['vendor_id']] = {
        'vendor': v['vendor_name'],
        **{k: g[k] for k in g if k != 'ledger'},
        'pages': result['pages'],
        'ledger': result['ledger'],
        'signals_detail': result['signals'],
    }
    print(f"{v['vendor_id']}  {v['vendor_name'][:38]:38}  {g['verdict']:20}  {', '.join(g['providers']) or '-'}")
    if args.save:
        for row in to_evidence_rows(v, result['signals']):
            store.add_evidence(v['vendor_id'], row)

if args.save:
    (ROOT / 'ledger').mkdir(exist_ok=True)
    (ROOT / 'ledger' / 'osint_findings.json').write_text(json.dumps(log, indent=1))
    print('saved ledger/osint_findings.json and evidence overlay')

issues = audit_evidence(evidence)
print(f"\nStored-evidence audit: {len(issues)} row(s) need a person to check")
for i in issues:
    print(f"  {i['vendor_id']} -> {i['target']}: {'; '.join(i['problems'])}")
