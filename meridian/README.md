# Meridian

Search a vendor’s own public site, grade the AI claims, then decide how much data you can send.

```
Search public pages -> read quotes -> grade -> Digital Twin -> send or block -> ledger
```

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
python -m pytest -q
```

Use **Ledger → Demo data** in the sidebar if you want sample history.

## Pages
| Page | What it does |
|---|---|
| Overview | Chance of AI, budget left, send / block state, sortable fleet |
| Digital Twin | Who could receive the data, and the evidence |
| Document Gate | Strip fields, decide, save a receipt |
| Sanitisation Lab | Upload PDF / DOCX / PPTX / image, remove sensitive content, verify the output, price it in the Gate |
| Live OSINT Explorer | Responsible web collection, evidence review, two separate assessments, analyst-gated Twin update |
| Policy Lab | Advisory contextual bandit shown beside the authoritative rule-based decision |
| What-If | Try a change on a copy |
| Ledger | Tamper-evident log |
| Search | Original quick single-site search |

## Setup
```bash
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# OCR needs the Tesseract program:  macOS: brew install tesseract | Debian/Ubuntu: apt install tesseract-ocr | Windows: UB-Mannheim installer, add to PATH
pip install -r requirements-optional.txt            # optional
python scripts/train_rl.py                          # (re)creates data/rl_model.json and data/rl_report.json, deterministic
streamlit run app.py
python -m pytest -q                                 # no internet, credentials or paid APIs needed
```

## Document sanitisation (`src/sanitize`)
`process_document(input_path, output_path, policy, options)` returns a structured result: findings (regions with stable IDs, keyed hashes, no raw text),
per-payload outcomes (inspected / sanitised / preserved / unsupported / failed), verification checks and a release status
(`APPROVED_FOR_RELEASE`, `REVIEW_REQUIRED`, `BLOCKED`, `REJECTED_INPUT`, `FAILED`). A file is released only if the output was re-opened and independently re-checked;
otherwise it is quarantined as `<output>.quarantined`.
- PDF: native, scanned and mixed pages; true redaction (text and image pixels removed), never an overlay. DOCX/PPTX: runs, tables, headers/footers, notes, comments, hidden text, properties, alt text, embedded images (per occurrence, copy-on-write). Images: OCR + pixel painting, EXIF handling, optional local face regions (detection only, no identification).
- Coordinates: explicit transform chains between pixels, PDF points, crop, rotation, deskew and scale; nothing assumes pixel == point.
- Rules: versioned registry with stable IDs; mandatory rules (government IDs, card numbers, secrets) cannot be disabled, allow-listed or downgraded. Regex safety is a static screen plus bounded evaluation.
- Face detection backends are optional: set `MERIDIAN_FACE_CASCADE` (Haar cascade file) or `MERIDIAN_FACE_MODEL` (YuNet ONNX) and install `opencv-python-headless`.

## Live OSINT (`src/liveosint`)
Search provider (optional) is chosen by environment variables: `BRAVE_SEARCH_API_KEY`, or `GOOGLE_CSE_API_KEY` + `GOOGLE_CSE_CX`, or `SEARXNG_URL`
(`MERIDIAN_SEARCH_PROVIDER=brave|google_cse|searxng` to force one). With none set, Search mode says so and returns nothing; use Manual URLs or Offline fixtures.
Collection respects robots.txt and Crawl-delay, rate-limits per domain, retries a bounded number of times, caps size and content type, and blocks
non-public addresses (also on every redirect). Evidence keeps URL, time, hashes and a verbatim excerpt. AI adoption and Meridian exposure are separate
assessments; only analyst-**accepted** evidence changes the Twin / Leak Budget (idempotent, audited, before/after shown). Offline-fixture evidence can never be accepted.

## Advisory RL (`src/rl`)
Offline, synthetic contextual bandit that can only keep or tighten the rule-based redaction plan. Hard rules are an action mask, not reward penalties.
`python scripts/train_rl.py` trains on seeds disjoint from the evaluation seeds and prints every baseline. It is advisory: the gate's decision is unchanged.

## Search
Public pages only. Extra domains are dropped. Chat widgets and job boards are ignored.

1. Find pages (trust, privacy, sub-processors, DPA, docs, careers, plus on-site links).
2. Read each page and keep the quote, URL, time, and a hash.
3. Grade on two axes: how reliable the page is, and how specific the claim is. We do not multiply those.
4. A person still has to check the quotes. No public page is not proof of no AI.

```bash
python scripts/collect.py
python scripts/collect.py --save
python scripts/collect.py --no-search
```

## Model
- Evidence → base p: Marketing 0.10 · Corroborated 0.40 · Technical proof 0.70 · Confirmed 0.95
- Chance of AI (P_ai) is an estimate, not a measured fact
- Budget = BASE × (1 − P_ai) × criticality × history
- A leftover government ID always blocks the send
