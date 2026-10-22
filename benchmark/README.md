# Kerrowan RAG benchmark

A fictional 25-page company annual report with an answer key that says exactly which page each answer is on.
Everything about Kerrowan Marine Systems plc is invented, so no model can answer from memory.

## Files

| File | What it is |
|---|---|
| `kerrowan_annual_report_2025.pdf` | The benchmark document (25 pages, ~15k tokens) |
| `answer_key.json` | 154 questions: type, reference answer, evidence page(s) and exact evidence text, superseded text, absent terms |
| `dev_dataset.json`, `test_dataset.json` | The same questions as LlamaIndex `LabelledRagDataset` files (103 dev / 51 test, stratified by type) |
| `split.json` | Which question IDs are in dev and test |
| `verification_report.txt` | Result of the last verification run |
| `facts.json` | The world bible: every key fact and number |
| `content_*.py`, `build_pdf.py` | The report text and the PDF renderer |
| `questions_a.py`, `questions_b.py` | The questions, as written |
| `verify_and_build.py` | Re-verifies everything and rebuilds the answer key and datasets |

## Question types

| Type | Count | Tests |
|---|---|---|
| single | 52 | One fact, one page |
| table | 16 | Answer only in a table |
| near_duplicate | 18 | T3 vs T3-S, Kestrel 400 vs 400X, Port Lennox vs Port Lenham, 2024 vs 2025 |
| multihop_section | 20 | Two facts on the same page, often with arithmetic |
| cross_section | 20 | Facts on pages far apart |
| summary | 12 | A section-level answer |
| superseded | 6 | A figure corrected on page 22; the earlier figure is a trap |
| unanswerable | 10 | Not in the report; the correct answer is that it doesn't say |

## Guarantees (checked by `verify_and_build.py`)

- The PDF has exactly 25 pages, and each page starts with its intended section.
- Every evidence string appears, word for word, on the page the answer key gives. Two answers also appear on a second page; those pages are listed in `also_on_pages`.
- Every superseded figure appears on its stated page, so the trap is real.
- Key terms of the unanswerable questions appear nowhere in the document.
- 24 arithmetic checks on the report's own numbers pass (totals, growth rates, cash flow, headcount).

## Rebuild

```
pip install reportlab pypdf llama-index-core==0.14.20
python build_pdf.py
python verify_and_build.py      # exits non-zero if anything fails
```

## Scoring retrieval by page

LlamaIndex's PDF reader (`llama-index-readers-file`, used by `SimpleDirectoryReader`) stores each page number in a node's `page_label` metadata. A retrieved node is a hit
when its page is in `evidence_pages` (or `also_on_pages`); this works for any chunk size.
