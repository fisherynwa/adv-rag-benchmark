"""Verify the answer key against the rendered PDF, then write the benchmark datasets.

Checks:
  1. The PDF has exactly 25 pages and each page starts with its intended section heading.
  2. Every evidence string appears on the page the answer key says (whitespace-normalised).
  3. Every superseded string appears on its stated page.
  4. Terms listed for unanswerable questions appear nowhere in the document.
  5. Internal arithmetic in the report (totals, growth rates, cash flow) adds up.
Outputs:
  answer_key.json                     every question with type, answer, evidence pages and strings
  dev_dataset.json / test_dataset.json LabelledRagDataset files (stratified by type, seed 42)
  verification_report.txt
"""
import json, random, re, sys
from collections import Counter, defaultdict
from pypdf import PdfReader

sys.path.insert(0, ".")
from questions_a import Q_A
from questions_b import Q_B
from build_pdf import PAGES

PDF = "kerrowan_annual_report_2025.pdf"
norm = lambda s: re.sub(r"\s+", " ", s).strip()

reader = PdfReader(PDF)
pages = {i + 1: norm(p.extract_text()) for i, p in enumerate(reader.pages)}
full_doc = " ".join(pages.values()).lower()
errors, warnings, log = [], [], []

# 1. page structure
if len(pages) != len(PAGES):
    errors.append(f"PDF has {len(pages)} pages, expected {len(PAGES)}")
for i, page in enumerate(PAGES, 1):
    heading = norm(page["blocks"][0][1])
    if heading not in pages.get(i, "")[:200]:
        errors.append(f"page {i}: heading '{heading}' not at top of page")

# 2-4. answer key
questions = Q_A + Q_B
for n, q in enumerate(questions, 1):
    q["id"] = f"Q{n:03d}"
    for page, text in q["ev"]:
        if norm(text) not in pages.get(page, ""):
            errors.append(f"{q['id']}: evidence not on page {page}: {text[:70]}")
        else:
            others = [p for p, t in pages.items() if p != page and norm(text) in t]
            if others:
                q.setdefault("also", set()).update(others)
                warnings.append(f"{q['id']}: evidence also appears on page(s) {others}: {text[:50]}")
    for page, text in q.get("superseded", []):
        if norm(text) not in pages.get(page, ""):
            errors.append(f"{q['id']}: superseded text not on page {page}: {text[:70]}")
    for term in q.get("absent", []):
        if term.lower() in full_doc:
            errors.append(f"{q['id']}: 'unanswerable' term found in document: {term}")
    if q["type"] != "unanswerable" and not q["ev"]:
        errors.append(f"{q['id']}: no evidence")

ids = Counter(q["q"] for q in questions)
errors += [f"duplicate question: {k}" for k, v in ids.items() if v > 1]

# 5. arithmetic in the report itself
def close(a, b, tol=0.051):
    return abs(a - b) <= tol
checks = {
    "quarters 2025 sum to 184.6": close(41.2 + 44.9 + 46.3 + 52.2, 184.6),
    "quarters 2024 sum to 162.3": close(37.5 + 39.8 + 41.6 + 43.4, 162.3),
    "segments sum to 184.6": close(96.4 + 61.3 + 26.9, 184.6),
    "segment profits ~27.1 (rounding)": abs(14.5 + 8.0 + 4.7 - 27.1) <= 0.15,
    "hardware + recurring = 184.6": close(146.2 + 38.4, 184.6),
    "price + volume = 22.3": close(7.1 + 15.2, 184.6 - 162.3),
    "growth 13.7%": close((184.6 / 162.3 - 1) * 100, 13.7),
    "margin 14.7%": close(27.1 / 184.6 * 100, 14.7),
    "EPS 32.6p": close(19.8 / 60.7 * 100, 32.6),
    "dividend 7.6 + 3.8 = 11.4": close(7.6 + 3.8, 11.4),
    "FCF 2025 = 42.9 - 7.1 + 0.8 - 14.9": close(42.9 - 7.1 + 0.8 - 14.9, 21.7),
    "FCF 2024 = 33.2 - 5.6 + 0.3 - 11.9": close(33.2 - 5.6 + 0.3 - 11.9, 16.0),
    "net cash change 2025": close(21.7 - 6.4 - 5.9, 9.4) and close(29.1 + 9.4, 38.5),
    "net cash change 2024": close(16.0 - 5.5 - 2.3, 8.2) and close(20.9 + 8.2, 29.1),
    "cash conversion 158%": close(42.9 / 27.1 * 100, 158.3, 0.1),
    "headcount table = 1,240": 512 + 186 + 214 + 153 + 71 + 104 == 1240,
    "final headcount = 1,240 + 22": 1240 + 22 == 1262,
    "regions sum to 100%": 22 + 39 + 21 + 14 + 4 == 100,
    "customer groups sum to 100%": 47 + 34 + 19 == 100,
    "scope 1+2 2025 / 2024": 1930 + 2940 == 4870 and 2180 + 3430 == 5610,
    "scope 1+2 change -13.2%": close((4870 / 5610 - 1) * 100, -13.2),
    "CEO pay = 1.42m": 520 + 468 + 432 == 1420,
    "revenue CAGR 11.7%": close(((184.6 / 118.4) ** 0.25 - 1) * 100, 11.7),
    "R&D 8.8% of revenue": close(16.2 / 184.6 * 100, 8.8),
}
errors += [f"arithmetic: {k}" for k, ok in checks.items() if not ok]

# token estimate
chars = sum(len(t) for t in pages.values())

# outputs
by_type = Counter(q["type"] for q in questions)
answer_key = [{"id": q["id"], "type": q["type"], "question": q["q"], "answer": q["a"],
               "evidence_pages": sorted({p for p, _ in q["ev"]}),
               "evidence": [{"page": p, "text": t} for p, t in q["ev"]],
               "also_on_pages": sorted(q.get("also", [])),
               "superseded": [{"page": p, "text": t} for p, t in q.get("superseded", [])],
               "absent_terms": q.get("absent", [])} for q in questions]

rng = random.Random(42)
groups = defaultdict(list)
for q in answer_key:
    groups[q["type"]].append(q)
dev, test = [], []
for t, qs in sorted(groups.items()):
    rng.shuffle(qs)
    k = round(len(qs) / 3)            # one third of each type goes to test
    test += qs[:k]
    dev += qs[k:]

try:
    from llama_index.core.llama_dataset import (LabelledRagDataset, LabelledRagDataExample,
                                                CreatedBy, CreatedByType)
    def to_ds(qs):
        by = CreatedBy(type=CreatedByType.AI, model_name="claude-opus-5-5 (verified against PDF)")
        return LabelledRagDataset(examples=[LabelledRagDataExample(
            query=q["question"], query_by=by, reference_answer=q["answer"], reference_answer_by=by,
            reference_contexts=[e["text"] for e in q["evidence"]]) for q in qs])
    to_ds(dev).save_json("dev_dataset.json")
    to_ds(test).save_json("test_dataset.json")
    ds_note = "LabelledRagDataset files written"
except ImportError:
    ds_note = "llama-index not installed here: LabelledRagDataset files NOT written"
json.dump({"dev": [q["id"] for q in dev], "test": [q["id"] for q in test]}, open("split.json", "w"), indent=1)
for q in dev:
    q["split"] = "dev"
for q in test:
    q["split"] = "test"
json.dump({"document": PDF, "pages": len(pages), "questions": answer_key},
          open("answer_key.json", "w"), indent=1, ensure_ascii=False)

report = [f"Pages: {len(pages)} | characters: {chars:,} (~{chars // 4:,} tokens)",
          f"Questions: {len(questions)} | " + ", ".join(f"{k}={v}" for k, v in sorted(by_type.items())),
          f"Split: dev={len(dev)} test={len(test)} (stratified by type, seed 42) | {ds_note}",
          f"Arithmetic checks: {sum(checks.values())}/{len(checks)} pass",
          f"ERRORS: {len(errors)}", *errors, f"NOTES (evidence also appears on another page; recorded as also_on_pages): {len(warnings)}", *warnings]
open("verification_report.txt", "w").write("\n".join(report) + "\n")
print("\n".join(report))
sys.exit(1 if errors else 0)
