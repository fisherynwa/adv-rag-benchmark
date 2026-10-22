### `results/results.csv`

One row per answer: 6 systems × 154 questions = 924 rows.

| Column | Meaning | Values |
|---|---|---|
| `system` | Which RAG system answered | e.g. `sentence window 5`, `full context` |
| `id` | Question ID in `benchmark/answer_key.json` | `Q001`–`Q154` |
| `type` | Question type | `single`, `table`, `near_duplicate`, `multihop_section`, `cross_section`, `summary`, `superseded`, `unanswerable` |
| `answer` | The model's answer | text |
| `found` | Share of the evidence sentences that appear word for word in the text the model read | 0–1; empty for unanswerable questions |
| `context_tokens` | Size of the text the model read (characters ÷ 4) | number |
| `seconds` | Time to retrieve and answer, excluding judging | number |
| `correct` | LLM judge: answer matches the reference answer (score ≥ 4 of 5) | 1 / 0; empty if the reply was unreadable |
| `grounded` | LLM judge: answer is supported by the retrieved text | 1 / 0; empty for full context |
| `answer_relevance` | RAG triad: answer addresses the question | 0–1; only sentence window 3 and auto-merging |
| `context_relevance` | RAG triad: retrieved text is relevant to the question | 0–1; only sentence window 3 and auto-merging |  