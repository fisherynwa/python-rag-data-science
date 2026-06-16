"""
make_eval_holdout.py — build a held-out eval set from the SAME dataset the
fetcher indexes, but from rows the system never sees.

Why: writing eval questions by hand biases the benchmark toward questions you
know the corpus answers. Drawing them from a disjoint slice of the same dataset
removes that authoring bias — the questions are real, unseen, and same-distribution.

How it stays disjoint:
  1. Load + shuffle(seed) + filter EXACTLY like HuggingFaceFetcher.
  2. The fetcher indexes rows [0 : max_rows].
  3. This script takes rows [max_rows : max_rows + n_eval]  -> guaranteed unseen.

Output: data/eval.yaml  (question = instruction, ground_truth = output)

Run:
    python make_eval_holdout.py
"""

import yaml
from datasets import load_dataset

# ── Must match your fetcher config ────────────────────────────────────────────
DATASET   = "iamtarun/python_code_instructions_18k_alpaca"
SEED      = 42          # same seed the fetcher uses
KEYWORDS  = ["python", "pandas", "scikit-learn", "sklearn"]
MAX_ROWS  = 200         # how many rows the fetcher INDEXES (the indexed set)
N_EVAL    = 12          # how many held-out questions to generate
OUT_PATH  = "data/eval.yaml"

# ── Load + shuffle + filter exactly like the fetcher ──────────────────────────
ds = load_dataset(DATASET, split="train").shuffle(seed=SEED)

def keep(row):
    text = (row["instruction"] + row["output"]).lower()
    return bool(row["instruction"]) and bool(row["output"]) and any(k in text for k in KEYWORDS)

ds = ds.filter(keep)
print(f"Filtered pool: {len(ds)} rows")

# ── Take the slice AFTER the indexed rows -> never seen by the system ─────────
start = MAX_ROWS
end   = MAX_ROWS + N_EVAL
if end > len(ds):
    raise SystemExit(
        f"Not enough rows: pool has {len(ds)}, need {end}. "
        f"Lower MAX_ROWS or N_EVAL."
    )

heldout = ds.select(range(start, end))
print(f"Held-out eval rows: {start}..{end} ({len(heldout)} questions)")

# ── Build eval.yaml ───────────────────────────────────────────────────────────
questions = []
for row in heldout:
    questions.append({
        "question":     row["instruction"].strip(),
        "ground_truth": row["output"].strip(),
        "weight":       1.0,   # uniform; weighting is a future refinement
    })

with open(OUT_PATH, "w") as f:
    yaml.safe_dump({"questions": questions}, f, sort_keys=False, allow_unicode=True, width=1000)

print(f"\nWrote {len(questions)} held-out questions to {OUT_PATH}")
print("These rows are NOT in the indexed corpus — unbiased evaluation.")
