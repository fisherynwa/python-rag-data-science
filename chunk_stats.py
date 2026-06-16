"""
chunk_stats.py — character-length distribution of fetched documents.

Run from the project root:
    python3 chunk_stats.py
    # or: uv run python chunk_stats.py

Prints stats to the terminal and saves a boxplot to chunk_dist.png
(no display window needed — works headless in WSL).
"""

import numpy as np
from datasets import load_dataset

DATASET  = "iamtarun/python_code_instructions_18k_alpaca"
KEYWORDS = ["python", "pandas", "scikit-learn", "sklearn"]
MAX_ROWS = 500

# ── Load + filter exactly like the fetcher ────────────────────────────────────
ds = load_dataset(DATASET, split="train").shuffle(seed=42)

def keep(row):
    text = (row["instruction"] + row["output"]).lower()
    return bool(row["instruction"]) and bool(row["output"]) and any(k in text for k in KEYWORDS)

ds = ds.filter(keep)
ds = ds.select(range(min(MAX_ROWS, len(ds))))
print(f"\nFiltered rows: {len(ds)}\n")

# ── Build bodies like the fetcher, count characters ───────────────────────────
def build_body(row):
    instruction = row["instruction"].strip()
    output      = row["output"].strip()
    context     = (row.get("input") or "").strip()
    body = f"Question:\n{instruction}"
    if context:
        body += f"\n\nContext:\n{context}"
    body += f"\n\nAnswer:\n{output}"
    return body

lengths = np.array([len(build_body(r)) for r in ds])

# ── Stats ─────────────────────────────────────────────────────────────────────
print(f"min     : {lengths.min()}")
print(f"max     : {lengths.max()}")
print(f"mean    : {lengths.mean():.0f}")
print(f"median  : {np.median(lengths):.0f}")
print()
for p in [25, 50, 75, 90, 95]:
    print(f"{p}th percentile: {np.percentile(lengths, p):.0f}")
print()
for size in [128, 256, 512, 1024]:
    pct = (lengths <= size).mean() * 100
    print(f"docs that fit whole in chunk_size={size}: {pct:.0f}%")
print()

# ── Save boxplot (no display window — headless safe) ──────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")  # no GUI needed
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.boxplot(lengths, showfliers=True)
    ax1.set_title("Document length (chars) — boxplot")
    ax1.set_ylabel("characters")
    for s in [256, 512]:
        ax1.axhline(s, ls="--", alpha=0.6, label=f"chunk_size={s}")
    ax1.legend()

    ax2.hist(lengths, bins=40, edgecolor="black", alpha=0.8)
    ax2.set_title("Document length distribution")
    ax2.set_xlabel("characters"); ax2.set_ylabel("docs")
    for s in [256, 512]:
        ax2.axvline(s, ls="--", alpha=0.6, label=f"chunk_size={s}")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("chunk_dist.png", dpi=120)
    print("Saved boxplot to chunk_dist.png")
except Exception as e:
    print(f"(plot skipped: {e})")
