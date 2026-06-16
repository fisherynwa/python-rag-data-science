# Python Data Extraction RAG
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.12%2B-blue?logo=python&logoColor=white)](https://www.python.org)
[![Topic: RAG](https://img.shields.io/badge/Topic-RAG%20%2F%20AI-orange)](https://en.wikipedia.org/wiki/Retrieval-augmented_generation)
[![GitHub issues](https://img.shields.io/github/issues/fisherynwa/python-rag-datascience?color=red)](https://github.com/fisherynwa/python-rag-datascience/issues)
[![GitHub forks](https://img.shields.io/github/forks/fisherynwa/python-rag-datascience?style=social)](https://github.com/fisherynwa/python-rag-datascience/network/members)


A retrieval-augmented generation (RAG) system that answers Python data-extraction
questions (pandas, CSV/JSON, scikit-learn) by grounding a local LLM in a corpus of
Python Q&A pairs. Fully local (using llama) and free no paid API keys required.

---

## What it does

Given a question like *"How do I read a CSV with missing values in pandas?"*, the
system retrieves the most relevant Q&A chunks from a vector store and passes them to
a local LLM, which generates an answer grounded in that retrieved context rather than
from the model's parametric memory alone.

```
HuggingFace dataset
   → fetch + filter      (Hugging Face Fetcher)
   → chunk               (Document Chunker)
   → embed + index       (Weaviate Indexer, hybrid search)
   → retrieve + generate (RAGPipeline → local LLM via Ollama)
   → evaluate            (RAGASEvaluator → MLflow)
```

---

## Stack

| Layer        | Choice                                          | Why |
|--------------|-------------------------------------------------|-----|
| Data         | `iamtarun/python_code_instructions_18k_alpaca`  | Free, static, Python-focused |
| Embeddings   | `all-MiniLM-L6-v2` (384-dim)                     | Fast, normalized, runs on CPU |
| Vector store | Weaviate (Docker)                                | Native hybrid search, metadata |
| Retrieval    | Hybrid (vector + BM25), `alpha=0.75`             | Code Q&A mixes semantic intent with exact terms (`alpha` is tunable in general) |
| LLM          | Qwen2.5 1.5B / Llama 3.2 1B via Ollama (local)   | Free, on-device, no API key |
| Evaluation   | RAGAS (local Ollama judge)                       | Answer relevancy + context precision |
| Tracking     | MLflow (Docker)                                  | Logs metrics + params per run |
| Config       | Hydra                                            | Every component swappable from the CLI |

---

## How to run

```bash
# 1. Start services
docker compose up -d            # Weaviate (8080 + 50051 gRPC) + MLflow (5000)

# 2. Pull models (once)
ollama pull qwen2.5:1.5b        # generation
ollama pull llama3.2:1b         # RAGAS judge
ollama pull nomic-embed-text    # RAGAS judge embeddings

# 3. Run
uv run python run.py                          # defaults
uv run python run.py fetcher.max_rows=200     # bigger corpus
uv run python run.py chunker.chunk_size=1024 chunker.overlap=128
```

Results print as a scorecard and log to MLflow at `http://localhost:5000` (run names
encode the config, e.g. `chunk512_ov64_a0.75`).

---

## Configuration (Hydra)

All settings live in `configs/`. Override any value from the command line:

```bash
uv run python run.py fetcher.max_rows=200 pipeline.top_k=5 top_k=5
```

Config groups: `fetcher`, `chunker`, `indexer`, `pipeline`, `evaluator`.

---

## Chunk size: a data-driven choice

Chunk size controls how documents are split before embedding. Rather than guessing,
the choice is grounded (and tested) in the actual length distribution of the corpus (see
`chunk_stats.py`). Measuring the character length of every fetched document gives the
share that fit *whole* inside each candidate chunk size:

| chunk_size | docs that fit whole |
|-----------:|--------------------:|
| 128        | 1%                  |
| 256        | 22%                 |
| **512**    | **69%**             |
| 1024       | 91%                 |

Why 512. At 256, 78% of documents would be split, for code answers, that often
means cutting a code block in half, so the model retrieves a fragment without the rest.
At 512, two-thirds of documents stay intact while only the longest are split. Smaller
chunks give sharper embeddings (more precise retrieval) but split more documents;
larger chunks keep answers whole but blur the embedding. 512 and 1024 are the
meaningful candidates (128/256 fragment too aggressively); overlap is scaled with
chunk size to keep the boundary-overlap ratio (512-64; 1024-128) constant when comparing them.
---

## Evaluation: results and methodology

Metrics are computed with RAGAS using a **local** Ollama judge (no paid API).
NOTE: results are based on small sample sizes due to hardware limitations.

| Run                                   | answer_relevancy | context_precision |
|---------------------------------------|-----------------:|------------------:|
| generator = judge = llama3.2:1b       | 0.32             | 1.00              |
| generator = qwen2.5:1.5b, judge = llama3.2:1b | **0.69** | **1.00**          |
| generator = qwen2.5:1.5b, judge = llama3.2:1b, held-out eval   | **0.5** | **1.00** |


**Two findings.**

1. **Retrieval is robust.** `context_precision` stays at 1.0 across configurations —
   the hybrid retrieval consistently pulls relevant context. Retrieval is not the
   bottleneck.
2. **The generation model is the quality changer.** Swapping the generator from
   llama3.2:1b to qwen2.5:1.5b roughly doubled `answer_relevancy` (0.32 → 0.69) with
   no change to retrieval. For this task, generation-model choice drives answer
   quality. Perhaps using a larger model may improve the pipeline's performance.

**Self-preference bias.** The first run used the same model to both generate and
judge, which inflates scores. Later
runs keep generator != judge so the evaluation is independent. For more information, users can find here: [SELF-PREFERENCE BIAS IN LLM-AS-A-JUDGE]https://arxiv.org/pdf/2410.21819

---

## Honest limitations

- **Small judge.** RAGAS runs on a local sub-2B judge to stay free and offline. Small
  judges produce indicative (not authoritative) scores and occasionally emit malformed
  JSON; runs use `raise_exceptions=False` for resilience.
- **Eval-set bias.** Hand-written eval questions partly test the system on questions it
  was set up to answer well. `make_eval_holdout.py` addresses this by drawing questions
  from a **disjoint slice of the same dataset** (rows never indexed) — an unbiased,
  same-distribution held-out set. Running large held-out evals locally is constrained
  by judge reliability; a hosted judge or GPU would enable scale.
- **Small N.** With ~5–12 eval questions, score differences can be within noise; treat
  them as indicative.
- **RAG, not fine-tuning.** Knowledge lives in the vector store and is retrieved at
  query time; model weights are never trained. A deliberate choice, suited to a
  knowledge base that can change without re-training. Those weights are left for future work; 
  for instance, using Optuna to give more weight on more advanced questions. 

---

## Repository structure

```
src/         fetcher, indexer (chunker + Weaviate), pipeline, evaluate
configs/     Hydra config groups (fetcher, chunker, indexer, pipeline, evaluator)
data/        eval.yaml (held-out Q&A pairs) eval_handwritten.yaml
run.py       Hydra entry point — wires the full pipeline
make_eval_holdout.py   generates the unbiased held-out eval set (you can select N_EVAL within the script)
chunk_stats.py         chunk-size distribution analysis
docker-compose.yml     Weaviate + MLflow
```

---

## Possible extensions

- Larger held-out evaluation with a hosted or GPU-backed judge
- Optuna sweep over `alpha`, `chunk_size`, `top_k` (retrieval-only metrics first)
- A Corrective-RAG (CRAG) branch that grades retrieval before trusting it
