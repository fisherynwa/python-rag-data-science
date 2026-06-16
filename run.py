"""
run.py — Hydra entry point for the Python RAG pipeline.

Flow:
  fetch (HuggingFace) → chunk → index (Weaviate) → test query → RAGAS eval → MLflow

Usage:
    docker compose up -d          # start Weaviate + MLflow
    ollama pull llama3.2:1b       # pull the model (once)

    uv run python run.py                                   # defaults
    uv run python run.py fetcher.max_rows=20               # quick test
    uv run python run.py pipeline.alpha=0.0                # pure BM25
    uv run python run.py chunker.chunk_size=256 chunker.overlap=32
"""

import logging

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.evaluate import RAGASEvaluator

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    indexer = None
    try:
        # Instantiate components from Hydra config
        fetcher  = instantiate(cfg.fetcher)
        chunker  = instantiate(cfg.chunker)
        indexer  = instantiate(cfg.indexer)
        pipeline = instantiate(cfg.pipeline).attach(indexer)

        # Evaluator needs the runtime pipeline — built directly, not via Hydra
        evaluator = RAGASEvaluator(
            pipeline=pipeline,
            judge_model=cfg.evaluator.judge_model,
        )

        # Fetch → chunk → index
        docs   = fetcher.fetch()
        chunks = chunker.chunk(docs)
        log.info(f"Fetched {len(docs)} docs → {len(chunks)} chunks")

        indexer.reset()
        indexer.index(chunks)

        # Test query
        result = pipeline.run("How do I read a CSV with missing values?")
        print("\n" + "-" * 50)
        print(f"Q: {result['question']}")
        print(f"\nA: {result['answer']}")
        print("-" * 50 + "\n")

        # 4. RAGAS evaluation — pass settings through for run naming + params
        evaluator.run(
            data_path       = "data/eval.yaml",
            experiment_name = cfg.mlflow.experiment_name,
            tracking_uri    = cfg.mlflow.tracking_uri,
            chunk_size      = cfg.chunker.chunk_size,
            overlap         = cfg.chunker.overlap,
            alpha           = cfg.pipeline.alpha,
            top_k           = cfg.pipeline.top_k,
        )

    finally:
        if indexer:
            indexer.close()


if __name__ == "__main__":
    main()