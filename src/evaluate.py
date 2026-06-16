
import logging
from typing import Dict, List
 
import mlflow
import yaml
from datasets import Dataset
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.run_config import RunConfig
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    #context_recall,
    #faithfulness,
)
 
from src.pipeline import RAGPipeline
 
log = logging.getLogger(__name__)
 
# A real embedding model — required for answer_relevancy.
# Chat models (llama3.2) do NOT work reliably as embedders.
EMBED_JUDGE = "nomic-embed-text"
 
 
class RAGASEvaluator:
    """Evaluate a RAGPipeline using RAGAS with a local Ollama judge."""
 
    def __init__(
        self,
        pipeline:    RAGPipeline,
        judge_model: str = None,
        embed_model: str = EMBED_JUDGE,
        timeout:     int = 90,
        max_workers: int = 1,
        **kwargs,
    ):
        self.pipeline    = pipeline
        # Judge LLM defaults to the same model the pipeline uses
        self.judge_model = judge_model or getattr(pipeline, "model", "llama3.2")
        self.embed_model = embed_model
        self.timeout     = timeout
        self.max_workers = max_workers
 
    def _load_questions(self, data_path: str) -> List[Dict]:
        with open(data_path) as f:
            data = yaml.safe_load(f)
        log.info(f"Loaded {len(data['questions'])} eval questions from {data_path}")
        return data["questions"]
 
    def run(
        self,
        data_path:       str,
        experiment_name: str,
        tracking_uri:    str,
        chunk_size:      int   = None,
        overlap:         int   = None,
        alpha:           float = None,
        top_k:           int   = None,
    ) -> Dict[str, float]:
        """Run RAGAS evaluation (local judge) and log to MLflow."""
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
 
        run_name = f"chunk{chunk_size}_ov{overlap}_a{alpha}"
 
        # Local judge — LLM for reasoning metrics, real embedder for relevancy
        judge_llm = LangchainLLMWrapper(
            ChatOllama(model=self.judge_model, temperature=0)
        )
        judge_embeddings = LangchainEmbeddingsWrapper(
            OllamaEmbeddings(model=self.embed_model)
        )
 
        # Timeout so one stuck call can't freeze the whole evaluation
        run_config = RunConfig(max_workers=self.max_workers, timeout=self.timeout)
 
        with mlflow.start_run(run_name=run_name):
            log.info(
                f"Running RAGAS... ({run_name}, judge={self.judge_model}, "
                f"embed={self.embed_model}, timeout={self.timeout}s)"
            )
 
            mlflow.log_params({
                "chunk_size":  chunk_size,
                "overlap":     overlap,
                "alpha":       alpha,
                "top_k":       top_k,
                "model":       getattr(self.pipeline, "model", "unknown"),
                "judge_model": self.judge_model,
                "embed_model": self.embed_model,
            })
 
            questions = self._load_questions(data_path)
            rows      = []
 
            for item in questions:
                log.info(f"  Q: {item['question'][:60]}")
                result = self.pipeline.run(item["question"])
                rows.append({
                    "question":     item["question"],
                    "answer":       result["answer"],
                    "contexts":     [s["snippet"] for s in result["sources"]],
                    "ground_truth": item["ground_truth"],
                    "weight":       item.get("weight", 1.0),
                })
 
            dataset = Dataset.from_list(rows)
            scores  = evaluate(
                dataset,
                #metrics    = [faithfulness, answer_relevancy, context_precision, context_recall],
                metrics = [answer_relevancy, context_precision],
                llm        = judge_llm,
                embeddings = judge_embeddings,
                run_config = run_config,
                raise_exceptions = False, 
            )
            results = {k: round(float(v), 4) for k, v in scores.items()}
 
            mlflow.log_metrics(results)
 
            print("\n" + "=" * 50)
            print(f"  RAGAS EVALUATION SCORES — {run_name}")
            print("=" * 50)
            for k, v in results.items():
                bar = "█" * int(v * 30) + "░" * (30 - int(v * 30))
                print(f"  {k:<25} {bar} {v:.3f}")
            print("=" * 50)
            print(f"\n  View in MLflow: {tracking_uri}\n")
 
            log.info(f"RAGAS scores: {results}")
            return results
 
