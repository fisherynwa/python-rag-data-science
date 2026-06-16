import logging
from typing import List, Dict

log = logging.getLogger(__name__)


class HuggingFaceFetcher:
    """
    Fetch Python Q&A pairs from a HuggingFace dataset.
    Hydra-instantiable via configs/fetcher/huggingface.yaml.
    All defaults defined in config; this class uses what it receives.
    """

    def __init__(self, dataset_name: str, max_rows: int, seed:int,
         keywords: List[str] = None, **kwargs, ):
        self.dataset_name = dataset_name
        self.max_rows     = max_rows
        self.seed         = seed
        self.keywords     = keywords or []

    def fetch(self, split: str = "train") -> List[Dict]:
        """
        Load, shuffle and filter Python Q&A pairs.
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Run: uv add datasets")

        log.info(
            f"Loading dataset: {self.dataset_name} "
            f"(max_rows={self.max_rows}, seed={self.seed})"
        )
        # Load and shuffle dataset; split in {"train", "test"}
        ds = load_dataset(self.dataset_name, split=split)
        ds = ds.shuffle(seed=self.seed)

        docs = []
        for i, row in enumerate(ds):
            if len(docs) >= self.max_rows:
                break

            instruction = row.get("instruction", "").strip() # remove leading/trailing whitespace; default to empty string if key is missing
            output      = row.get("output", "").strip()
            context     = row.get("input", "").strip()

            if not instruction or not output:
                continue

            # Keep only Python/data related rows
            text = (instruction + output).lower()
            if not any(kw in text for kw in self.keywords):
                continue

            body = f"Question:\n{instruction}"
            if context:
                body += f"\n\nContext:\n{context}"
            body += f"\n\nAnswer:\n{output}"

            docs.append({
                "id":    f"hf_{i}",
                "title": instruction[:100], # Truncate title to 100 chars; title is not indexed, so we can afford to keep it short and focused on the question
                "body":  body, # body is indexed and chunked, so it can be longer and contain the full Q&A with context
                "url":   f"https://huggingface.co/datasets/{self.dataset_name}",
                "tags":  ["python", "data-extraction"],
            })

        log.info(f"Loaded {len(docs)} rows from {self.dataset_name}")
        return docs