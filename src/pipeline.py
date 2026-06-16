import logging
from typing import Any, Dict, List

import weaviate
from langchain.embeddings.base import Embeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_weaviate.vectorstores import WeaviateVectorStore
from sentence_transformers import SentenceTransformer

from src.indexer import WeaviateIndexer

log = logging.getLogger(__name__)

# Embeddings adapter 
"""
#LangChain calls embed_query(text)
    1. MiniLMEmbeddings receives it
    2. calls self._model.encode(text)  ← SentenceTransformer
    3.  returns a list of floats
    4. LangChain is happy
"""
class MiniLMEmbeddings(Embeddings):
    """LangChain embeddings adapter for SentenceTransformer models."""

    def __init__(self, model: SentenceTransformer):
        self._model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self._model.encode(text, normalize_embeddings=True).tolist()
    
# ── RAG Pipeline 

class RAGPipeline:
    """
    Simple RAG pipeline:
      question → Weaviate hybrid search → Llama generation → answer + sources

    Hydra-instantiable via configs/pipeline/default.yaml.
    Call .attach(indexer) after instantiation to wire up Weaviate.
    """

    def __init__(self, model: str, temperature: float, top_k: int, alpha: float,
                 num_predict: int, system_prompt: str, **kwargs,):
    
        self.model       = model
        self.temperature = temperature
        self.top_k       = top_k
        self.alpha       = alpha
        self.num_predict = num_predict
        self.system_prompt = system_prompt
        self._store      = None   # set by attach()

        # LLM — all params from Hydra config
        self._llm = ChatOllama(
            model       = model,
            temperature = temperature,
            num_predict = num_predict,
        )

        self._prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}"),])

        # LCEL chain — context pre-populated, no double retrieval; the | operator chains steps together like a Unix pipe
        self._chain = self._prompt | self._llm | StrOutputParser()

        log.info(f"RAGPipeline ready model={model}, top_k={top_k}, alpha={alpha}")

    def attach(self, indexer: WeaviateIndexer) -> "RAGPipeline":
        """Wire up Weaviate store after Hydra instantiation.
           WeaviateVectorStore
           calls embed_query()
           MiniLMEmbeddings receives it
           calls SentenceTransformer.encode()
           returns vector
           WeaviateVectorStore searches Weaviate"""
        self._store = WeaviateVectorStore(
            client     = indexer.client,
            index_name = indexer.class_name,
            text_key   = "body",
            embedding  = MiniLMEmbeddings(indexer.model),
            attributes = ["title", "url", "tags"],
        )
        log.info(f"RAGPipeline attached to Weaviate collection: {indexer.class_name}")
        return self

    def run(self, question: str) -> Dict[str, Any]:
        """Answer a question using retrieved context."""
        if self._store is None:
            raise RuntimeError("Call .attach(indexer) before running the pipeline.")

        log.info(f"Query: {question[:80]}")

        # Single retrieval — no double fetch
        docs    = self._store.similarity_search(question, k=self.top_k, alpha=self.alpha)
        context = self._format_docs(docs)

        # Generate answer
        answer = self._chain.invoke({"question": question, "context": context})

        return {
            "question": question,
            "answer":   answer,
            "sources":  self._extract_sources(docs),
        }

    @staticmethod
    def _format_docs(docs) -> str:
        return "\n\n---\n\n".join(
            f"[{d.metadata.get('title', '')} | {d.metadata.get('url', '')}]\n{d.page_content}"
            for d in docs
        )

    @staticmethod
    def _extract_sources(docs) -> List[Dict]:
        return [
            {
                "title":   d.metadata.get("title", ""),
                "url":     d.metadata.get("url", ""),
                "snippet": d.page_content[:150] + "...",
            }
            for d in docs
        ]