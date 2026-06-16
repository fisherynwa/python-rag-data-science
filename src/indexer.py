import logging
from typing import List, Dict

import weaviate
import weaviate.classes as wvc
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)


class DocumentChunker:
    def __init__(self, chunk_size: int, overlap: int, splitter: str, **kwargs):
        self.chunk_size = chunk_size
        self.overlap    = overlap
        self.splitter   = splitter
    def chunk(self, docs: List[Dict]) -> List[Dict]:
        """
        Split documents into smaller chunks using LangChain's RecursiveCharacterTextSplitter.
        Returns list of {id, title, body, url, tags}.
        """
        if self.splitter == "python":
            text_splitter = RecursiveCharacterTextSplitter.from_language(
                language      = Language.PYTHON,
                chunk_size    = self.chunk_size,
                chunk_overlap = self.overlap,
            )
        elif self.splitter == "markdown":
            text_splitter = RecursiveCharacterTextSplitter.from_language(
                language      = Language.MARKDOWN,
                chunk_size    = self.chunk_size,
                chunk_overlap = self.overlap,
            )
        else:   # default recursive
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size    = self.chunk_size,
                chunk_overlap = self.overlap,
            )
        chunks = []
        for doc in docs:
            doc_chunks = text_splitter.create_documents([doc["body"]])
            for i, chunk in enumerate(doc_chunks):
                    chunks.append({
                    "id":    f"{doc['id']}_{i}",
                    "title": doc["title"],
                    "body":  chunk.page_content,
                    "url":   doc["url"],
                    "tags":  ",".join(doc["tags"]) if isinstance(doc["tags"], list) else doc["tags"],  # fix here
                })

        return chunks

 

class WeaviateIndexer:
    """
    Connect to Weaviate, create collection schema, and index chunks.
    Hydra-instantiable via configs/indexer/weaviate.yaml.
    """
 
    def __init__(self, host: str, port: int, embed_model:str, class_name: str,
        dimension: int, batch_size:  int, **kwargs, ):
        self.host        = host
        self.port        = port
        self.embed_model = embed_model
        self.class_name  = class_name
        self.dimension   = dimension
        self.batch_size  = batch_size
 
        # Connect to Weaviate
        self.client = weaviate.connect_to_local(host = self.host, port = self.port, skip_init_checks = True,)
        log.info(f"Connected to Weaviate: {self.host}:{self.port}")
 
        # Load embedding model once — reused in index()
        self.model = SentenceTransformer(self.embed_model)
        log.info(f"Embedding model loaded: {self.embed_model}")
 
    def setup(self) -> None:
        """Create collection if it doesn't exist."""
        if self.client.collections.exists(self.class_name):
            log.info(f"Collection '{self.class_name}' already exists — skipping")
            return
 
        self.client.collections.create(
            name                = self.class_name,
            description         = "Python data extraction doc chunks",
            vectorizer_config   = wvc.config.Configure.Vectorizer.none(),
            vector_index_config = wvc.config.Configure.VectorIndex.hnsw(
                distance_metric = wvc.config.VectorDistances.COSINE,
            ),
            properties = [
                wvc.config.Property(name="title", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="body",  data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="url",   data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="tags",  data_type=wvc.config.DataType.TEXT),
            ],
        )
        log.info(f"Collection '{self.class_name}' created")
 
    def index(self, chunks: List[Dict]) -> None:
        """
        Embed all chunks in one batch and upload to Weaviate.
        Vectors are normalized to unit length for cosine similarity.
        """
        log.info("Generating embeddings...")
        texts   = [chunk["body"] for chunk in chunks]
        vectors = self.model.encode(
            texts,
            batch_size           = self.batch_size,
            show_progress_bar    = True, 
            normalize_embeddings = True, # If cosine is chosen,
             # all vectors are normalized to length 1 at read time and dot product is used to calculate the distance for computational efficiency.
        )
 
        log.info("Uploading to Weaviate...")
        collection = self.client.collections.get(self.class_name)
        with collection.batch.fixed_size(batch_size=self.batch_size) as batch:
            for chunk, vector in zip(chunks, vectors):
                batch.add_object(
                    properties = {
                        "title": chunk["title"],
                        "body":  chunk["body"],
                        "url":   chunk["url"],
                        "tags":  chunk["tags"],
                    },
                    vector = vector.tolist(),
                )
 
        log.info(f"Indexed {len(chunks)} chunks into '{self.class_name}' ✓")

    def reset(self) -> None:
        """Delete and recreate the collection — fresh start."""
        if self.client.collections.exists(self.class_name):
            self.client.collections.delete(self.class_name)
            log.info(f"Deleted collection '{self.class_name}'")
        self.setup()

    def close(self) -> None:
        """Close Weaviate connection."""
        self.client.close()
        log.info("Weaviate connection closed")

