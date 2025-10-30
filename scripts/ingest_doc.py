#!/usr/bin/env python
# -*- coding: utf-8 -*-

##########################
##   IMPORT MODULES
##########################
# - Import standard modules
import argparse
import os
import sys
import structlog
from typing import List, Iterable, Optional

# - Import llama-index modules
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    Settings,
)
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core import ServiceContext

# - Import qdrant modules
import qdrant_client

logger = structlog.get_logger()


#############################
##  HELPER CLASSES
#############################
class SafeEmbedder(BaseEmbedding):
    """Guards against non-string/blank inputs and delegates to an inner embedder."""
    def __init__(self, inner):
        super().__init__()
        self.inner = inner  # e.g., HuggingFaceEmbedding / SentenceTransformerEmbedding

    # ------- helpers -------
    @staticmethod
    def _ok(s: Optional[str]) -> bool:
        return isinstance(s, str) and bool(s.strip())

    @classmethod
    def _flt(cls, texts: Iterable[Optional[str]]) -> List[str]:
        return [t for t in texts if cls._ok(t)]

    # ------- public API (keep these for robustness) -------
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        texts = self._flt(texts)
        if not texts:
            return []
        return self.inner.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        if not self._ok(text):
            raise ValueError("Query text is empty or not a string")
        return self.inner.embed_query(text)

    # ------- abstract methods required by BaseEmbedding -------
    # Single items
    def _get_text_embedding(self, text: str) -> List[float]:
        if not self._ok(text):
            raise ValueError("Text is empty or not a string")
        # Delegate via batch to keep behavior identical to inner
        return self.inner.embed_documents([text])[0]

    def _get_query_embedding(self, query: str) -> List[float]:
        if not self._ok(query):
            raise ValueError("Query is empty or not a string")
        return self.inner.embed_query(query)

    # Batches (some LI versions call these; implement to be safe)
    def _get_text_embedding_batch(self, texts: List[str]) -> List[List[float]]:
        return self.embed_documents(texts)

    # Async variants (some LI versions require these)
    async def _aget_query_embedding(self, query: str) -> List[float]:
        # simple sync delegate; replace with await inner.aget_query_embedding if available
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    async def _aget_text_embedding_batch(self, texts: List[str]) -> List[List[float]]:
        return self._get_text_embedding_batch(texts)
        
#############################
##  HELPER METHODS
#############################
def ingest(
    data_path,
    collection_name,
    chunk_size,
    embedder,
    qdrant_url,
    file_exts=[".pdf"],
    recursive=True,
    skip_baddoc=False,
    use_safe_embedder=False
):
    """ Method to ingest paper in DB """
    logger.info("Indexing data...")
    documents = SimpleDirectoryReader(data_path, required_exts=file_exts, recursive=recursive).load_data()

    logger.info("Creating qdrant store ...")
    client = qdrant_client.QdrantClient(url=qdrant_url)
    qdrant_vector_store = QdrantVectorStore(
        client=client, collection_name=collection_name
    )
    storage_context = StorageContext.from_defaults(vector_store=qdrant_vector_store)


    # - Validate docs
    # 1) Helper to extract safe text from any LlamaIndex object
    def get_text_safe(obj):
        # Documents have .text; Nodes have get_content(...)
        t = None
        if hasattr(obj, "get_content"):
            try:
                t = obj.get_content(metadata_mode="none")
            except Exception:
                t = None
        if t is None:
            t = getattr(obj, "text", None)
        return t if isinstance(t, str) else None

    # 2) Validate & filter documents
    bad = []
    good_documents = []
    for i, d in enumerate(documents):
        t = get_text_safe(d)
        if t is None or not t.strip():
            bad.append((i, getattr(d, "id_", None), getattr(d, "metadata", {})))
        else:
            good_documents.append(d)

    if bad:
        logger.warning("%d bad/invalid docs (empty/non-string text) found. Example: %s", len(bad), bad[0])

    logger.info(f"#good/tot={len(good_documents)}/{len(documents)} documents to be stored ...")

    # - Create index store
    logger.info("Creating index store ...")
    Settings.llm = None
    Settings.embed_model = embedder
    Settings.chunk_size = chunk_size
    
    documents_to_be_stored= documents
    if skip_baddoc:
        logger.info(f"Storing only good documents (N={len(good_documents)}) ...")
        documents_to_be_stored= good_documents 
        
    logger.info("doc types: %s", {type(d) for d in documents_to_be_stored})
        
    # - Create Safe embedder wrapper to ignore/guard bad strings
    if use_safe_embedder:
        logger.info("Creating safe embedder ...")
        safe_embedder = SafeEmbedder(embedder)
    
        logger.info("Creating service context ...")
        service_context = ServiceContext.from_defaults(
            llm=None,
            embed_model=safe_embedder,
            chunk_size=chunk_size,
        )

        # - Store documents
        logger.info("Storing documents ...")
        index = VectorStoreIndex.from_documents(
            documents_to_be_stored,
            storage_context=storage_context,
            service_context=service_context,
        )
    else:
        # - Store documents   
        logger.info("Storing documents ...") 
        index = VectorStoreIndex.from_documents(
            documents_to_be_stored, 
            storage_context=storage_context, 
            Settings=Settings,
        )
    
    logger.info(
       "Data indexed successfully to Qdrant",
       collection=collection_name,
    )
    return index

######################################
##     MAIN
######################################
def main():

    #=============================================
    # ==      ARGUMENTS
    # ============================================
    parser = argparse.ArgumentParser()
    
    # - Doc options
    parser.add_argument("-data_path", "--data_path", type=str, required=True, help="Data directory containing papers/docs to be ingested")
    parser.add_argument("-collection_name", "--collection_name", type=str, required=True, help="Collection name")
    parser.add_argument("--recursive", dest="recursive", action='store_true',help='Search file recursively from the given data_path (default=False)')	
    parser.set_defaults(recursive=False)
    parser.add_argument("-file_exts", "--file_exts", dest="file_exts", required=False, type=str, default='.pdf', action='store', help='Required file extensions, separated by commas')
    parser.add_argument("-chunk_size", "--chunk_size", type=int, required=False, default=1024, help="Document chunk size")
    parser.add_argument("--skip_baddoc", dest="skip_baddoc", action='store_true',help='Skip bad documents (default=False)')	
    parser.set_defaults(skip_baddoc=False)
    
    # - Model options
    parser.add_argument("-embedding_model", "--embedding_model", type=str, required=False, default="mixedbread-ai/mxbai-embed-large-v1", help="Embedder model")
    #parser.add_argument("-embedding_model", "--embedding_model", type=str, required=False, default="sentence-transformers/all-mpnet-base-v2", help="Embedder model")
    #parser.add_argument("-embedding_model", "--embedding_model", type=str, required=False, default="Qwen/Qwen3-Embedding-8B", help="Embedder model")
    #parser.add_argument("-embedding_model", "--embedding_model", type=str, required=False, default="nvidia/llama-embed-nemotron-8b", help="Embedder model")
    parser.add_argument("--use_safe_embedder", dest="use_safe_embedder", action='store_true',help='Use safe embedder (in case of upload errors) (default=False)')	
    parser.set_defaults(use_safe_embedder=False)
    
    # - Storage options
    parser.add_argument("-qdrant_url", "--qdrant_url", type=str, required=False, default="http://localhost:6333", help="QDRant URL")
    
    # - Parse options
    args = parser.parse_args()
    
    file_exts= [str(x.strip()) for x in args.file_exts.split(',')]

    #=============================================
    # ==      LOAD MODEL
    # ============================================
    # - Load embedded model
    logger.info(f"Loading embedder model {args.embedding_model} ...")
    embed_model = HuggingFaceEmbedding(
        model_name=args.embedding_model, 
        #cache_folder="./cache", 
        trust_remote_code=True
    )

    #=============================================
    # ==      UPLOAD PAPERS/DOCS
    # ============================================
    # - Ingest paper
    logger.info("Ingesting docs/papers ...")
    ingest(
        data_path=args.data_path,
        collection_name=args.collection_name,
        chunk_size=args.chunk_size,
        embedder=embed_model,
        qdrant_url=args.qdrant_url,
        file_exts=args.file_exts,
        recursive=args.recursive,
        skip_baddoc=args.skip_baddoc,
        use_safe_embedder=args.use_safe_embedder,
    )
    
    logger.info("Ingest completed.")

########################
##    RUN MAIN
########################
if __name__ == "__main__":
    main()
