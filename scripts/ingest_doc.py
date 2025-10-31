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
from typing import List, Iterable, Optional, Any
from pydantic import PrivateAttr

# - Import llama-index modules
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    Settings,
    ServiceContext
)
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.node_parser import SentenceSplitter

# - Import qdrant modules
import qdrant_client

logger = structlog.get_logger()


#############################
##  HELPER CLASSES
#############################
class SafeEmbedder(BaseEmbedding):
    _inner: Any = PrivateAttr()

    def __init__(self, inner: Any):
        super().__init__()
        self._inner = inner

    # ---- helpers ----
    @staticmethod
    def _coerce_one(x: Any) -> Optional[str]:
        # bytes → utf-8
        if isinstance(x, bytes):
            try:
                x = x.decode("utf-8", errors="ignore")
            except Exception:
                return None
        # keep only non-empty strings
        if isinstance(x, str) and x.strip():
            return x
        return None

    @classmethod
    def _coerce_batch(cls, texts: Iterable[Any]) -> List[str]:
        fixed: List[str] = []
        bad_idx: List[int] = []
        for i, t in enumerate(texts):
            ok = cls._coerce_one(t)
            if ok is None:
                bad_idx.append(i)
                fixed.append(" ")  # placeholder to preserve batch length
            else:
                fixed.append(ok)
        if bad_idx:
            # Only log; do not raise. We’ve sanitized them.
            # (Use your logger instead of print if you prefer)
            print(f"[SafeEmbedder] sanitized {len(bad_idx)} items; sample idx: {bad_idx[:5]}")
        return fixed

    # ---- delegation helpers ----
    def _delegate_text_batch(self, texts: List[str]) -> List[List[float]]:
        if hasattr(self._inner, "get_text_embedding_batch"):
            return self._inner.get_text_embedding_batch(texts)
        if hasattr(self._inner, "get_text_embedding"):
            return [self._inner.get_text_embedding(t) for t in texts]
        if hasattr(self._inner, "embed_documents"):
            return self._inner.embed_documents(texts)
        if hasattr(self._inner, "encode"):
            return self._inner.encode(texts, convert_to_numpy=False)
        raise AttributeError("Inner embedder lacks a compatible batch method")

    def _delegate_text_single(self, text: str) -> List[float]:
        if hasattr(self._inner, "get_text_embedding"):
            return self._inner.get_text_embedding(text)
        if hasattr(self._inner, "get_text_embedding_batch"):
            return self._inner.get_text_embedding_batch([text])[0]
        if hasattr(self._inner, "embed_documents"):
            return self._inner.embed_documents([text])[0]
        if hasattr(self._inner, "encode"):
            return self._inner.encode([text], convert_to_numpy=False)[0]
        raise AttributeError("Inner embedder lacks a compatible single-text method")

    def _delegate_query_single(self, query: str) -> List[float]:
        if hasattr(self._inner, "get_query_embedding"):
            return self._inner.get_query_embedding(query)
        return self._delegate_text_single(query)

    # ---- BaseEmbedding hooks ----
    def _get_text_embedding_batch(self, texts: List[Any]) -> List[List[float]]:
        texts = self._coerce_batch(texts)
        return self._delegate_text_batch(texts)

    def _get_text_embedding(self, text: Any) -> List[float]:
        text = self._coerce_one(text)
        if text is None:
            text = " "
        return self._delegate_text_single(text)

    def _get_query_embedding(self, query: Any) -> List[float]:
        query = self._coerce_one(query)
        if query is None:
            query = " "
        return self._delegate_query_single(query)

    # ---- async shims ----
    async def _aget_text_embedding_batch(self, texts: List[Any]) -> List[List[float]]:
        return self._get_text_embedding_batch(texts)

    async def _aget_text_embedding(self, text: Any) -> List[float]:
        return self._get_text_embedding(text)

    async def _aget_query_embedding(self, query: Any) -> List[float]:
        return self._get_query_embedding(query)
        
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

    
    documents_to_be_stored= documents
    if skip_baddoc:
        logger.info(f"Storing only good documents (N={len(good_documents)}) ...")
        documents_to_be_stored= good_documents 
        
    logger.info("doc types: %s", {type(d) for d in documents_to_be_stored})
        
    # - Create Safe embedder wrapper to ignore/guard bad strings
    if use_safe_embedder:
        logger.info("Creating safe embedder ...")
        safe_embedder = SafeEmbedder(embedder)
    
        # - Create settings
        logger.info("Creating settings ...")
        Settings.llm = None
        Settings.embed_model = safe_embedder
        Settings.chunk_size = chunk_size

        # - Store documents
        #logger.info("Storing documents ...")
        #index = VectorStoreIndex.from_documents(
        #    documents_to_be_stored,
        #    storage_context=storage_context,
        #    Settings=Settings,
        #)
        
        # - Creating splitter
        logger.info("Creating sentence splitter ...")
        splitter = SentenceSplitter(chunk_size=chunk_size)
            
        # - Get nodes from documents
        logger.info("Getting nodes from documents ...")
        nodes = splitter.get_nodes_from_documents(documents_to_be_stored)

        def node_text(n):
            try:
                return n.get_content(metadata_mode="none")
            except Exception:
                return None

        # - Selecting good nodes
        logger.info("Selecting good nodes ...")
        clean_nodes = []
        bad_nodes = []
        for n in nodes:
            t = node_text(n)
            # Also repair bytes → str
            if isinstance(t, bytes):
                try:
                    t = t.decode("utf-8", errors="ignore")
                    n.text = t  # persist the decoded text on the node
                except Exception:
                    t = None
            if isinstance(t, str) and t.strip():
                clean_nodes.append(n)
            else:
                bad_nodes.append(getattr(n, "id_", None))

        if bad_nodes:
            logger.warning("Skipping %d invalid chunks (post-split). Example: %s", len(bad_nodes), bad_nodes[:3])

        logger.info("nodes=%d, sample_text_len=%s", len(clean_nodes), len(clean_nodes[0].get_content()) if clean_nodes else None)
        logger.info("embedder=%s", type(Settings.embed_model).__name__)

        # - Define method to store index
        def build_index_from_nodes(nodes, storage_context):
            try:
                # Newer API (?)
                return VectorStoreIndex.from_nodes(nodes, storage_context=storage_context)
            except Exception as e:
                logger.warning(f"VectorStoreIndex.from_nodes failed (err={str(e)}), trying an alternative method ...")
                return VectorStoreIndex(nodes, storage_context=storage_context)

        # - Store documents
        logger.info("Storing documents ...")
        index = build_index_from_nodes(clean_nodes, storage_context)
        
    else:
        # - Create settings
        logger.info("Creating settings ...")
        Settings.llm = None
        Settings.embed_model = embedder
        Settings.chunk_size = chunk_size
    
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
