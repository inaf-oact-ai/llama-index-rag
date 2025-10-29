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

# - Import qdrant modules
import qdrant_client

logger = structlog.get_logger()

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
    skip_baddoc=False
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
        logger.warning("Skipping %d invalid docs (empty/non-string text). Example: %s", len(bad), bad[0])

    # - Create index store
    logger.info("Creating index store ...")
    Settings.llm = None
    Settings.embed_model = embedder
    Settings.chunk_size = chunk_size
    index = VectorStoreIndex.from_documents(
        good_documents if skip_baddoc else documents, 
        storage_context=storage_context, 
        Settings=Settings
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
        skip_baddoc=args.skip_baddoc
    )
    
    logger.info("Ingest completed.")

########################
##    RUN MAIN
########################
if __name__ == "__main__":
    main()
