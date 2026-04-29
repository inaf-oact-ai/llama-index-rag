#!/usr/bin/env python
# -*- coding: utf-8 -*-

##########################
##   IMPORT MODULES
##########################
# - import standard modules
import os
import sys
import argparse
from typing import Optional
import structlog

# - Pydantic
from pydantic import BaseModel, Field

# - Import OLLAMA
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import PydanticSingleSelector, PydanticMultiSelector, LLMMultiSelector
from llama_index.core.selectors.embedding_selectors import EmbeddingSingleSelector
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

# - Import FastAPI
import uvicorn
from fastapi import FastAPI

# - Import qdrant
import qdrant_client

logger = structlog.get_logger()

#############################
##    HELPERS
#############################
def _safe_first_author(value):
    if isinstance(value, list) and value:
        return value[0]
    if isinstance(value, str) and value.strip():
        return value
    return None
    
def _metadata_file_path(md):
    return md.get("file_path") or md.get("filepath")

def _source_collection(sn, md):
    return (
        md.get("_collection_name")
        or md.get("collection")
        or md.get("collection_name")
    )

def _node_text(sn):
    try:
        return sn.node.get_content() or ""
    except Exception:
        return ""

def _node_id(sn):
    md = sn.node.metadata or {}
    return getattr(sn.node, "node_id", None) or md.get("node_id") or md.get("id_")

def _infer_collection_from_metadata(md):
    fp = (md.get("file_path") or md.get("filepath") or "").lower()
    fn = (md.get("file_name") or "").lower()
    kind = md.get("kind")

    if kind == "book":
        return "radiobooks"
    if kind == "annual-review":
        return "annreviews"
    if "radioimg-arxiv-dataset" in fp or fn.endswith(".pdf"):
        return "radiopapers"

    return None

#############################
##    RAG CLASS
#############################
collection_descriptions = {
    "radiopapers": "Scientific papers stored in ArXiV repository with subject keywords related to radio astronomy.",
    "radiobooks": "Textbooks and monographs related to radio astronomy.",
    "annreviews": "Annual Reviews articles and review papers relevant to astronomy and astrophysics.",
    #"solar-papers": "Scientific papers stored in ArXiV repository with subject keywords related to radio astronomy.",
    "solar-living-reviews": "Springer Living Reviews in Solar Physics articles",
}
DOMAIN_COLLECTIONS = {
    "radio": ["radiopapers", "radiobooks", "annreviews"],
    #"solar": ["solar-papers", "solar-living-reviews", "annreviews"],
    "solar": ["solar-living-reviews", "annreviews"],
}

class RAG:
    def __init__(
        self,
        llm,
        embedding_model,
        chunk_size,
        collection_name,
        qdrant_url,
        collection_names=None
    ):
        """ RAG class constructor"""
        self.llm = llm  # ollama llm
        self.embedding_model= embedding_model
        self.chunk_size= chunk_size
        self.collection_name= collection_name
        self.qdrant_url=qdrant_url
        self.collection_names= collection_names
        #self.qdrant_client = qdrant_client.QdrantClient(url=qdrant_url)

    def load_embedder(self):
        logger.info(f"Load embedder model {self.embedding_model} ...")
        embed_model = HuggingFaceEmbedding(model_name=self.embedding_model)
        return embed_model

    def qdrant_index(self):
        logger.info(f"Create qrrant client (url={self.qdrant_url}) ...")
        client = qdrant_client.QdrantClient(url=self.qdrant_url)

        logger.info(f"Create QdrantVectorStore (collection_name={self.collection_name}) ...")
        qdrant_vector_store = QdrantVectorStore(
            client=client, collection_name=self.collection_name
        )

        logger.info("Create settings ...")
        if self.llm is not None:
            Settings.llm = self.llm
        Settings.embed_model = self.load_embedder()
        Settings.chunk_size = self.chunk_size

        logger.info("Retrieve index ...")
        index = VectorStoreIndex.from_vector_store(
            vector_store=qdrant_vector_store, settings=Settings
        )
        return index

    def qdrant_indices(self):
        """Build an index per Qdrant collection (multi-source)."""
        logger.info(f"Create qdrant client (url={self.qdrant_url}) ...")
        client = qdrant_client.QdrantClient(url=self.qdrant_url)

        logger.info("Create settings ...")
        if self.llm is not None:
            Settings.llm = self.llm
        Settings.embed_model = self.load_embedder()
        Settings.chunk_size = self.chunk_size

        logger.info("Retrieve index ...")
        indices = {}
        for cname in self.collection_names:
            logger.info(f"Create QdrantVectorStore (collection_name={cname}) ...")
            vs = QdrantVectorStore(client=client, collection_name=cname)
            
            logger.info("Retrieve index ...")
            idx = VectorStoreIndex.from_vector_store(vector_store=vs, settings=Settings)
            indices[cname] = idx
            
        return indices

class Query(BaseModel):
    query: str
    similarity_top_k: Optional[int] = Field(default=1, ge=1, le=10)
    domain: Optional[str] = None
    collections: Optional[list[str]] = None
    similarity_thr: Optional[float] = None
    num_queries: Optional[int] = None 

class Response(BaseModel):
    search_result: str
    content_found: bool
    status: int
    sources: list
    
class RetrieveRequest(BaseModel):
    query: str
    similarity_top_k: Optional[int] = Field(default=8, ge=1, le=100)
    collections: Optional[list[str]] = None
    similarity_thr: Optional[float] = None
    num_queries: Optional[int] = None
    response_mode: Optional[str] = "no_text"
    include_text: bool = True

class RetrievedDocument(BaseModel):
    doc_id: str
    title: Optional[str] = None
    text: str = ""
    score: Optional[float] = None
    collection: Optional[str] = None
    metadata: dict = Field(default_factory=dict)

class RetrieveResponse(BaseModel):
    status: int
    message: str = "ok"
    content_found: bool
    documents: list[RetrievedDocument] = Field(default_factory=list)
    debug: dict = Field(default_factory=dict)    


def resolve_requested_collections(
    requested_domain: Optional[str],
    requested_collections: Optional[list[str]],
    default_collections: Optional[list[str]],
    available_collections: list[str],
):
    """
    Resolve and validate the collection subset requested by the frontend.

    Priority:
    1. Explicit collections from request
    2. Domain-level collection mapping
    3. Backend startup default collections
    """

    if requested_collections:
        resolved = [
            c.strip()
            for c in requested_collections
            if isinstance(c, str) and c.strip()
        ]
    elif requested_domain:
        if requested_domain not in DOMAIN_COLLECTIONS:
            raise ValueError(
                f"Unknown domain '{requested_domain}'. Available domains: {list(DOMAIN_COLLECTIONS.keys())}"
            )
        resolved = DOMAIN_COLLECTIONS[requested_domain]
    else:
        resolved = default_collections or available_collections

    resolved = list(dict.fromkeys(resolved))

    missing = [c for c in resolved if c not in available_collections]
    if missing:
        raise ValueError(
            f"Requested collections are not loaded: {missing}. "
            f"Available collections: {available_collections}"
        )

    return resolved

def source_to_retrieved_document(source: dict) -> RetrievedDocument:
    original_metadata = dict(source.get("metadata") or {})

    metadata = dict(original_metadata)
    for key, value in source.items():
        if key not in {"text", "metadata"}:
            metadata.setdefault(key, value)

    doc_id = (
        source.get("node_id")
        or source.get("doc_id")
        or source.get("source_id")
        or source.get("doi")
        or source.get("arxiv_id")
        or source.get("file_name")
        or "unknown"
    )

    collection = (
        source.get("collection")
        or source.get("_collection_name")
        or source.get("collection_name")
        or metadata.get("collection")
        or metadata.get("_collection_name")
    )

    title = (
        source.get("title")
        or source.get("paper_title")
        or source.get("document_title")
        or source.get("file_name")
        or str(doc_id)
    )

    return RetrievedDocument(
        doc_id=str(doc_id),
        title=title,
        text=str(source.get("text") or ""),
        score=source.get("score"),
        collection=collection,
        metadata=metadata,
    )


def source_from_node(sn):
    md = sn.node.metadata or {}
    logger.debug("Parsed source metadata", metadata=md)

    doctype = md.get("kind")
    source = None

    if doctype:
        if doctype == "book": # Book
            source = get_book_metadata(sn, md)
        elif doctype == "annual-review": # Annual Review
            source = get_annreview_metadata(sn, md)
        else: # - Arxiv paper
            logger.warning(f"Unknown doctype parsed ({doctype}), treating as generic/arxiv-like entry ...")
            source = get_arxiv_metadata(sn, md)
    else:
        source = get_arxiv_metadata(sn, md)

    if source is None:
        return None

    # Preserve collection hints if available.
    source["collection"] = (
        md.get("_collection_name")
        or md.get("collection")
        or source.get("collection")
        or _infer_collection_from_metadata(md)
    )

    # Preserve all original node metadata for MAASAI.
    source["metadata"] = dict(md)

    return source


def sources_from_nodes(source_nodes):
    response_sources = []

    for sn in source_nodes or []:
        source = source_from_node(sn)
        if source is None:
            logger.warning("Response source parsed is None, skipping entry ...")
            continue
        response_sources.append(source)

    return response_sources


def get_arxiv_metadata(sn, md):
    """ Get arxiv paper metadata """

    # - Set arxiv id
    # --- arXiv extraction & normalization ---
    arxiv_raw = (
        md.get("arxiv_id")
        or md.get("arXiv")
        or md.get("arxiv")
        or md.get("eprint")
        or md.get("identifier")
    )
    
    arxiv_norm = None
    if isinstance(arxiv_raw, str):
        import re as _re
        # strip leading "arXiv:" and try to capture 2011.07620 (ignore version)
        s = _re.sub(r"^arXiv:\s*", "", arxiv_raw.strip(), flags=_re.IGNORECASE)
        m = _re.search(r"(\d{4}\.\d{4,5})", s)
        arxiv_norm = m.group(1) if m else s

    # fallback: derive from file_name like 2011.07620v2.pdf
    if not arxiv_norm:
        fn = (md.get("file_name") or "").strip()
        import re as _re
        m = _re.search(r"(\d{4}\.\d{4,5})(?:v\d+)?\.pdf$", fn)
        if m:
            arxiv_norm = m.group(1)

    arxiv_abs_url = f"https://arxiv.org/abs/{arxiv_norm}" if arxiv_norm else None
    arxiv_pdf_url = f"https://arxiv.org/pdf/{arxiv_norm}.pdf" if arxiv_norm else None
                
    # - Set response source
    source= {
        "doctype": "arxiv",
        "node_id": sn.node.node_id,
        "score": sn.score,                     # similarity score
        #"file_path": md.get("file_path"),
        "file_path": _metadata_file_path(md),
        "file_name": md.get("file_name"),
        "page_label": md.get("page_label"),
        # - Add custom metadata fields
        "title": md.get("title"),
        "paper_title": md.get("paper_title"),
        "document_title": md.get("document_title"),
        "authors": md.get("authors"),
        "author": md.get("author"),
        "first_author": md.get("first_author"),
        "journal": md.get("journal"),
        "journal_name": md.get("journal_name"),
        "container_title": md.get("container_title"),
        "publication": md.get("publication"),
        "volume": md.get("volume"),
        "issue": md.get("issue"),
        "number": md.get("number"),
        "pages": md.get("pages"),
        "page": md.get("page"),
        "year": md.get("year"),
        "pub_year": md.get("pub_year"),
        "date": md.get("date"),
        "bibcode": md.get("bibcode"),
        "doi": md.get("doi"),
        "arxiv_id": arxiv_norm,
        "arxiv_abs_url": arxiv_abs_url,
        "arxiv_pdf_url": arxiv_pdf_url,
        "text": sn.node.get_content(),
    }
    
    return source


def get_book_metadata(sn, md):
    """ Get book metadata """
    
    # - Set response source
    source= {
        "doctype": "book",
        "node_id": sn.node.node_id,
        "score": sn.score,                     # similarity score
        #"file_path": md.get("file_path"),
        "file_path": _metadata_file_path(md),
        "file_name": md.get("file_name"),
        "page_label": md.get("page_label"),
        # - Add custom metadata fields
        "title": md.get("title"),
        "subtitle": md.get("subtitle"),
        "publisher": md.get("publisher"),
        "year": md.get("year"),
        "pages": md.get("pages"),
        "authors": md.get("authors"),
        #"first_author": md.get("authors")[0],
        "first_author": _safe_first_author(md.get("authors")),
        "doi": md.get("doi"),
        "isbn": md.get("isbn"),
        "issn": md.get("issn"),
        "url": md.get("url"),
        "download_url": md.get("download_url"),
        "text": sn.node.get_content(),
    }
    
    return source
    
def get_annreview_metadata(sn, md):
    """ Get book metadata """
    
    # - Set response source
    source= {
        "doctype": "annual-review",
        "node_id": sn.node.node_id,
        "score": sn.score,                     # similarity score
        #"file_path": md.get("file_path"),
        "file_path": _metadata_file_path(md),
        "file_name": md.get("file_name"),
        "page_label": md.get("page_label"),
        # - Add custom metadata fields
        "title": md.get("title"),
        "publisher": md.get("publisher"),
        "journal": md.get("journal"),
        "volume": md.get("volume"),
        "issue": md.get("issue"),
        "year": md.get("year"),
        "month": md.get("month"),
        "pages": md.get("pages"),
        "authors": md.get("authors"),
        #"first_author": md.get("authors")[0],
        "first_author": _safe_first_author(md.get("authors")),
        "doi": md.get("doi"),
        "issn": md.get("issn"),
        "keywords": md.get("keywords"),
        "publication_type": md.get("publication_type"),
        "url": md.get("url"),
        "download_url": md.get("download_url"),
        "text": sn.node.get_content(),
    }
    
    return source



def build_fusion_retriever(
    indices: dict,
    similarity_top_k: int,
    num_queries: int,
):
    """ Build retriever """
    retrievers = [
        idx.as_retriever(similarity_top_k=similarity_top_k)
        for idx in indices.values()
    ]

    return QueryFusionRetriever(
        retrievers=retrievers,
        similarity_top_k=similarity_top_k,
        num_queries=num_queries,
        use_async=False,
        verbose=True,
    )


def build_query_engine(
    index=None,
    indices: dict | None = None,
    multi_mode: bool = False,
    similarity_top_k: int = 5,
    similarity_thr: float = 0.5,
    num_queries: int = 1,
):
    """ Build query engine """
    if not multi_mode:
        return index.as_query_engine(
            similarity_top_k=similarity_top_k,
            output=Response,
            response_mode="tree_summarize",
            include_metadata=True,
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=similarity_thr)],
            verbose=True,
        )

    fusion = build_fusion_retriever(
        indices=indices,
        similarity_top_k=similarity_top_k,
        num_queries=num_queries,
    )

    return RetrieverQueryEngine.from_args(
        fusion,
        response_mode="tree_summarize",
        include_metadata=True,
        output=Response,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=similarity_thr)],
    )


def retrieve_source_nodes(
    query_text: str,
    index=None,
    indices: dict | None = None,
    multi_mode: bool = False,
    similarity_top_k: int = 8,
    similarity_thr: float = 0.5,
    num_queries: int = 1,
):
    """ Retrieve source node """
    if not multi_mode:
        retriever = index.as_retriever(similarity_top_k=similarity_top_k)
    else:
        retriever = build_fusion_retriever(
            indices=indices,
            similarity_top_k=similarity_top_k,
            num_queries=num_queries,
        )

    nodes = retriever.retrieve(query_text)

    if similarity_thr is not None:
        nodes = [
            sn for sn in nodes
            if sn.score is None or sn.score >= similarity_thr
        ]

    return nodes



def build_llm(args):
    """Build a LlamaIndex-compatible LLM from CLI args."""

    # - Check if LLM is really needed
    if args.retrieval_only_no_llm:
        logger.info("Retrieval-only mode: disabling LLM.")
        return None

    logger.info(f"Building LLM backend={args.llm_backend}, model={args.llm} ...")

    #=========================
    #      OLLAMA
    #=========================
    if args.llm_backend == "ollama":
        return Ollama(
            model=args.llm,
            base_url=args.llm_url,
            request_timeout=args.llm_timeout,
            context_window=args.llm_ctx_window,
            keep_alive=args.llm_keep_alive,
            thinking=args.llm_thinking,
        )

    #=========================
    #      VLLM (LOCAL)
    #=========================
    elif args.llm_backend == "vllm":
        # Direct local vLLM integration
        # Requires: pip install llama-index-llms-vllm vllm
        from llama_index.llms.vllm import Vllm
        return Vllm(
            model=args.llm,
            temperature=args.llm_temperature,
            max_new_tokens=args.llm_max_new_tokens,
            tensor_parallel_size=args.llm_tensor_parallel_size,
            # Extra native vLLM params go here
            vllm_kwargs={
                # Example:
                # "gpu_memory_utilization": 0.9,
                # "max_model_len": args.llm_ctx_window,
            },
        )

    #=========================
    #      VLLM (API/REMOTE)
    #=========================
    elif args.llm_backend == "vllm-openai":
        # vLLM served with: vllm serve <model> ...
        # Requires: pip install llama-index-llms-openai-like
        from llama_index.llms.openai_like import OpenAILike
        return OpenAILike(
            model=args.llm,
            api_base=args.llm_url.rstrip("/") + "/v1",
            api_key=args.llm_api_key,
            is_chat_model=args.llm_is_chat_model,
            is_function_calling_model=args.llm_is_function_calling_model,
            context_window=args.llm_ctx_window,
            max_tokens=args.llm_max_new_tokens,
            temperature=args.llm_temperature,
            timeout=float(args.llm_timeout),
        )
    else:
        raise ValueError(f"Unsupported llm_backend={args.llm_backend}")

######################################
##    ARGS
######################################
# - Parse arguments
def load_args():
    """ Load arguments """
    parser = argparse.ArgumentParser()
    parser.add_argument("-port", "--port", type=int, required=False, default=8000, help="Bind socket to this port.")
    parser.add_argument("-host", "--host", type=str, required=False, default="127.0.0.1", help="Bind socket to this host. Use 0.0.0.0 to make the server available externally. ")
    
    parser.add_argument("-embedding_model", "--embedding_model", type=str, required=False, default="mixedbread-ai/mxbai-embed-large-v1", help="Embedder model")
    parser.add_argument("-chunk_size", "--chunk_size", type=int, required=False, default=1024, help="Chunk size")
    parser.add_argument("-collection_name", "--collection_name", type=str, required=False, default="radiopapers", help="Collection name")
    #parser.add_argument("-collection_names", "--collection_names", type=str, required=False, default="radiopapers,radiobooks,annreviews", help="Comma-separated list of Qdrant collection names to query across (overrides --collection_name)")
    parser.add_argument("-collection_names", "--collection_names", type=str, required=False, default="radiopapers,radiobooks,annreviews,solar-living-reviews", help="Comma-separated list of Qdrant collection names to load at startup. Requests can dynamically select a subset.")
    parser.add_argument("-similarity_thr", "--similarity_thr", type=float, required=False, default=0.5, help="Similarity threshold")
    parser.add_argument("-llm", "--llm", type=str, required=False, default="", help="LLM model name")
    parser.add_argument("-llm_url", "--llm_url", type=str, required=False, default="http://localhost:11434", help="LLM ollama url")
    parser.add_argument("-llm_ctx_window", "--llm_ctx_window", type=int, required=False, default=4096, help="LLM context window")
    parser.add_argument("-llm_timeout", "--llm_timeout", type=int, required=False, default=120, help="LLM response timeout in seconds")
    parser.add_argument(
        "-llm_keep_alive",
        "--llm_keep_alive",
        type=str,
        required=False,
        default="-1m",
        help=(
            "Ollama keep_alive option controlling how long the model stays loaded "
            "after a request. Examples: 0s unloads immediately after each request; "
            "5m keeps it loaded for 5 minutes; 1h keeps it loaded for 1 hour; "
            "a negative duration with unit, e.g. -1m, keeps it loaded indefinitely. "
        ),
    )
    parser.add_argument("--llm_thinking", dest="llm_thinking", action='store_true',help='Enable LLM thinking (default=False)')
    parser.set_defaults(llm_thinking=False)
    parser.add_argument("-qdrant_url", "--qdrant_url", type=str, required=False, default="http://localhost:6333", help="QDRant URL")
    parser.add_argument("-num_queries", "--num_queries", type=int, required=False, default=1, help="For multi-source aggregation, set to 1 to disable extra augmented queries")
    
    parser.add_argument("--llm_backend", type=str, default="ollama", choices=["ollama", "vllm", "vllm-openai"], help="LLM backend: ollama, vllm (local Python), or vllm-openai (OpenAI-compatible vLLM server)")
    parser.add_argument("--llm_api_key", type=str, default="dummy", help="API key for OpenAI-compatible backends (vLLM server can use a dummy value if auth is disabled or arbitrary token if required)")
    parser.add_argument("--llm_max_new_tokens", type=int, default=512, help="Maximum number of generated tokens")
    parser.add_argument("--llm_temperature", type=float, default=0.1, help="Generation temperature") 
    parser.add_argument("--llm_tensor_parallel_size", type=int, default=1, help="Tensor parallel size for local vLLM backend")
    parser.add_argument("--llm_is_chat_model", dest="llm_is_chat_model", action="store_true", help="Use chat-completions style API for OpenAI-compatible backends")
    parser.set_defaults(llm_is_chat_model=True)
    parser.add_argument("--llm_is_function_calling_model", dest="llm_is_function_calling_model", action="store_true", help="Whether the backend supports tool/function calling")
    parser.set_defaults(llm_is_function_calling_model=False)
 
    parser.add_argument("--retrieval_only_no_llm", dest="retrieval_only_no_llm", action="store_true", help="Start retrieval service without requiring an LLM. /api/retrieve works; /api/search may be disabled.")
    parser.set_defaults(retrieval_only_no_llm=False)
    parser.add_argument("--max_retrieve_top_k", type=int, default=100, help="Maximum top-k accepted by /api/retrieve.") 

    logger.info("Parsing arguments ...")
    args = parser.parse_args()

    return args

#############################################
##     MAIN
#############################################
def main():
    """ Main method """

    # - Parse args
    logger.info("Loading args ...")
    args= load_args()
    
    # - Set single- or multi-collection
    multi_mode = False
    collections = None
    if args.collection_names:
        collections = [c.strip() for c in args.collection_names.split(",") if c.strip()]
        multi_mode = len(collections) > 0

    # - Load model
    #logger.info(f"Loading model {args.llm} ...")
    #llm = Ollama(
    #  model=args.llm,
    #  base_url=args.llm_url,
    #  request_timeout=args.llm_timeout,
    #  context_window=args.llm_ctx_window,
    #  keep_alive=args.llm_keep_alive,
    #  thinking=args.llm_thinking,
    #  #additional_kwargs={
    #  #    "num_ctx": 4096,
    #  #    "num_batch": 128
    #  #}
    #)

    logger.info(f"Loading model {args.llm} with backend {args.llm_backend} ...")
    llm = build_llm(args)

    # - Create RAG
    logger.info(f"Creating RAG served by previously loaded model ...")
    rag = RAG(
        llm=llm,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        collection_name=args.collection_name,
        qdrant_url=args.qdrant_url,
        collection_names=collections
    )

    index = None
    indices = {}
    
    if multi_mode:
        logger.info("RAG: building multi-collection indices ...")
        indices = rag.qdrant_indices()
        logger.info(f"indices: {list(indices.keys())}")
    else:
        logger.info("RAG indexing (single collection) ...")
        index = rag.qdrant_index()
        logger.info(f"index: {index}")

    # - Create web app
    logger.info("Creating web app ...")
    app = FastAPI()

    # - Create routes
    @app.get("/")
    def root():
        return {"message": "Research RAG"}


    query_instr = "You can only answer based on the provided context. If a response cannot be formed strictly using the context, politely say you don't have knowledge about that topic"

    #######################################
    ##     API SEARCH
    #######################################    
    @app.post("/api/search", response_model=Response, status_code=200)
    def search(query: Query):
    
        logger.info(f"Received query: {query}")
    
        # - Return if LLM was not defined
        if rag.llm is None:
            return Response(
                status=-1,
                search_result="/api/search requires an LLM. Start without --retrieval_only_no_llm.",
                sources=[],
                content_found=False,
            )

        # - Resolve requested collections
        try:
            available_collections = list(indices.keys()) if multi_mode else [args.collection_name]

            requested_collections = resolve_requested_collections(
                requested_domain=query.domain,
                requested_collections=query.collections,
                default_collections=collections,
                available_collections=available_collections,
            )

            logger.info(
                "Resolved search collections",
                domain=query.domain,
                requested_collections=requested_collections,
                available_collections=available_collections,
            )

        except Exception as e:
            logger.error(f"Failed to resolve requested collections (err={str(e)})!")
            return Response(
                status=-1,
                search_result=f"Failed to resolve requested collections: {str(e)}",
                sources=[],
                content_found=False,
            )

        # - Select active indices without rebuilding them
        active_index = index
        active_indices = indices
        active_multi_mode = multi_mode

        if multi_mode:
            active_indices = {
                cname: idx
                for cname, idx in indices.items()
                if cname in requested_collections
            }

            if not active_indices:
                return Response(
                    status=-1,
                    search_result=f"No requested collections are available: {requested_collections}",
                    sources=[],
                    content_found=False,
                )

            active_multi_mode = len(active_indices) > 1
            active_index = None if active_multi_mode else next(iter(active_indices.values()))

        # - Build query engine    
        try:
            query_engine = build_query_engine(
                #index=index if not multi_mode else None,
                #indices=indices if multi_mode else None,
                index=active_index if not active_multi_mode else None,
                indices=active_indices if active_multi_mode else None,
                #multi_mode=multi_mode,
                multi_mode=active_multi_mode,
                similarity_top_k=query.similarity_top_k,
                #similarity_thr=args.similarity_thr,
                similarity_thr=args.similarity_thr if query.similarity_thr is None else query.similarity_thr,
                #num_queries=args.num_queries,
                num_queries=args.num_queries if query.num_queries is None else query.num_queries
            )
            
        except Exception as e:
            logger.error(f"Failed to retrieve query engine (err={str(e)})!")
            err_resp = Response(
                status=-1,
                search_result="Failed to retrieve query engine",
                sources=[],
                content_found=False,
            )
            return err_resp
            
        # - Run query
        logger.info("Querying engine ...")
        try:
            response = query_engine.query(query.query + "\n\n" + query_instr)
        except Exception as e:
            logger.error(f"Failed to run query engine (err={str(e)})!")
            err_resp= Response(status=-1, search_result="Failed to query engine", sources=[], content_found=False)
            return err_resp
            
        # - Parsing response
        logger.info(f"Parsing response: {response} ...")
        try:
            # - Retrieve the generated text
            response_text = getattr(response, "response", None) or getattr(response, "text", "")
            response_text = str(response_text).strip()
            print("response_text")
            print(response_text)

            # - Retrieve the metadata dict
            response_meta = response.metadata
            print("response_meta")
            print(response_meta)

            # - Set flag indicating if no content was found (given similarity threshold)
            content_found= len(response.source_nodes) > 0
            print(f"content_found? {content_found}")

            # - Retrieve the chunks with similarity scores
            response_sources = sources_from_nodes(getattr(response, "source_nodes", []))
            print("response_sources")
            print(response_sources)

            # - Modify the message in case of nothing found
            if not content_found and response_text=="Empty Response":
                response_text= "No contents found"

            # - Set final response
            response_object = Response(
                search_result=response_text,
                sources=response_sources,
                status=0,
                content_found=content_found
            )
        except Exception as e:
            logger.error(f"Failed to parse response (err={str(e)})!")
            err_resp = Response(
                status=-1,
                search_result="Failed to parse response",
                sources=[],
                content_found=False,
            )
            return err_resp

        return response_object


    
    #######################################
    ##     API RETRIEVE
    #######################################  
    @app.post("/api/retrieve", response_model=RetrieveResponse, status_code=200)
    def retrieve(req: RetrieveRequest):
        """ Retrieve endpoint """
        logger.info(f"Received retrieval request: {req}")

        try:
            #requested_collections = req.collections or collections
            #if requested_collections:
            #    requested_collections = [
            #        c.strip()
            #        for c in requested_collections
            #        if isinstance(c, str) and c.strip()
            #    ]

            available_collections = list(indices.keys()) if multi_mode else [args.collection_name]
            requested_collections = resolve_requested_collections(
                requested_domain=None,
                requested_collections=req.collections,
                default_collections=collections,
                available_collections=available_collections,
            )

            # If the request specifies a subset of collections, build a temporary
            # subset view over already-loaded indices. This does not rebuild indices.
            active_indices = indices
            active_multi_mode = multi_mode

            if requested_collections and multi_mode:
                active_indices = {
                    cname: idx
                    for cname, idx in indices.items()
                    if cname in requested_collections
                }

                if not active_indices:
                    return RetrieveResponse(
                        status=-1,
                        message=f"No requested collections are available: {requested_collections}",
                        content_found=False,
                        documents=[],
                        debug={
                            "requested_collections": requested_collections,
                            "available_collections": list(indices.keys()),
                        },
                    )

                active_multi_mode = len(active_indices) > 1

                if len(active_indices) == 1:
                    active_index = next(iter(active_indices.values()))
                else:
                    active_index = None
            else:
                active_index = index if not multi_mode else None

            top_k = req.similarity_top_k or 8
            top_k = min(top_k, args.max_retrieve_top_k)
            thr = args.similarity_thr if req.similarity_thr is None else req.similarity_thr
            nq = args.num_queries if req.num_queries is None else req.num_queries

            source_nodes = retrieve_source_nodes(
                query_text=req.query,
                index=active_index,
                indices=active_indices if active_multi_mode else None,
                multi_mode=active_multi_mode,
                similarity_top_k=top_k,
                similarity_thr=thr,
                num_queries=nq,
            )

            sources = sources_from_nodes(source_nodes)

            documents = [
                source_to_retrieved_document(source)
                for source in sources
            ]

            if not req.include_text:
                for doc in documents:
                    doc.text = ""
                    doc.metadata.pop("text", None)

            return RetrieveResponse(
                status=0,
                message="ok",
                content_found=len(documents) > 0,
                documents=documents,
                debug={
                    "query": req.query,
                    "similarity_top_k": top_k,
                    "similarity_thr": thr,
                    "num_queries": nq,
                    "multi_mode": active_multi_mode,
                    "requested_collections": requested_collections,
                    #"available_collections": list(indices.keys()) if multi_mode else [args.collection_name],
                    "available_collections": available_collections,
                    "returned": len(documents),
                },
            )

        except Exception as e:
            logger.error(f"Failed retrieval request (err={str(e)})!")
            return RetrieveResponse(
                status=-1,
                message=f"Failed retrieval request: {str(e)}",
                content_found=False,
                documents=[],
                debug={},
            )

    #####################################
    ##      RUN SERVER
    #####################################
    # - Running server
    logger.info("Running app ...")
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
