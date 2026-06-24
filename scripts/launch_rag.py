#!/usr/bin/env python
# -*- coding: utf-8 -*-

##########################
##   IMPORT MODULES
##########################
# - import standard modules
import os
import sys
import argparse
from typing import Optional, Any
import structlog
import re

# - Pydantic
from pydantic import BaseModel, Field

# - Import OLLAMA
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.postprocessor import SimilarityPostprocessor
try:
    from llama_index.core.postprocessor import SentenceTransformerRerank
except Exception:
    SentenceTransformerRerank = None    
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import PydanticSingleSelector, PydanticMultiSelector, LLMMultiSelector
from llama_index.core.selectors.embedding_selectors import EmbeddingSingleSelector
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import PromptTemplate
from llama_index.core.schema import NodeWithScore, QueryBundle
try:
    from llama_index.core.postprocessor.types import BaseNodePostprocessor
except:
    from llama_index.core.postprocessor import BaseNodePostprocessor

# - Import FastAPI
import uvicorn
from fastapi import FastAPI

# - Import qdrant
import qdrant_client

logger = structlog.get_logger()

#############################
##    HELPERS
#############################
def json_safe_value(value):
    """Convert numpy/scalar-like values to JSON-serializable Python values."""

    if value is None:
        return None

    # numpy scalar values, e.g. np.float32, np.float64, np.int64
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    return value


def json_safe_score(score):
    """Convert retrieval score to native Python float or None."""

    if score is None:
        return None

    try:
        return float(score)
    except Exception:
        return None


def resolve_score_type(args=None):
	"""Return active scoring mode."""

	if args is not None and getattr(args, "reranker", "none") != "none":
		return "reranker"

	return "similarity"


def extract_similarity_score(source: dict, score_type: str):
	"""Extract original vector similarity score."""

	metadata = source.get("metadata") or {}

	if score_type == "reranker":
		return json_safe_score(
			metadata.get("similarity_score")
			or source.get("similarity_score")
		)

	return json_safe_score(source.get("score"))


def extract_rerank_score(source: dict, score_type: str):
	"""Extract reranker score, if reranking is active."""

	if score_type != "reranker":
		return None

	return json_safe_score(source.get("score"))

def preserve_source_scores(sources: list[dict], score_type: str):
	"""Expose similarity and reranker scores while keeping score as similarity."""

	output_sources = []

	for source in sources or []:
		source = dict(source)
		metadata = dict(source.get("metadata") or {})

		similarity_score = extract_similarity_score(source, score_type=score_type)
		rerank_score = extract_rerank_score(source, score_type=score_type)

		source["similarity_score"] = similarity_score
		source["rerank_score"] = rerank_score
		source["score_type"] = score_type

		# Keep legacy frontend score as vector similarity.
		source["score"] = similarity_score

		source["metadata"] = {
			str(k): json_safe_value(v)
			for k, v in metadata.items()
		}

		output_sources.append(source)

	return output_sources


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
    #if "radioimg-arxiv-dataset" in fp or fn.endswith(".pdf"):
    if "radioimg-arxiv-dataset" in fp or "/radiopapers/" in fp:
        return "radiopapers"
    if "/solar-papers/" in fp:
        return "solar-papers"
    if "/exoplanets-papers/" in fp:
        return "exoplanets-papers"
    if "/solar-living-reviews/" in fp:
        return "solar-living-reviews"
		
    #return None
    return md.get("collection") or md.get("collection_name") or md.get("_collection_name")

#############################
##    RAG CLASSES
#############################
collection_descriptions = {
    "radiopapers": "Scientific papers stored in ArXiV repository with subject keywords related to radio astronomy.",
    "radiobooks": "Textbooks and monographs related to radio astronomy.",
    "annreviews": "Annual Reviews articles and review papers relevant to astronomy and astrophysics.",
    "solar-papers": "Scientific papers related to solar physics, solar activity, solar flares, CMEs, and heliophysics.",
    "solar-living-reviews": "Springer Living Reviews in Solar Physics articles",
    "exoplanets-papers": "Scientific papers related to exoplanets, planetary systems, atmospheres, detection methods, and characterization.",
}
DOMAIN_COLLECTIONS = {
    "radio": ["radiopapers", "radiobooks", "annreviews"],
    "solar": ["solar-papers", "solar-living-reviews", "annreviews"],
    "exoplanets": ["exoplanets-papers", "annreviews"],
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
    
    # - Final number of chunks used for synthesis
    similarity_top_k: Optional[int] = Field(default=5, ge=1, le=50)
    
    # - Number of chunks initially retrieved before final truncation/reranking
    retrieval_top_k: Optional[int] = Field(default=None, ge=1, le=200)
    
    domain: Optional[str] = None
    collections: Optional[list[str]] = None
    similarity_thr: Optional[float] = None
    num_queries: Optional[int] = None
    response_mode: Optional[str] = None

class Response(BaseModel):
    search_result: str
    content_found: bool
    status: int
    sources: list
    
class RetrieveRequest(BaseModel):
    query: str
    
    # - Final number of chunks returned to the calle
    similarity_top_k: Optional[int] = Field(default=8, ge=1, le=100)
    
    # - Number of chunks initially retrieved before final truncation/reranking.
    retrieval_top_k: Optional[int] = Field(default=None, ge=1, le=500)
    
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
    similarity_score: Optional[float] = None
    rerank_score: Optional[float] = None
    score_type: Optional[str] = None
    collection: Optional[str] = None
    metadata: dict = Field(default_factory=dict)

class RetrieveResponse(BaseModel):
    status: int
    message: str = "ok"
    content_found: bool
    documents: list[RetrievedDocument] = Field(default_factory=list)
    debug: dict = Field(default_factory=dict)    

class CollectionSummary(BaseModel):
    collection: str
    points_count: int = 0
    estimated_documents: int = 0
    year_min: Optional[int] = None
    year_max: Optional[int] = None
    status: int = 0
    message: str = "ok"

class CollectionsSummaryResponse(BaseModel):
    status: int
    message: str = "ok"
    collections: list[CollectionSummary] = Field(default_factory=list)


class TopKPostprocessor(BaseNodePostprocessor):
    """Keep only the first top-k nodes after retrieval/filtering/reranking."""

    top_k: int = 5

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle=None,
        query_str: Optional[str] = None,
    ):
        return nodes[: self.top_k]

class PreserveOriginalScorePostprocessor(BaseNodePostprocessor):
	"""Store the original vector retrieval score before reranking."""

	def _postprocess_nodes(
		self,
		nodes: list[NodeWithScore],
		query_bundle=None,
		query_str: Optional[str] = None,
	):
		for sn in nodes or []:
			if sn.node.metadata is None:
				sn.node.metadata = {}

			if "similarity_score" not in sn.node.metadata:
				sn.node.metadata["similarity_score"] = json_safe_score(sn.score)

		return nodes

class DocumentDedupPostprocessor(BaseNodePostprocessor):
    """Keep only the top-ranked chunk(s) per source document."""

    keep_per_document: int = 1

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle=None,
        query_str: Optional[str] = None,
    ):
        seen_counts = {}
        deduped_nodes = []

        for sn in nodes or []:
            key = document_key_from_node(sn)
            count = seen_counts.get(key, 0)

            if count >= self.keep_per_document:
                continue

            seen_counts[key] = count + 1
            deduped_nodes.append(sn)

        return deduped_nodes
		
###################################
####    RAG HELPERS
###################################
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


def _extract_year_from_arxiv_like_value(value: Any) -> Optional[int]:
    """
    Extract year from arXiv-like identifiers.

    Handles:
    - modern IDs: 2305.12345, 0704.0001, 9912.1234
    - old-style IDs: astro-ph/0012345, astro-ph/0601234
    - filenames/URLs containing those IDs
    """

    if value is None or not isinstance(value, str):
        return None

    s = value.strip()
    if not s:
        return None

    s = re.sub(r"^arxiv:\s*", "", s, flags=re.IGNORECASE)


    # Old-style arXiv IDs, e.g. astro-ph/0012345, astro-ph/0601234.
    # First two digits after slash are year.
    m_old = re.search(
        r"(?:astro-ph|hep-ph|hep-th|gr-qc|quant-ph|cond-mat|math|cs|physics|nucl-th|nucl-ex|stat|q-bio|q-fin|nlin)/(\d{2})\d{5}",
        s,
        flags=re.IGNORECASE,
    )
    if m_old:
        yy = int(m_old.group(1))
        year = 1900 + yy if yy >= 90 else 2000 + yy
        if 1991 <= year <= 2100:
            return year

    # Old numeric arXiv-like filenames/IDs, e.g. 0505148v1.pdf, 0210040v1.pdf.
    # These are common for old arXiv papers when the category prefix was stripped.
    m_old_numeric = re.search(
        r"(?<!\d)(\d{2})\d{5}(?:v\d+)?(?:\.pdf)?(?!\d)",
        s,
        flags=re.IGNORECASE,
    )
    if m_old_numeric:
        yy = int(m_old_numeric.group(1))
        year = 1900 + yy if yy >= 90 else 2000 + yy
        if 1991 <= year <= 2100:
            return year

    # Modern arXiv IDs: YYMM.NNNN or YYMM.NNNNN, optionally with vN.
    m = re.search(r"(?<!\d)(\d{2})(\d{2})\.\d{4,5}(?:v\d+)?(?!\d)", s)
    if m:
        yy = int(m.group(1))
        mm = int(m.group(2))

        if not 1 <= mm <= 12:
            return None

        year = 1900 + yy if yy >= 90 else 2000 + yy

        if 1991 <= year <= 2100:
            return year

    return None


def _extract_year_from_metadata(md: dict[str, Any]) -> Optional[int]:
    """Extract a reasonable publication year from common metadata fields."""

    # 1. Explicit year/date-like metadata fields.
    candidates = [
        md.get("year"),
        md.get("pub_year"),
        md.get("publication_year"),
        md.get("date"),
        md.get("published"),
        md.get("publication_date"),
        md.get("created"),
        md.get("updated"),
    ]

    for value in candidates:
        if value is None:
            continue

        if isinstance(value, int):
            if 1500 <= value <= 2100:
                return value

        if isinstance(value, float):
            year = int(value)
            if 1500 <= year <= 2100:
                return year

        if isinstance(value, str):
            m = re.search(r"(15|16|17|18|19|20|21)\d{2}", value)
            if m:
                year = int(m.group(0))
                if 1500 <= year <= 2100:
                    return year

    # 2. arXiv-like metadata fields and filenames.
    arxiv_candidates = [
        md.get("arxiv_id"),
        md.get("arXiv"),
        md.get("arxiv"),
        md.get("eprint"),
        md.get("identifier"),
        md.get("file_name"),
        md.get("file_path"),
        md.get("url"),
        md.get("arxiv_abs_url"),
        md.get("arxiv_pdf_url"),
    ]

    for value in arxiv_candidates:
        year = _extract_year_from_arxiv_like_value(value)
        if year is not None:
            return year

    return None


def _document_key_from_metadata(md: dict[str, Any], fallback_id: Any) -> str:
    """
    Estimate a unique source document key from chunk metadata.

    This is used to estimate number of papers/books/reviews from Qdrant chunks.
    """

    for key in [
        "doi",
        "arxiv_id",
        "bibcode",
        "isbn",
        "issn",
        "source_id",
        "document_id",
        "doc_id",
        "file_name",
        "file_path",
        "title",
        "paper_title",
        "document_title",
    ]:
        value = md.get(key)
        if isinstance(value, str) and value.strip():
            return f"{key}:{value.strip()}"

    return f"point:{fallback_id}"


def document_key_from_node(sn: NodeWithScore) -> str:
    """
    Build a stable document-level key for deduplication.

    The key should identify the source document, not the individual chunk.
    """

    md = sn.node.metadata or {}

    for key in [
        "source_id",
        "doi",
        "arxiv_id",
        "bibcode",
        "isbn",
        "document_hash",
        "document_id",
        "doc_id",
        "file_path",
        "filepath",
        "file_name",
        "title",
        "paper_title",
        "document_title",
    ]:
        value = md.get(key)

        if isinstance(value, str) and value.strip():
            return f"{key}:{value.strip().lower()}"

        if isinstance(value, list) and value:
            return f"{key}:{str(value[0]).strip().lower()}"

    return f"node:{_node_id(sn) or sn.node.node_id}"

def summarize_qdrant_collection(
    client: qdrant_client.QdrantClient,
    collection_name: str,
    max_scroll_points: int = 20000,
    page_size: int = 256,
) -> CollectionSummary:
    """
    Summarize a Qdrant collection.

    points_count is the Qdrant point count.
    estimated_documents is based on unique document-like metadata keys.
    year range is estimated from metadata year/date fields.
    """

    try:
        info = client.get_collection(collection_name=collection_name)
        points_count = int(info.points_count or 0)

        doc_keys = set()
        years = []

        offset = None
        scanned = 0

        while scanned < max_scroll_points:
            records, offset = client.scroll(
                collection_name=collection_name,
                limit=page_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            if not records:
                break

            for rec in records:
                payload = rec.payload or {}

                # LlamaIndex/Qdrant payloads can store metadata either directly
                # or under nested metadata-like keys depending on ingestion version.
                md = payload.get("metadata") or payload.get("_node_content") or payload

                if isinstance(md, str):
                    # Some LlamaIndex payloads serialize node content as JSON string.
                    # Keep this conservative to avoid hard failures.
                    md = payload

                if not isinstance(md, dict):
                    md = payload

                doc_keys.add(_document_key_from_metadata(md, rec.id))

                year = _extract_year_from_metadata(md)
                if year is not None:
                    years.append(year)

            scanned += len(records)

            if offset is None:
                break

        return CollectionSummary(
            collection=collection_name,
            points_count=points_count,
            estimated_documents=len(doc_keys),
            year_min=min(years) if years else None,
            year_max=max(years) if years else None,
            status=0,
            message="ok",
        )

    except Exception as e:
        logger.error(f"Failed to summarize collection {collection_name} (err={str(e)})!")
        return CollectionSummary(
            collection=collection_name,
            points_count=0,
            estimated_documents=0,
            year_min=None,
            year_max=None,
            status=-1,
            message=str(e),
        )

def source_to_retrieved_document(
    source: dict,
    score_type: str = "similarity",
) -> RetrievedDocument:

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
    
    similarity_score = extract_similarity_score(source, score_type=score_type)
    rerank_score = extract_rerank_score(source, score_type=score_type)

    # Keep frontend-facing score as the original vector similarity score.
    display_score = similarity_score

    return RetrievedDocument(
        doc_id=str(doc_id),
        title=title,
        text=str(source.get("text") or ""),
        #score=json_safe_score(source.get("score")),
        score=display_score,
        similarity_score=similarity_score,
	      rerank_score=rerank_score,
	      score_type=score_type,
        collection=collection,
        #metadata=metadata,
        metadata={
            str(k): json_safe_value(v)
            for k, v in metadata.items()
        },
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
        elif doctype in {
            "living-review-solar-physics",
		        "solar-living-review",
		        "living-review",
	      }:
		        source = get_solar_living_review_metadata(sn, md)    
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
    #source["metadata"] = dict(md)
    source["metadata"] = {
        str(k): json_safe_value(v)
        for k, v in dict(md).items()
    }

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
        #"score": sn.score,                     # similarity score
        "score": json_safe_score(sn.score),
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
        #"score": sn.score,                     # similarity score
        "score": json_safe_score(sn.score),
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
        #"score": sn.score,                     # similarity score
        "score": json_safe_score(sn.score),
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

def get_solar_living_review_metadata(sn, md):
    """Get Springer Living Reviews in Solar Physics metadata."""

    source = {
        "doctype": "solar-living-review",
        "node_id": sn.node.node_id,
        #"score": sn.score,
        "score": json_safe_score(sn.score),
        "file_path": _metadata_file_path(md),
        "file_name": md.get("file_name") or md.get("pdf_filename"),
        "page_label": md.get("page_label"),
        "title": md.get("title"),
        "authors": md.get("authors"),
        "first_author": md.get("first_author"),
        "journal": md.get("journal") or md.get("source_name"),
        "journal_short": md.get("journal_short"),
        "publisher": md.get("publisher"),
        "volume": md.get("volume"),
        "issue": md.get("issue"),
        "article_number": md.get("article_number"),
        "year": md.get("year"),
        "published_date": md.get("published_date"),
        "doi": md.get("doi"),
        "url": md.get("url"),
        "pdf_url": md.get("pdf_url"),
        "html_url": md.get("html_url"),
        "download_url": md.get("download_url") or md.get("pdf_url"),
        "source_id": md.get("source_id"),
        "source_family": md.get("source_family"),
        "source_type": md.get("source_type"),
        "text": _node_text(sn),
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


def build_text_qa_template():
    """Build the QA prompt used by LlamaIndex during answer synthesis."""

    return PromptTemplate(
        "You are a scientific Retrieval-Augmented Generation assistant.\n"
        "You must answer using only the provided context.\n"
        "If the context does not contain enough information to answer the question, "
        "say that the provided sources do not contain enough information.\n\n"
        "Context:\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n\n"
        "Question:\n"
        "{query_str}\n\n"
        "Answer:"
    )

def build_refine_template():
    """Build the refine prompt used by response modes that refine an existing answer."""

    return PromptTemplate(
        "You are refining a scientific answer using additional retrieved context.\n"
        "The existing answer is shown below.\n"
        "Only improve the answer if the new context supports the improvement.\n"
        "If the new context is not useful, keep the existing answer unchanged.\n\n"
        "Existing answer:\n"
        "---------------------\n"
        "{existing_answer}\n"
        "---------------------\n\n"
        "New context:\n"
        "---------------------\n"
        "{context_msg}\n"
        "---------------------\n\n"
        "Question:\n"
        "{query_str}\n\n"
        "Refined answer:"
    )



def resolve_top_k(
    final_top_k: Optional[int],
    retrieval_top_k: Optional[int],
    default_final_top_k: int,
    default_retrieval_top_k: int,
    max_final_top_k: int,
    max_retrieval_top_k: int,
):
    """
    Resolve final and retrieval top-k values.

    final_top_k controls how many chunks are finally used/returned.
    retrieval_top_k controls how many candidate chunks are initially retrieved.
    """

    resolved_final_top_k = final_top_k or default_final_top_k
    resolved_retrieval_top_k = retrieval_top_k or default_retrieval_top_k

    resolved_final_top_k = max(1, min(resolved_final_top_k, max_final_top_k))
    resolved_retrieval_top_k = max(1, min(resolved_retrieval_top_k, max_retrieval_top_k))

    # Retrieval candidate pool should never be smaller than final top-k.
    resolved_retrieval_top_k = max(resolved_retrieval_top_k, resolved_final_top_k)

    return resolved_final_top_k, resolved_retrieval_top_k


def truncate_nodes(nodes, top_k: Optional[int]):
    """Keep only the first top_k nodes."""
    if top_k is None:
        return nodes
    return list(nodes or [])[:top_k]

def build_node_postprocessors(
    similarity_thr: Optional[float],
    final_top_k: int,
    args=None,
):
    """
    Build the ordered postprocessing chain.

    Order:
    1. Similarity threshold filtering
    2. Optional reranking
    3. Final top-k truncation
    """

    node_postprocessors = []

    # - Add similarity processor
    if similarity_thr is not None:
        node_postprocessors.append(
            SimilarityPostprocessor(similarity_cutoff=similarity_thr)
        )

    # - Add reranker processor
    if args is not None and getattr(args, "reranker", "none") == "sentence_transformer":
        if SentenceTransformerRerank is None:
            raise RuntimeError(
                "SentenceTransformerRerank is not available in this LlamaIndex installation. "
                "Install the required LlamaIndex postprocessor dependencies or set --reranker none."
            )

        rerank_top_n = getattr(args, "rerank_top_n", final_top_k)
        rerank_top_n = max(1, int(rerank_top_n))
        rerank_top_n = max(rerank_top_n, final_top_k)

        node_postprocessors.append(
            PreserveOriginalScorePostprocessor()
	      )

        node_postprocessors.append(
            SentenceTransformerRerank(
                model=getattr(args, "reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
                top_n=rerank_top_n,
            )
        )

    # - Add doc deduplication processor
    if args is not None and getattr(args, "deduplicate_documents", True):
        dedup_keep_per_document = getattr(args, "dedup_keep_per_document", 1)

        try:
            dedup_keep_per_document = int(dedup_keep_per_document)
        except Exception:
            dedup_keep_per_document = 1

        node_postprocessors.append(
            DocumentDedupPostprocessor(keep_per_document=max(1, dedup_keep_per_document))
        )

    # - Add top-k post processor
    node_postprocessors.append(
        TopKPostprocessor(top_k=final_top_k)
    )

    return node_postprocessors


def postprocess_retrieved_nodes(
    nodes: list[NodeWithScore],
    query_text: str,
    similarity_thr: Optional[float],
    final_top_k: int,
    args=None,
):
    """
    Apply the common postprocessing chain used by both /api/search and /api/retrieve.

    Current chain:
    1. Similarity threshold filtering
    2. Optional original-score preservation
    3. Optional reranking
    4. Final top-k truncation
    """

    node_postprocessors = build_node_postprocessors(
        similarity_thr=similarity_thr,
        final_top_k=final_top_k,
        args=args,
    )

    query_bundle = QueryBundle(query_str=query_text)

    for postprocessor in node_postprocessors:
        nodes = postprocessor.postprocess_nodes(
            nodes,
            query_bundle=query_bundle,
        )

    return nodes


def describe_node_postprocessors(
	similarity_thr: Optional[float],
	final_top_k: int,
	args=None,
):
	"""Return postprocessor class names for debug output."""

	return [
		type(pp).__name__
		for pp in build_node_postprocessors(
			similarity_thr=similarity_thr,
			final_top_k=final_top_k,
			args=args,
		)
	]

def build_query_engine(
    index=None,
    indices: dict | None = None,
    multi_mode: bool = False,
    #similarity_top_k: int = 5,
    retrieval_top_k: int = 30,
    final_top_k: int = 5,
    similarity_thr: float = 0.5,
    num_queries: int = 1,
    response_mode: str = "compact",
    args=None,
):
    """Build query engine."""

    text_qa_template = build_text_qa_template()
    refine_template = build_refine_template()
    node_postprocessors = build_node_postprocessors(
        similarity_thr=similarity_thr,
        final_top_k=final_top_k,
        args=args,
    )

    if not multi_mode:
        return index.as_query_engine(
            #similarity_top_k=similarity_top_k,
            similarity_top_k=retrieval_top_k,
            output=Response,
            response_mode=response_mode,
            include_metadata=True,
            text_qa_template=text_qa_template,
            refine_template=refine_template,
            node_postprocessors=node_postprocessors,
            verbose=True,
        )

    fusion = build_fusion_retriever(
        indices=indices,
        #similarity_top_k=similarity_top_k,
        similarity_top_k=retrieval_top_k,
        num_queries=num_queries,
    )

    return RetrieverQueryEngine.from_args(
        fusion,
        response_mode=response_mode,
        include_metadata=True,
        output=Response,
        text_qa_template=text_qa_template,
        refine_template=refine_template,
        node_postprocessors=node_postprocessors,
    )


def retrieve_source_nodes(
    query_text: str,
    index=None,
    indices: dict | None = None,
    multi_mode: bool = False,
    #similarity_top_k: int = 8,
    retrieval_top_k: int = 30,
    final_top_k: int = 8,
    similarity_thr: float = 0.5,
    num_queries: int = 1,
    args=None,
):
    """ Retrieve source node """
    if not multi_mode:
        ##retriever = index.as_retriever(similarity_top_k=similarity_top_k)
        retriever = index.as_retriever(similarity_top_k=retrieval_top_k)
    else:
        retriever = build_fusion_retriever(
            indices=indices,
            #similarity_top_k=similarity_top_k,
            similarity_top_k=retrieval_top_k,
            num_queries=num_queries,
        )

    nodes = retriever.retrieve(query_text)

    #if similarity_thr is not None:
    #    nodes = [
    #        sn for sn in nodes
    #        if sn.score is None or sn.score >= similarity_thr
    #    ]
        
    #nodes = truncate_nodes(nodes, final_top_k)        

    return postprocess_retrieved_nodes(
        nodes=nodes,
        query_text=query_text,
        similarity_thr=similarity_thr,
        final_top_k=final_top_k,
        args=args,
    )


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
    parser.add_argument(
        "-collection_names",
        "--collection_names",
        type=str,
        required=False,
        default="radiopapers,radiobooks,annreviews,solar-living-reviews,solar-papers,exoplanets-papers",
        help=(
            "Comma-separated list of Qdrant collection names to load at startup. "
            "Requests can dynamically select a subset. Include the union of all "
            "collections used by the frontend domains."
        ),
    )
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
    parser.add_argument(
        "--retrieval_top_k",
        type=int,
        default=30,
        help=(
            "Number of chunks initially retrieved before final truncation or reranking. "
            "Use a larger value than similarity_top_k, e.g. 30."
        ),
    )
    parser.add_argument("--max_search_retrieval_top_k", type=int, default=100, help="Maximum retrieval_top_k accepted by /api/search.")
    parser.add_argument("--max_retrieve_retrieval_top_k", type=int, default=500, help="Maximum retrieval_top_k accepted by /api/retrieve.")

    parser.add_argument(
        "--response_mode",
        type=str,
        default="compact",
        choices=["compact", "refine", "tree_summarize", "simple_summarize"],
        help=(
            "LlamaIndex response synthesis mode. "
            "Use compact for concise QA, tree_summarize for broad summaries, "
            "and refine for careful source-by-source refinement."
        ),
    )

    parser.add_argument(
        "--append_query_instruction",
        dest="append_query_instruction",
        action="store_true",
        help=(
            "Legacy behavior: append the RAG instruction directly to the user query. "
            "Disabled by default because it can perturb retrieval embeddings."
        ),
    )
    parser.set_defaults(append_query_instruction=False)
    
    parser.add_argument(
        "--reranker",
        type=str,
        default="none",
        choices=["none", "sentence_transformer"],
        help=(
            "Optional reranker applied after vector retrieval and similarity filtering. "
            "Use 'sentence_transformer' to enable SentenceTransformerRerank."
        ),
    )

    parser.add_argument(
        "--reranker_model",
        type=str,
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="SentenceTransformer cross-encoder model used when --reranker sentence_transformer.",
    )

    parser.add_argument(
        "--rerank_top_n",
        type=int,
        default=5,
        help=(
            "Number of nodes kept by the reranker. "
            "Usually this should match or slightly exceed the final synthesis top-k."
        ),
    )
    
    parser.add_argument("--deduplicate_documents", dest="deduplicate_documents", action="store_true", help="Deduplicate retrieved chunks by source document after filtering/reranking and before final top-k truncation.")
    parser.add_argument("--no_deduplicate_documents", dest="deduplicate_documents", action="store_false", help="Disable document-level deduplication.")
    parser.set_defaults(deduplicate_documents=True)
    parser.add_argument("--dedup_keep_per_document", type=int, default=1, help="Number of chunks to keep per source document when deduplication is enabled.")
    
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

    # - Create Qdrant collection summary
    summary_qdrant_client = qdrant_client.QdrantClient(url=args.qdrant_url)
    COLLECTION_SUMMARY_CACHE = {}

    # - Create web app
    logger.info("Creating web app ...")
    app = FastAPI()

    # - Create routes
    @app.get("/")
    def root():
        return {"message": "Research RAG"}


    query_instr = "You can only answer based on the provided context. If a response cannot be formed strictly using the context, politely say you don't have knowledge about that topic"


    #######################################
    ##     API COLLECTION SUMMARIES
    #######################################
    @app.get("/api/collections/summary", response_model=CollectionsSummaryResponse, status_code=200)
    def collections_summary(refresh: bool = False):
        """Return lightweight collection summaries for the frontend landing page."""

        try:
            available_collections = list(indices.keys()) if multi_mode else [args.collection_name]

            summaries = []

            for cname in available_collections:
                if refresh or cname not in COLLECTION_SUMMARY_CACHE:
                    COLLECTION_SUMMARY_CACHE[cname] = summarize_qdrant_collection(
                        client=summary_qdrant_client,
                        collection_name=cname,
                    )

                summaries.append(COLLECTION_SUMMARY_CACHE[cname])

            return CollectionsSummaryResponse(
                status=0,
                message="ok",
                collections=summaries,
            )

        except Exception as e:
            logger.error(f"Failed to build collection summaries (err={str(e)})!")
            return CollectionsSummaryResponse(
                status=-1,
                message=str(e),
                collections=[],
            )

    #######################################
    ##     API COLLECTION YEAR DEBUG
    #######################################
    @app.get("/api/collections/{collection_name}/year_debug", status_code=200)
    def collection_year_debug(collection_name: str, limit: int = 2000):
        """
        Inspect inferred years for a Qdrant collection.

        Returns:
        - year histogram
        - sample records for min/max years
        - sample records where no year was found
        """

        try:
            records_seen = 0
            offset = None
            year_counts = {}
            no_year_samples = []
            year_samples = {}

            while records_seen < limit:
                records, offset = summary_qdrant_client.scroll(
                    collection_name=collection_name,
                    limit=min(256, limit - records_seen),
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )

                if not records:
                    break

                for rec in records:
                    payload = rec.payload or {}
                    md = _payload_to_metadata(payload) if "_payload_to_metadata" in globals() else payload

                    year = _extract_year_from_metadata(md)

                    sample = {
                        "point_id": str(rec.id),
                        "year": year,
                        "file_name": md.get("file_name"),
                        "file_path": md.get("file_path") or md.get("filepath"),
                        "arxiv_id": md.get("arxiv_id") or md.get("arXiv") or md.get("arxiv") or md.get("eprint") or md.get("identifier"),
                        "title": md.get("title") or md.get("paper_title") or md.get("document_title"),
                        "date": md.get("date") or md.get("published") or md.get("publication_date") or md.get("year"),
                        "metadata_keys": sorted(list(md.keys()))[:80],
                    }

                    if year is None:
                        if len(no_year_samples) < 10:
                            no_year_samples.append(sample)
                    else:
                        year_counts[year] = year_counts.get(year, 0) + 1
                        year_samples.setdefault(year, sample)

                records_seen += len(records)

                if offset is None:
                    break

            years = sorted(year_counts.keys())

            return {
                "status": 0,
                "collection": collection_name,
                "records_seen": records_seen,
                "year_counts": {str(y): year_counts[y] for y in years},
                "year_min": min(years) if years else None,
                "year_max": max(years) if years else None,
                "min_year_sample": year_samples.get(min(years)) if years else None,
                "max_year_sample": year_samples.get(max(years)) if years else None,
                "no_year_samples": no_year_samples,
            }

        except Exception as e:
            logger.error(f"Failed year debug for {collection_name} (err={str(e)})!")
            return {
                "status": -1,
                "collection": collection_name,
                "message": str(e),
            }

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
        response_mode = query.response_mode or args.response_mode
        
        final_top_k, retrieval_top_k = resolve_top_k(
            final_top_k=query.similarity_top_k,
            retrieval_top_k=query.retrieval_top_k,
            default_final_top_k=5,
            default_retrieval_top_k=args.retrieval_top_k,
            max_final_top_k=50,
            max_retrieval_top_k=args.max_search_retrieval_top_k,
        )
        
        logger.info(
            f"Building query engine with: response_mode={response_mode}, "
            f"final_top_k={final_top_k}, retrieval_top_k={retrieval_top_k}, "
            f"reranker={args.reranker}, reranker_model={args.reranker_model}, "
            f"rerank_top_n={args.rerank_top_n}"
        )
        
        try:    
            query_engine = build_query_engine(
                index=active_index if not active_multi_mode else None,
                indices=active_indices if active_multi_mode else None,
                multi_mode=active_multi_mode,
                #similarity_top_k=query.similarity_top_k,
                retrieval_top_k=retrieval_top_k,
                final_top_k=final_top_k,
                similarity_thr=args.similarity_thr if query.similarity_thr is None else query.similarity_thr,
                num_queries=args.num_queries if query.num_queries is None else query.num_queries,
                response_mode=response_mode,
                args=args,
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
            query_text = query.query
            if args.append_query_instruction:
                query_text = query.query + "\n\n" + query_instr

            response = query_engine.query(query_text)
            ###response = query_engine.query(query.query + "\n\n" + query_instr)
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
            
            score_type = resolve_score_type(args)
            response_sources = preserve_source_scores(
                response_sources,
                score_type=score_type,
            )

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

            #top_k = req.similarity_top_k or 8
            #top_k = min(top_k, args.max_retrieve_top_k)
            #thr = args.similarity_thr if req.similarity_thr is None else req.similarity_thr
            #nq = args.num_queries if req.num_queries is None else req.num_queries

            final_top_k, retrieval_top_k = resolve_top_k(
                final_top_k=req.similarity_top_k,
                retrieval_top_k=req.retrieval_top_k,
                default_final_top_k=8,
                default_retrieval_top_k=args.retrieval_top_k,
                max_final_top_k=args.max_retrieve_top_k,
                max_retrieval_top_k=args.max_retrieve_retrieval_top_k,
            )
            thr = args.similarity_thr if req.similarity_thr is None else req.similarity_thr
            nq = args.num_queries if req.num_queries is None else req.num_queries

            logger.info(f"Retrieving source nodes with: final_top_k={final_top_k}, retrieval_top_k={retrieval_top_k}, thr={thr}, nq={nq}")

            source_nodes = retrieve_source_nodes(
                query_text=req.query,
                index=active_index,
                indices=active_indices if active_multi_mode else None,
                multi_mode=active_multi_mode,
                #similarity_top_k=top_k,
                retrieval_top_k=retrieval_top_k,
                final_top_k=final_top_k,
                similarity_thr=thr,
                num_queries=nq,
                args=args,
            )

            score_type = resolve_score_type(args)

            sources = sources_from_nodes(source_nodes)

            #documents = [
            #    source_to_retrieved_document(source)
            #    for source in sources
            #]
            documents = [
                source_to_retrieved_document(source, score_type=score_type)
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
                    #"similarity_top_k": top_k,
                    "similarity_top_k": final_top_k,
                    "retrieval_top_k": retrieval_top_k,
                    "similarity_thr": thr,
                    "num_queries": nq,
                    "reranker": args.reranker,
                    "reranker_model": args.reranker_model if args.reranker != "none" else None,
                    "rerank_top_n": args.rerank_top_n if args.reranker != "none" else None,
                    "score_type": score_type,
                    "postprocessors": describe_node_postprocessors(
                        similarity_thr=thr,
                        final_top_k=final_top_k,
                        args=args,
                    ),
                    "deduplicate_documents": getattr(args, "deduplicate_documents", True),
                    "dedup_keep_per_document": getattr(args, "dedup_keep_per_document", 1),
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
