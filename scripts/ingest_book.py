#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import json
import structlog
from typing import List, Iterable, Optional, Any, Dict
import traceback

# ---- llama-index
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    Settings,
)
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.node_parser import SentenceSplitter

# ---- qdrant
import qdrant_client

logger = structlog.get_logger()

# ====== SafeEmbedder & helpers (unchanged from ingest_doc.py) ======
class SafeEmbedder(BaseEmbedding):
    _inner: Any

    def __init__(self, inner: Any):
        super().__init__()
        self._inner = inner

    @staticmethod
    def _coerce_one(x: Any) -> Optional[str]:
        if isinstance(x, bytes):
            try:
                x = x.decode("utf-8", errors="ignore")
            except Exception:
                return None
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
                fixed.append(" ")
            else:
                fixed.append(ok)
        if bad_idx:
            print(f"[SafeEmbedder] sanitized {len(bad_idx)} items; sample idx: {bad_idx[:5]}")
        return fixed

    def _delegate_text_batch(self, texts: List[str], **kwargs) -> List[List[float]]:
        inner = self._inner
        if hasattr(inner, "get_text_embedding_batch"):
            try:
                return inner.get_text_embedding_batch(texts, **kwargs)
            except TypeError:
                return inner.get_text_embedding_batch(texts)
        if hasattr(inner, "get_text_embedding"):
            return [inner.get_text_embedding(t) for t in texts]
        if hasattr(inner, "embed_documents"):
            try:
                return inner.embed_documents(texts, **kwargs)
            except TypeError:
                return inner.embed_documents(texts)
        if hasattr(inner, "encode"):
            return inner.encode(texts, convert_to_numpy=False)
        raise AttributeError("Inner embedder lacks a compatible batch method")

    def _delegate_text_single(self, text: str, **kwargs) -> List[float]:
        inner = self._inner
        if hasattr(inner, "get_text_embedding"):
            try:
                return inner.get_text_embedding(text, **kwargs)
            except TypeError:
                return inner.get_text_embedding(text)
        if hasattr(inner, "get_text_embedding_batch"):
            return self._delegate_text_batch([text], **kwargs)[0]
        if hasattr(inner, "embed_documents"):
            try:
                return inner.embed_documents([text], **kwargs)[0]
            except TypeError:
                return inner.embed_documents([text])[0]
        if hasattr(inner, "encode"):
            return inner.encode([text], convert_to_numpy=False)[0]
        raise AttributeError("Inner embedder lacks a compatible single-text method")

    def _delegate_query_single(self, query: str, **kwargs) -> List[float]:
        inner = self._inner
        if hasattr(inner, "get_query_embedding"):
            try:
                return inner.get_query_embedding(query, **kwargs)
            except TypeError:
                return inner.get_query_embedding(query)
        return self._delegate_text_single(query, **kwargs)

    def get_text_embedding_batch(self, texts: List[Any], **kwargs) -> List[List[float]]:
        texts = self._coerce_batch(texts)
        return self._delegate_text_batch(texts, **kwargs)

    def get_text_embedding(self, text: Any, **kwargs) -> List[float]:
        text = self._coerce_one(text) or " "
        return self._delegate_text_single(text, **kwargs)

    def get_query_embedding(self, query: Any, **kwargs) -> List[float]:
        query = self._coerce_one(query) or " "
        return self._delegate_query_single(query, **kwargs)

    async def _aget_text_embedding_batch(self, texts: List[Any], **kwargs) -> List[List[float]]:
        return self.get_text_embedding_batch(texts, **kwargs)

    async def _aget_text_embedding(self, text: Any, **kwargs) -> List[float]:
        return self.get_text_embedding(text, **kwargs)

    async def _aget_query_embedding(self, query: Any, **kwargs) -> List[float]:
        return self.get_query_embedding(query, **kwargs)


def embed_nodes_resilient(nodes, embed_model, batch_size=64):
    good_ids, good_embs, skipped = [], [], []

    def try_batch(batch_nodes):
        texts = [n.get_content(metadata_mode="none") for n in batch_nodes]
        return embed_model.get_text_embedding_batch(texts)

    def process(batch_nodes):
        if not batch_nodes:
            return
        try:
            embs = try_batch(batch_nodes)
            for n, e in zip(batch_nodes, embs):
                good_ids.append(n.node_id)
                good_embs.append(e)
        except Exception:
            if len(batch_nodes) == 1:
                skipped.append(getattr(batch_nodes[0], "node_id", getattr(batch_nodes[0], "id_", None)))
            else:
                mid = len(batch_nodes) // 2
                process(batch_nodes[:mid])
                process(batch_nodes[mid:])

    for i in range(0, len(nodes), batch_size):
        process(nodes[i:i + batch_size])

    return good_ids, good_embs, skipped


def build_index_from_vector_store(vs, storage_context):
    if hasattr(VectorStoreIndex, "from_vector_store"):
        return VectorStoreIndex.from_vector_store(vs, storage_context=storage_context)
    return VectorStoreIndex([], storage_context=storage_context)


# ====== metadata loading & normalization ======
def load_metadata_map(metadata_json: str, base_dir: Optional[str]) -> Dict[str, Dict[str, Any]]:
    """
    Returns a dict keyed by absolute file path -> normalized metadata dict.
    """
    with open(metadata_json, "r", encoding="utf-8") as f:
        items = json.load(f)

    if not isinstance(items, list):
        raise ValueError("metadata.json must contain a JSON array of objects")

    meta_by_path: Dict[str, Dict[str, Any]] = {}
    for i, raw in enumerate(items):
        if not isinstance(raw, dict):
            logger.warning("Skipping non-object entry at index %d in metadata.json", i)
            continue

        rel_path = raw.get("filepath") or raw.get("path") or raw.get("file")
        if not rel_path:
            logger.warning("Skipping entry %d (missing 'filepath')", i)
            continue

        # Resolve absolute path (allow prefixing with base_dir)
        abs_path = os.path.abspath(os.path.join(base_dir or "", rel_path))
        # Normalize to avoid duplicate keys due to different slashes/relatives
        norm_path = os.path.normpath(abs_path)

        # Basic validation
        if not norm_path.lower().endswith(".pdf"):
            logger.warning("Skipping non-PDF entry at %d: %s", i, norm_path)
            continue
        if not os.path.exists(norm_path):
            logger.warning("File not found for metadata entry %d: %s", i, norm_path)
            continue

        # Normalize/curate metadata fields (kept flexible & book-friendly)
        md = {
            "kind": "book",
            "title": raw.get("title"),
            "subtitle": raw.get("subtitle"),
            "publisher": raw.get("publisher"),
            "year": raw.get("year"),
            "pages": raw.get("pages"),
            "authors": raw.get("authors"),      # list[str] expected
            "doi": raw.get("doi"),
            "isbn": raw.get("isbn"),            # list[str] allowed
            "issn": raw.get("issn"),            # list[str] allowed
            "url": raw.get("url"),
            "download_url": raw.get("download_url"),
        }

        # Lightweight cleanup: drop Nones/empties
        md = {k: v for k, v in md.items() if v not in (None, "", [], {})}

        # Add a stable source_id if available (doi > first isbn > filename)
        source_id = raw.get("doi")
        if not source_id:
            isbns = raw.get("isbn") or []
            if isinstance(isbns, list) and isbns:
                source_id = isbns[0]
        if not source_id:
            source_id = os.path.basename(norm_path)
        md["source_id"] = source_id

        meta_by_path[norm_path] = md

    if not meta_by_path:
        raise ValueError("No valid PDF entries found in metadata.json (missing files / bad entries).")

    return meta_by_path


def ingest_books(
    metadata_json: str,
    collection_name: str,
    chunk_size: int,
    embedder,
    qdrant_url: str,
    base_dir: Optional[str] = None,
    use_safe_embedder: bool = False,
):
    logger.info("Loading book metadata from %s ...", metadata_json)
    meta_by_path = load_metadata_map(metadata_json, base_dir)

    # Reader: only load files present in metadata, and attach per-file metadata
    def file_metadata_fn(path: str) -> Dict[str, Any]:
        # SimpleDirectoryReader calls this with the absolute path
        norm = os.path.normpath(os.path.abspath(path))
        return dict(meta_by_path.get(norm, {}))

    input_files = list(meta_by_path.keys())
    logger.info("Preparing %d PDF(s) listed in metadata ...", len(input_files))
    documents = SimpleDirectoryReader(
        input_files=input_files,
        required_exts=[".pdf"],
        recursive=False,
        file_metadata=file_metadata_fn,
    ).load_data()

    # Minimal validation (same as your doc-ingest behavior)
    def get_text_safe(obj):
        t = None
        if hasattr(obj, "get_content"):
            try:
                t = obj.get_content(metadata_mode="none")
            except Exception:
                t = None
        if t is None:
            t = getattr(obj, "text", None)
        return t if isinstance(t, str) else None

    bad = []
    good_documents = []
    for i, d in enumerate(documents):
        t = get_text_safe(d)
        if t is None or not t.strip():
            bad.append((i, getattr(d, "id_", None), getattr(d, "metadata", {})))
        else:
            good_documents.append(d)

    if bad:
        logger.warning("%d invalid docs (empty/non-string text). Example: %s", len(bad), bad[0])
    logger.info("#good/tot=%d/%d", len(good_documents), len(documents))

    # Qdrant setup (same as your script)
    logger.info("Creating qdrant store ...")
    client = qdrant_client.QdrantClient(url=qdrant_url)
    qdrant_vector_store = QdrantVectorStore(
        client=client, collection_name=collection_name
    )
    storage_context = StorageContext.from_defaults(vector_store=qdrant_vector_store)

    # Embedding/indexing (mirrors your resilient path)
    Settings.llm = None
    Settings.chunk_size = chunk_size

    if use_safe_embedder:
        logger.info("Creating safe embedder ...")
        safe_embedder = SafeEmbedder(embedder)
        Settings.embed_model = safe_embedder

        logger.info("Creating sentence splitter ...")
        splitter = SentenceSplitter(chunk_size=chunk_size)

        logger.info("Getting nodes from documents ...")
        nodes = splitter.get_nodes_from_documents(good_documents)

        def clean_node_text(n):
            try:
                t = n.get_content(metadata_mode="none")
            except Exception:
                t = getattr(n, "text", None)
            if isinstance(t, (bytes, bytearray)):
                try:
                    t = bytes(t).decode("utf-8", errors="ignore")
                except Exception:
                    t = None
                if hasattr(n, "text"):
                    n.text = t
            return t

        logger.info("Selecting good nodes ...")
        clean_nodes = [n for n in nodes if isinstance((t := clean_node_text(n)), str) and t.strip()]
        logger.info("nodes=%d, clean=%d", len(nodes), len(clean_nodes))

        good_ids, good_embs, skipped = embed_nodes_resilient(clean_nodes, Settings.embed_model, batch_size=64)
        logger.warning("embedded=%d, skipped=%d", len(good_ids), len(skipped))

        vs = storage_context.vector_store
        added = False
        try:
            vs.add(ids=good_ids, embeddings=good_embs, nodes=None, metadata=None)
            added = True
        except TypeError:
            vs.add(embedding_results=list(zip(good_ids, good_embs)))
        if not added:
            logger.info("Vector store add() completed via fallback path.")

        index = build_index_from_vector_store(vs, storage_context)
        logger.info("Index ready. Skipped nodes: %d", len(skipped))
    else:
        Settings.embed_model = embedder
        logger.info("Storing documents (standard path) ...")
        index = VectorStoreIndex.from_documents(
            good_documents,
            storage_context=storage_context,
            Settings=Settings,
            show_progress=True,
        )

    logger.info("Books indexed successfully to Qdrant", collection=collection_name)
    return index


def main():
    parser = argparse.ArgumentParser()

    # metadata & base dir
    parser.add_argument(
        "-metadata_json", "--metadata_json",
        type=str, required=True,
        help="Path to metadata.json describing the books to ingest",
    )
    parser.add_argument(
        "-data_path", "--data_path",
        type=str, required=False, default=None,
        help="Optional base directory to resolve relative filepaths in metadata.json",
    )

    parser.add_argument(
        "-collection_name", "--collection_name",
        type=str, required=True, help="Qdrant collection name",
    )
    parser.add_argument(
        "-chunk_size", "--chunk_size",
        type=int, required=False, default=1024, help="Document chunk size",
    )
    parser.add_argument(
        "--use_safe_embedder", dest="use_safe_embedder",
        action="store_true", help="Use SafeEmbedder/resilient path (default=False)",
    )
    parser.set_defaults(use_safe_embedder=False)

    parser.add_argument(
        "-embedding_model", "--embedding_model",
        type=str, required=False, default="mixedbread-ai/mxbai-embed-large-v1",
        help="HF embedding model",
    )
    parser.add_argument(
        "-qdrant_url", "--qdrant_url",
        type=str, required=False, default="http://localhost:6333",
        help="Qdrant URL",
    )

    args = parser.parse_args()

    logger.info("Loading embedder model %s ...", args.embedding_model)
    embed_model = HuggingFaceEmbedding(
        model_name=args.embedding_model,
        trust_remote_code=True,
    )

    ingest_books(
        metadata_json=args.metadata_json,
        base_dir=args.data_path,
        collection_name=args.collection_name,
        chunk_size=args.chunk_size,
        embedder=embed_model,
        qdrant_url=args.qdrant_url,
        use_safe_embedder=args.use_safe_embedder,
    )

    logger.info("Ingest completed.")


if __name__ == "__main__":
    main()

