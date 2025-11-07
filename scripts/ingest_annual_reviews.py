#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import json
import structlog
from typing import List, Iterable, Optional, Any, Dict, Tuple
import traceback
import glob
import re
import structlog
import xml.etree.ElementTree as ET

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

# ---- reuse your existing pipeline (unchanged)
# We import the helpers & main ingest function plumbing from your script.
# Make sure this file is in the same folder as ingest_book.py, or adjust PYTHONPATH.
from ingest_book import (
    SafeEmbedder,
    embed_nodes_resilient,
    build_index_from_vector_store,
)

# ---- qdrant
import qdrant_client

logger = structlog.get_logger()

# ---------- XML parsing helpers ----------

def _text(x: Optional[ET.Element]) -> Optional[str]:
    return x.text.strip() if (x is not None and x.text) else None

def _find_one(parent: ET.Element, path: str) -> Optional[ET.Element]:
    return parent.find(path)

def _findall(parent: ET.Element, path: str) -> List[ET.Element]:
    found = parent.findall(path)
    return found if found is not None else []

def _slug_from_doi(doi: str) -> str:
    # DOI like '10.1146/annurev-astro-013125-122023' -> 'annurev-astro-013125-122023'
    return doi.split("/", 1)[-1].lower()

def _title_to_slug(title: str) -> str:
    s = title.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s

def _prune_empties(x):
    if isinstance(x, dict):
        return {k: _prune_empties(v) for k, v in x.items() if v not in (None, "", [], {})}
    if isinstance(x, list):
        return [ _prune_empties(v) for v in x if v not in (None, "", [], {}) ]
    return x

def parse_article(article_el: ET.Element) -> Dict[str, Any]:
    # Journal / issue
    journal = _find_one(article_el, "Journal")
    year = _text(_find_one(journal, "PubDate/Year")) if journal is not None else None
    month = _text(_find_one(journal, "PubDate/Month")) if journal is not None else None
    volume = _text(_find_one(journal, "Volume")) if journal is not None else None
    issue = _text(_find_one(journal, "Issue")) if journal is not None else None
    issn  = _text(_find_one(journal, "Issn")) if journal is not None else None
    publisher = _text(_find_one(journal, "PublisherName")) if journal is not None else None
    journal_title = _text(_find_one(journal, "JournalTitle")) if journal is not None else None

    # Article core
    title = _text(_find_one(article_el, "ArticleTitle"))
    first_page = _text(_find_one(article_el, "FirstPage"))
    last_page = _text(_find_one(article_el, "LastPage"))
    pub_type = _text(_find_one(article_el, "PublicationType"))

    # DOI: try ELocationID first, then ArticleIdList
    doi = None
    eloc = _find_one(article_el, "ELocationID[@EIdType='doi']")
    if eloc is not None and _text(eloc):
        doi = _text(eloc)
    if not doi:
        for aid in _findall(article_el, "ArticleIdList/ArticleId"):
            if aid.get("IdType", "").lower() == "doi" and _text(aid):
                doi = _text(aid)
                break

    # Authors
    authors = []
    for auth in _findall(article_el, "AuthorList/Author"):
        fn = _text(_find_one(auth, "FirstName")) or ""
        ln = _text(_find_one(auth, "LastName")) or ""
        fullname = f"{fn} {ln}".strip()
        if fullname:
            authors.append(fullname)

    # Keywords
    keywords = []
    for obj in _findall(article_el, "ObjectList/Object[@Type='keyword']"):
        p = _find_one(obj, "Param[@Name='value']")
        if p is not None and _text(p):
            keywords.append(_text(p))

    meta = {
        "kind": "annual-review",
        "publisher": publisher,
        "journal": journal_title,
        "issn": issn,
        "volume": volume,
        "issue": issue,
        "year": year,
        "month": month,
        "title": title,
        "first_page": first_page,
        "last_page": last_page,
        "publication_type": pub_type,
        "authors": authors,       # list[str]
        "doi": doi,
        "keywords": keywords,
    }
    # drop empty
    meta = {k: v for k, v in meta.items() if v not in (None, "", [], {})}
    return meta

def parse_issue_xml(xml_path: str) -> List[Dict[str, Any]]:
    """
    Returns a list of article metadata dicts extracted from one XML file.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        logger.warning("Failed to parse XML: %s (%s)", xml_path, e)
        return []

    # Support both <ArticleSet><Article>… and just <Article>…
    articles = root.findall(".//Article") if root.tag != "Article" else [root]
    out = []
    for a in articles:
        meta = parse_article(a)
        if meta.get("title") or meta.get("doi"):
            out.append(meta)
    return out

# ---------- PDF matching ----------

def discover_pdfs(pdf_dir: str) -> Dict[str, str]:
    """
    Returns mapping: {basename_without_ext.lower(): absolute_path}
    Also returns an auxiliary dict keyed by full filename substring for fuzzy contains.
    """
    paths = {}
    for p in glob.glob(os.path.join(pdf_dir, "**", "*.pdf"), recursive=True):
        base = os.path.splitext(os.path.basename(p))[0].lower()
        paths[base] = os.path.abspath(p)
    return paths

def match_pdf_for_article(meta: Dict[str, Any], pdf_index: Dict[str, str]) -> Optional[str]:
    """
    Try to match via DOI slug first, then via 'annurev-astro-...' slug in filename,
    finally fall back to title-based slug (rarely needed).
    """
    # 1) DOI slug
    doi = meta.get("doi")
    if doi:
        slug = _slug_from_doi(doi)  # e.g., 'annurev-astro-013125-122023'
        if slug in pdf_index:
            return pdf_index[slug]
        # Sometimes filenames prefix/suffix; try containment scan
        for base, path in pdf_index.items():
            if slug in base:
                return path

    # 2) Heuristic Annual Reviews slug in title (very rare fallback)
    title = meta.get("title")
    if title:
        tslug = _title_to_slug(title)
        for base, path in pdf_index.items():
            if tslug in base:
                return path

    return None

# ---------- Orchestration ----------
def build_metadata_json_from_xmls(
    xml_metadata_list: list,
    pdf_index: Optional[Dict[str, str]] = None
) -> Tuple[int, int]:
    """ Scans all XML metadata, matches to PDFs in pdf_index and return metadata in dict format for ingestion"""
    
    # - Scan and match XML metadata to file
    md= {}
    found= False

    for meta in xml_metadata_list:
        # - Match PDF and XML metadata    
        pdf_path = match_pdf_for_article(meta, pdf_index)
        if not pdf_path:
            #logger.debug("No PDF match for DOI=%s title=%s", meta.get("doi"), meta.get("title"))
            continue
            
        found=True

        # - Normalize to ingest schema
        authors = meta.get("authors", [])
        first_author = authors[0] if authors else None
        url= "https://doi.org/" + meta.get("doi")
            
        title = meta.get("title")
        md = {
            "kind": meta.get("kind"),
            "filepath": pdf_path,
            "title": title,
            "publisher": meta.get("publisher"),
            "journal": meta.get("journal"),
            "volume": meta.get("volume"),
            "issue": meta.get("issue"),
            "year": meta.get("year"),
            "month": meta.get("month"),
            "pages": f"{meta.get('first_page','?')}-{meta.get('last_page','?')}",
            "authors": authors,
            "first_author": first_author,
            "doi": meta.get("doi"),
            "issn": [meta.get("issn")] if meta.get("issn") else None,
            "keywords": meta.get("keywords"),
            "publication_type": meta.get("publication_type"),
            "url": url,
            "download_url": None,
        }
            
        # drop Nones
        md = {k: v for k, v in md.items() if v not in (None, "", [], {})}
        break
        
            
    if not md:
        logger.warning("PDF was not matched to XML metadata. Check naming and directories.")

    return md


def ingest_annreviews(
    md: dict,
    collection_name: str,
    chunk_size: int,
    embedder,
    qdrant_url: str,
    use_safe_embedder: bool = False,
):
    """ Ingest annual reviews from parsed metadata """

    # - Set metadata function (needed by SimpleDirectoryReader)
    def file_metadata_fn(path: str) -> Dict[str, Any]:
          # - Just return the input metadata dict
          return md
    
    # - Set input files from path
    input_files= [md.get("filepath")]  
    logger.info("Preparing %d PDF(s) listed in metadata ...", len(input_files))
    
    # - Set simple directory reader
    documents = SimpleDirectoryReader(
        input_files=input_files,
        required_exts=[".pdf"],
        recursive=False,
        file_metadata=file_metadata_fn,
    ).load_data()
      
    # - Minimal validation (same as your doc-ingest behavior)
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

    # - Qdrant setup (same as your script)
    logger.info("Creating qdrant store ...")
    client = qdrant_client.QdrantClient(url=qdrant_url)
    qdrant_vector_store = QdrantVectorStore(
        client=client, collection_name=collection_name
    )
    storage_context = StorageContext.from_defaults(vector_store=qdrant_vector_store)

    # - Embedding/indexing (mirrors your resilient path)
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

    logger.info("Reviews indexed successfully to Qdrant", collection=collection_name)
    return index  
      

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml_dir", required=True, help="Directory containing Annual Reviews XML files")
    ap.add_argument("--pdf_path", help="Single PDF file to ingest")
    ap.add_argument("--collection_name", required=True, help="Qdrant collection name")
    ap.add_argument("--qdrant_url", default="http://localhost:6333")
    ap.add_argument("--embedding_model", default="mixedbread-ai/mxbai-embed-large-v1")
    ap.add_argument("--chunk_size", type=int, default=1024)
    ap.add_argument("--use_safe_embedder", action="store_true", default=False)
    
    args = ap.parse_args()

    # - Find and parse all metadata in xml_dir
    logger.info(f"Finding all XML metadata in dir {args.xml_dir} ...")
    xml_files = sorted(glob.glob(os.path.join(args.xml_dir, "*.xml")))
    
    logger.info(f"Parsing all XML metadata found (n={len(xml_files)}) ...")
    xml_metadata_list= []
    for xf in xml_files:
        meta_xml= parse_issue_xml(xf) # this is a list
        xml_metadata_list.extend(meta_xml)
   
    # - Build metadata dict
    pdf_dir = os.path.dirname(args.pdf_path)
    pdf_index = {os.path.splitext(os.path.basename(args.pdf_path))[0].lower(): os.path.abspath(args.pdf_path)}
    logger.info(f"Building dict metadata from XMLs meta for PDF {pdf_index} ...")
    md= build_metadata_json_from_xmls(xml_metadata_list, pdf_index)
    
    if not md:
        raise RuntimeError("No PDFs matched to XML metadata. Check naming and directories.")

    # - Create embedder
    logger.info("Loading embedder model %s ...", args.embedding_model)
    embed_model = HuggingFaceEmbedding(model_name=args.embedding_model, trust_remote_code=True)
    
    # - Ingest 
    logger.info("Start ingestion ...")
    ingest_annreviews(
        md=md,
        collection_name=args.collection_name,
        chunk_size=args.chunk_size,
        embedder=embed_model,
        qdrant_url=args.qdrant_url,
        use_safe_embedder=args.use_safe_embedder,
    )
    
    logger.info("Annual Reviews ingest completed.", collection=args.collection_name)

if __name__ == "__main__":
    main()

