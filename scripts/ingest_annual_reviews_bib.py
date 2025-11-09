#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import argparse
import os
import json
import glob
import re
import structlog
from typing import List, Dict, Any, Optional, Tuple

# llama-index & qdrant
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    Settings,
)
from llama_index.core.node_parser import SentenceSplitter
import qdrant_client

# reuse your resilient pipeline pieces
from ingest_book import SafeEmbedder, embed_nodes_resilient, build_index_from_vector_store

logger = structlog.get_logger()

# -------------------- helpers --------------------

DOI_PREFIXES = ("https://doi.org/", "http://doi.org/", "doi:", "doi.org/")

def _prune_empties(x):
    if isinstance(x, dict):
        return {k: _prune_empties(v) for k, v in x.items() if v not in (None, "", [], {})}
    if isinstance(x, list):
        return [_prune_empties(v) for v in x if v not in (None, "", [], {})]
    return x

def normalize_doi(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    s = raw.strip()
    for p in DOI_PREFIXES:
        if s.lower().startswith(p):
            s = s[len(p):]
            break
    return s.strip("{}\" ").strip() or None

def doi_slug(doi: Optional[str]) -> Optional[str]:
    return doi.split("/", 1)[-1].lower() if doi else None

def discover_pdfs_from_single(pdf_path: str) -> Dict[str, str]:
    base = os.path.splitext(os.path.basename(pdf_path))[0].lower()
    return {base: os.path.abspath(pdf_path)}

def match_pdf_for_article(meta: Dict[str, Any], pdf_index: Dict[str, str]) -> Optional[str]:
    slug = doi_slug(meta.get("doi"))
    if slug:
        if slug in pdf_index:
            return pdf_index[slug]
        for b, p in pdf_index.items():
            if slug in b:
                return p
    title = (meta.get("title") or "").strip()
    if title:
        tslug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
        for b, p in pdf_index.items():
            if tslug in b:
                return p
    return None

# -------------------- BibTeX parsing (entry splitter) --------------------

ENTRY_START_RE = re.compile(r'@\w+\s*\{', re.IGNORECASE)

def split_bib_entries(text: str) -> List[str]:
    """Split a .bib file into raw entry blocks (brace-depth aware)."""
    entries, i, n = [], 0, len(text)
    while True:
        m = ENTRY_START_RE.search(text, i)
        if not m:
            break
        start = m.start()
        depth, j = 0, start
        while j < n:
            ch = text[j]
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    j += 1
                    break
            j += 1
        entries.append(text[start:j])
        i = j
    return entries

# -------------------- NEW: quote/brace-aware field parser --------------------

def _skip_ws(s: str, i: int) -> int:
    n = len(s)
    while i < n and s[i].isspace():
        i += 1
    return i

def _read_key(s: str, i: int) -> Tuple[Optional[str], int]:
    i = _skip_ws(s, i)
    n = len(s)
    start = i
    while i < n and (s[i].isalnum() or s[i] in "_-"):
        i += 1
    if i == start:
        return None, i
    return s[start:i].lower(), _skip_ws(s, i)

def _expect_char(s: str, i: int, ch: str) -> int:
    i = _skip_ws(s, i)
    if i < len(s) and s[i] == ch:
        return _skip_ws(s, i + 1)
    return i  # be permissive; upstream BibTeX often has minor formatting quirks

def _read_quoted(s: str, i: int) -> Tuple[str, int]:
    # assumes s[i] == '"'
    i += 1
    out = []
    n = len(s)
    esc = False
    brace_depth = 0  # braces may appear inside quotes in BibTeX
    while i < n:
        c = s[i]
        if esc:
            out.append(c)
            esc = False
        elif c == '\\':
            esc = True
        elif c == '{':
            brace_depth += 1
            out.append(c)
        elif c == '}':
            if brace_depth > 0:
                brace_depth -= 1
            out.append(c)
        elif c == '"':
            if brace_depth == 0:
                i += 1
                break
            else:
                out.append(c)
        else:
            out.append(c)
        i += 1
    return "".join(out).strip(), _skip_ws(s, i)

def _read_braced(s: str, i: int) -> Tuple[str, int]:
    # assumes s[i] == '{'
    i += 1
    out = []
    n = len(s)
    depth = 1
    esc = False
    while i < n and depth > 0:
        c = s[i]
        if esc:
            out.append(c)
            esc = False
        elif c == '\\':
            esc = True
            out.append(c)
        elif c == '{':
            depth += 1
            out.append(c)
        elif c == '}':
            depth -= 1
            if depth > 0:
                out.append(c)
        else:
            out.append(c)
        i += 1
    return "".join(out).strip(), _skip_ws(s, i)

def _read_bare(s: str, i: int) -> Tuple[str, int]:
    # until next comma or end-of-entry '}' (but not counting commas in nested braces/quotes)
    out = []
    n = len(s)
    while i < n and s[i] not in ',}':
        out.append(s[i])
        i += 1
    return "".join(out).strip(), _skip_ws(s, i)

def parse_bib_fields(block: str) -> Dict[str, Any]:
    """
    Quote/brace-aware parser for fields inside a single @entry{...}.
    Accumulates repeated keys into lists (e.g., keywords).
    """
    # find start of field list: after first '{' and the first comma that ends the key part
    lbrace = block.find('{')
    if lbrace < 0:
        return {}
    # skip entry key up to the first comma after '{'
    i = lbrace + 1
    depth = 1
    n = len(block)
    while i < n and block[i] != ',':
        # entry key may contain slashes/colons; just scan to comma
        i += 1
    i += 1  # move past that comma
    fields: Dict[str, Any] = {}
    while i < n:
        i = _skip_ws(block, i)
        if i >= n or block[i] == '}':
            break
        key, i2 = _read_key(block, i)
        if not key:
            # skip until next comma or closing brace
            while i < n and block[i] not in ',}':
                i += 1
            if i < n and block[i] == ',':
                i += 1
            continue
        i = _expect_char(block, i2, '=')
        i = _skip_ws(block, i)
        if i < n and block[i] == '"':
            val, i = _read_quoted(block, i)
        elif i < n and block[i] == '{':
            val, i = _read_braced(block, i)
        else:
            val, i = _read_bare(block, i)

        # accumulate repeats (e.g., repeated 'keywords' lines)
        if key in fields:
            if isinstance(fields[key], list):
                fields[key].append(val)
            else:
                fields[key] = [fields[key], val]
        else:
            fields[key] = val

        # optional trailing comma after a field
        i = _skip_ws(block, i)
        if i < n and block[i] == ',':
            i += 1
    return fields

# -------------------- higher-level mapping --------------------

def parse_bib_authors(author_field: Optional[str]) -> List[str]:
    """
    Convert BibTeX 'author' into list of display names "First M. Last".
    Handles:
      - "Last, First Middle and Last, First M."
      - "First Middle Last and First Last"
    """
    if not author_field:
        return []
    # split on ' and ' outside braces (rarely used in these files, but safe)
    parts = [a.strip() for a in re.split(r'\s+and\s+', author_field) if a.strip()]
    out = []
    for a in parts:
        # strip stray surrounding quotes/braces if present
        a = a.strip().strip('{}"')
        if ',' in a:
            last, rest = a.split(',', 1)
            last = last.strip()
            rest = re.sub(r'\s+', ' ', rest).strip()
            name = (rest + " " + last).strip()
        else:
            name = re.sub(r'\s+', ' ', a).strip()
        out.append(name)
    return out

def parse_bib_file(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    entries = split_bib_entries(text)
    metas: List[Dict[str, Any]] = []
    for block in entries:
        fdict = parse_bib_fields(block)

        title     = fdict.get("title")
        journal   = fdict.get("journal")
        year      = fdict.get("year")
        volume    = fdict.get("volume")
        number    = fdict.get("number")  # often "Volume 53, 2015"
        pages     = fdict.get("pages")
        doi       = normalize_doi(fdict.get("doi"))
        url       = fdict.get("url")
        publisher = fdict.get("publisher")
        issn      = fdict.get("issn")
        pub_type  = fdict.get("type")
        abstract  = fdict.get("abstract")

        # authors
        authors   = parse_bib_authors(fdict.get("author"))

        # keywords: can be repeated fields (list) or single string with delimiters
        kw_raw = fdict.get("keywords")
        if isinstance(kw_raw, list):
            # some lines contain multiple keywords separated by ';' or ','
            tmp = []
            for k in kw_raw:
                k = k.strip().strip('{}"')
                tmp.extend([t.strip() for t in re.split(r'[;,]', k) if t.strip()])
            keywords = tmp or None
        elif isinstance(kw_raw, str):
            s = kw_raw.strip().strip('{}"')
            keywords = [t.strip() for t in re.split(r'[;,]', s) if t.strip()] or None
        else:
            keywords = None

        # normalize issue: keep only if purely numeric
        issue = number.strip() if isinstance(number, str) and number.strip().isdigit() else None

        meta = {
            "kind": "annual-review",
            "publisher": publisher,
            "journal": journal,
            "issn": issn,
            "volume": volume,
            "issue": issue,
            "year": year,
            "month": None,  # not present in these .bib files
            "title": title,
            "first_page": pages.split("-")[0] if isinstance(pages, str) and "-" in pages else (pages or None),
            "last_page":  pages.split("-")[1] if isinstance(pages, str) and "-" in pages else None,
            "publication_type": pub_type,
            "authors": authors,
            "doi": doi,
            "keywords": keywords,
            "abstract": abstract.strip() if isinstance(abstract, str) else None,
            "url": url,
        }
        
        print("meta")
        print(meta)
        
        metas.append(_prune_empties(meta))
    return metas

def parse_bib_dir(bib_dir: str) -> List[Dict[str, Any]]:
    metas: List[Dict[str, Any]] = []
    for path in sorted(glob.glob(os.path.join(bib_dir, "*.bib"))):
        metas.extend(parse_bib_file(path))
    return metas

# -------------------- build md + ingest --------------------

def build_metadata_from_bib(
    bib_metadata_list: List[Dict[str, Any]],
    pdf_index: Dict[str, str],
) -> Dict[str, Any]:
    md: Dict[str, Any] = {}
    for meta in bib_metadata_list:
        pdf_path = match_pdf_for_article(meta, pdf_index)
        if not pdf_path:
            continue

        authors = meta.get("authors", [])
        first_author = authors[0] if authors else None

        pages_fmt = None
        if meta.get("first_page") or meta.get("last_page"):
            fp = meta.get("first_page", "?")
            lp = meta.get("last_page", "?")
            pages_fmt = f"{fp}-{lp}"

        url = meta.get("url") or (f"https://doi.org/{meta['doi']}" if meta.get("doi") else None)

        md = {
            "kind": meta.get("kind"),
            "filepath": pdf_path,

            # top-level (for payload_update)
            "title": meta.get("title"),
            "authors": authors,
            "first_author": first_author,

            # bibliographic bundle (b.*)
            "b": {
                "journal": meta.get("journal"),
                "volume": meta.get("volume"),
                "issue": meta.get("issue"),
                "pages": pages_fmt,
                "year": meta.get("year"),
                "month": meta.get("month"),
                "publisher": meta.get("publisher"),
                "issn": meta.get("issn"),
                "publication_type": meta.get("publication_type"),
            },

            # identifiers bundle (ids.*)
            "ids": {
                "doi": meta.get("doi"),
                "arxiv_id": None,
                "bibcode": None,
            },

            # extras
            "keywords": meta.get("keywords"),
            "abstract": meta.get("abstract"),
            "url": url,
            "download_url": None,
        }
        md = _prune_empties(md)
        break

    if not md:
        logger.warning("PDF was not matched to BibTeX metadata. Check DOI/filename.")
    return md

def ingest_annreviews(
    md: dict,
    collection_name: str,
    chunk_size: int,
    embedder,
    qdrant_url: str,
    use_safe_embedder: bool = False,
):
    """Same resilient ingestion path you use elsewhere."""

    def file_metadata_fn(_: str) -> Dict[str, Any]:
        return md

    input_files = [md.get("filepath")]
    logger.info("Preparing %d PDF(s) listed in metadata ...", len(input_files))

    documents = SimpleDirectoryReader(
        input_files=input_files,
        required_exts=[".pdf"],
        recursive=False,
        file_metadata=file_metadata_fn,
    ).load_data()

    # filter empty docs
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

    bad, good_documents = [], []
    for i, d in enumerate(documents):
        t = get_text_safe(d)
        if t is None or not t.strip():
            bad.append((i, getattr(d, "id_", None), getattr(d, "metadata", {})))
        else:
            good_documents.append(d)
    if bad:
        logger.warning("%d invalid docs (empty/non-string text). Example: %s", len(bad), bad[0])
    logger.info("#good/tot=%d/%d", len(good_documents), len(documents))

    # qdrant setup
    logger.info("Creating qdrant store ...")
    client = qdrant_client.QdrantClient(url=qdrant_url)
    vs = QdrantVectorStore(client=client, collection_name=collection_name)
    storage_context = StorageContext.from_defaults(vector_store=vs)

    # embedding/indexing
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

        good_ids, good_embs, skipped = embed_nodes_resilient(clean_nodes, Settings.embed_model, batch_size=64)
        logger.warning("embedded=%d, skipped=%d", len(good_ids), len(skipped))

        try:
            vs.add(ids=good_ids, embeddings=good_embs, nodes=None, metadata=None)
        except TypeError:
            vs.add(embedding_results=list(zip(good_ids, good_embs)))

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

# -------------------- CLI --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bib_dir", required=True, help="Directory containing Annual Reviews .bib files")
    ap.add_argument("--pdf_path", required=True, help="Single PDF file to ingest (matches by DOI slug)")
    ap.add_argument("--collection_name", required=True, help="Qdrant collection name")
    ap.add_argument("--qdrant_url", default="http://localhost:6333")
    ap.add_argument("--embedding_model", default="mixedbread-ai/mxbai-embed-large-v1")
    ap.add_argument("--chunk_size", type=int, default=1024)
    ap.add_argument("--use_safe_embedder", action="store_true", default=False)
    args = ap.parse_args()

    logger.info("Parsing BibTeX metadata from %s ...", args.bib_dir)
    bib_metadata_list = parse_bib_dir(args.bib_dir)

    pdf_index = discover_pdfs_from_single(args.pdf_path)
    logger.info("Matching metadata to PDF %s ...", list(pdf_index.values())[0])
    md = build_metadata_from_bib(bib_metadata_list, pdf_index)
    if not md:
        raise RuntimeError("No PDFs matched to BibTeX metadata. Check DOI/filename.")

    print(f"pdf file: {args.pdf_path}")
    print("pdf_index")
    print(pdf_index)
    print("md")
    print(md)
    sys.exit(0)

    logger.info("Loading embedder model %s ...", args.embedding_model)
    embed_model = HuggingFaceEmbedding(model_name=args.embedding_model, trust_remote_code=True)

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

