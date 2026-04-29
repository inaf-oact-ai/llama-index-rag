#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import json
import glob
import re
import hashlib
import html
from typing import List, Iterable, Optional, Any, Dict, Tuple

import structlog

# ---- llama-index
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import (
	SimpleDirectoryReader,
	StorageContext,
	VectorStoreIndex,
	Settings,
)
from llama_index.core.node_parser import SentenceSplitter

# ---- reuse your resilient pipeline pieces from ingest_book.py
# Keep this file in the same directory as ingest_book.py, or adjust PYTHONPATH.
from ingest_book import SafeEmbedder, embed_nodes_resilient, build_index_from_vector_store

# ---- qdrant
import qdrant_client

logger = structlog.get_logger()

DOI_PREFIXES = (
	"https://doi.org/",
	"http://doi.org/",
	"doi:",
	"doi.org/",
)


def _prune_empties(x):
	if isinstance(x, dict):
		return {k: _prune_empties(v) for k, v in x.items() if v not in (None, "", [], {})}
	if isinstance(x, list):
		return [_prune_empties(v) for v in x if v not in (None, "", [], {})]
	return x


def first_or_none(value):
	if isinstance(value, list):
		return value[0] if value else None
	return value


def normalize_doi(raw: Optional[str]) -> Optional[str]:
	if not raw:
		return None
	s = str(raw).strip()
	for p in DOI_PREFIXES:
		if s.lower().startswith(p):
			s = s[len(p):]
			break
	return s.strip().strip("{}\" ") or None


def doi_slug(doi: Optional[str]) -> Optional[str]:
	doi = normalize_doi(doi)
	return doi.split("/", 1)[-1].lower() if doi and "/" in doi else doi


def normalize_for_match(s: Optional[str]) -> Optional[str]:
	if not s:
		return None
	s = s.lower()
	s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
	return s or None


def date_parts_to_iso(value: Any) -> Optional[str]:
	"""Convert Crossref {'date-parts': [[YYYY, MM, DD]]} to ISO date when possible."""
	if not isinstance(value, dict):
		return None
	parts = value.get("date-parts")
	if not parts or not isinstance(parts, list) or not parts[0]:
		return None
	p = parts[0]
	try:
		y = int(p[0])
		m = int(p[1]) if len(p) > 1 else 1
		d = int(p[2]) if len(p) > 2 else 1
		return f"{y:04d}-{m:02d}-{d:02d}"
	except Exception:
		return None


def year_from_date(value: Optional[str]) -> Optional[int]:
	if not value:
		return None
	try:
		return int(str(value)[:4])
	except Exception:
		return None


def clean_crossref_abstract(raw: Optional[str]) -> Optional[str]:
	"""Remove simple JATS/XML tags without adding heavy dependencies."""
	if not raw:
		return None
	s = html.unescape(str(raw))
	s = re.sub(r"<[^>]+>", " ", s)
	s = re.sub(r"\s+", " ", s).strip()
	return s or None


def get_assertion_value(item: Dict[str, Any], name: str) -> Optional[str]:
	for a in item.get("assertion", []) or []:
		if not isinstance(a, dict):
			continue
		if str(a.get("name", "")).lower() == name.lower():
			return a.get("value")
	return None


def extract_link(item: Dict[str, Any], content_type: str) -> Optional[str]:
	for link in item.get("link", []) or []:
		if not isinstance(link, dict):
			continue
		if link.get("content-type") == content_type and link.get("URL"):
			return link["URL"]
	return None


def extract_license_url(item: Dict[str, Any]) -> Optional[str]:
	licenses = item.get("license", []) or []
	for lic in licenses:
		if isinstance(lic, dict) and lic.get("URL"):
			return lic["URL"]
	return None


def extract_authors(item: Dict[str, Any]) -> Tuple[List[str], List[str]]:
	authors = []
	orcids = []
	for a in item.get("author", []) or []:
		if not isinstance(a, dict):
			continue
		given = a.get("given") or ""
		family = a.get("family") or ""
		name = f"{given} {family}".strip()
		if name:
			authors.append(name)
		if a.get("ORCID"):
			orcids.append(a["ORCID"])
	return authors, orcids


def sha256_file(path: str, block_size: int = 1024 * 1024) -> Optional[str]:
	try:
		h = hashlib.sha256()
		with open(path, "rb") as f:
			while True:
				b = f.read(block_size)
				if not b:
					break
				h.update(b)
		return h.hexdigest()
	except Exception:
		return None


def unwrap_crossref_json(raw: Any) -> List[Dict[str, Any]]:
	"""
	Accepts:
	  - one Crossref work object
	  - Crossref API response {'status': 'ok', 'message': {...}}
	  - list of work objects
	  - {'items': [...]} or {'message': {'items': [...]}}
	"""
	if isinstance(raw, list):
		return [x for x in raw if isinstance(x, dict)]
	if not isinstance(raw, dict):
		return []
	if isinstance(raw.get("message"), dict):
		msg = raw["message"]
		if isinstance(msg.get("items"), list):
			return [x for x in msg["items"] if isinstance(x, dict)]
		return [msg]
	if isinstance(raw.get("items"), list):
		return [x for x in raw["items"] if isinstance(x, dict)]
	return [raw]


def load_crossref_metadata_paths(metadata_path: Optional[str], metadata_dir: Optional[str]) -> List[Dict[str, Any]]:
	items: List[Dict[str, Any]] = []
	paths: List[str] = []

	if metadata_path:
		paths.append(metadata_path)
	if metadata_dir:
		paths.extend(sorted(glob.glob(os.path.join(metadata_dir, "**", "*.json"), recursive=True)))

	seen = set()
	for path in paths:
		path = os.path.abspath(path)
		if path in seen:
			continue
		seen.add(path)
		try:
			with open(path, "r", encoding="utf-8") as f:
				raw = json.load(f)
		except Exception as e:
			logger.warning("Skipping metadata JSON %s: %s", path, e)
			continue
		for item in unwrap_crossref_json(raw):
			item["_metadata_json_path"] = path
			items.append(item)

	return items


def discover_pdfs(pdf_path: Optional[str], pdf_dir: Optional[str]) -> Dict[str, str]:
	pdfs: Dict[str, str] = {}
	paths: List[str] = []
	if pdf_path:
		paths.append(pdf_path)
	if pdf_dir:
		paths.extend(sorted(glob.glob(os.path.join(pdf_dir, "**", "*.pdf"), recursive=True)))

	for p in paths:
		ap = os.path.abspath(p)
		if not os.path.isfile(ap):
			logger.warning("PDF not found: %s", ap)
			continue
		base = os.path.splitext(os.path.basename(ap))[0].lower()
		pdfs[base] = ap
	return pdfs


def match_pdf_for_crossref_item(item: Dict[str, Any], pdf_index: Dict[str, str]) -> Optional[str]:
	# explicit filepath in custom aggregate JSON wins
	for key in ("filepath", "pdf_path", "path", "file"):
		if item.get(key):
			p = os.path.abspath(item[key])
			if os.path.isfile(p):
				return p

	doi = normalize_doi(item.get("DOI") or item.get("doi"))
	slug = doi_slug(doi)
	if slug:
		if slug in pdf_index:
			return pdf_index[slug]
		for base, path in pdf_index.items():
			if slug in base or base in slug:
				return path
		# Springer filenames often contain DOI suffix with punctuation converted.
		norm_slug = normalize_for_match(slug)
		for base, path in pdf_index.items():
			if norm_slug and (norm_slug in normalize_for_match(base) or normalize_for_match(base) in norm_slug):
				return path

	title = first_or_none(item.get("title")) or item.get("title")
	tslug = normalize_for_match(title)
	if tslug:
		for base, path in pdf_index.items():
			bslug = normalize_for_match(base)
			if bslug and (tslug in bslug or bslug in tslug):
				return path

	return None


def crossref_to_metadata(item: Dict[str, Any], pdf_path: str, include_references: bool = False) -> Dict[str, Any]:
	doi = normalize_doi(item.get("DOI") or item.get("doi"))
	title = first_or_none(item.get("title"))
	journal = first_or_none(item.get("container-title"))
	journal_short = first_or_none(item.get("short-container-title"))
	published_date = (
		date_parts_to_iso(item.get("published"))
		or date_parts_to_iso(item.get("published-online"))
		or date_parts_to_iso(item.get("issued"))
	)
	year = year_from_date(published_date)
	authors, orcids = extract_authors(item)
	license_url = extract_license_url(item)
	pdf_url = extract_link(item, "application/pdf")
	html_url = extract_link(item, "text/html")
	url = item.get("URL") or (f"https://doi.org/{doi}" if doi else html_url)
	funders = [f.get("name") for f in item.get("funder", []) or [] if isinstance(f, dict) and f.get("name")]
	keywords = item.get("subject") if isinstance(item.get("subject"), list) else None

	md = {
		"kind": "living-review-solar-physics",
		"collection": "living_reviews_solar_physics",
		"domain": "solar_physics",
		"source_family": "springer_living_reviews",
		"source_name": journal or "Living Reviews in Solar Physics",
		"source_type": "journal_review_article",
		"crossref_type": item.get("type"),

		"filepath": os.path.abspath(pdf_path),
		"pdf_filename": os.path.basename(pdf_path),
		"document_hash": sha256_file(pdf_path),

		"title": title,
		"authors": authors,
		"first_author": authors[0] if authors else None,
		"author_orcids": orcids,

		"year": year,
		"published_date": published_date,
		"published_online_date": date_parts_to_iso(item.get("published-online")),
		"received_date": get_assertion_value(item, "received"),
		"accepted_date": get_assertion_value(item, "accepted"),

		"journal": journal,
		"journal_short": journal_short,
		"publisher": item.get("publisher"),
		"volume": item.get("volume"),
		"issue": item.get("issue"),
		"article_number": item.get("article-number") or item.get("article_number"),
		"issn": item.get("ISSN") or item.get("issn"),
		"language": item.get("language"),

		"doi": doi,
		"url": url,
		"pdf_url": pdf_url,
		"html_url": html_url,
		"download_url": pdf_url,

		#"abstract": clean_crossref_abstract(item.get("abstract")),
		"keywords": keywords,
		"license": license_url,
		"is_open_access": bool(license_url and "creativecommons.org" in license_url.lower()),

		"reference_count": item.get("reference-count") or item.get("references-count"),
		"is_referenced_by_count": item.get("is-referenced-by-count"),
		"funders": funders,
		"metadata_json_path": item.get("_metadata_json_path"),
	}

	if include_references:
		md["references"] = item.get("reference")

	source_id = doi or os.path.basename(pdf_path)
	md["source_id"] = f"doi:{doi}" if doi else source_id

	return _prune_empties(md)


def build_metadata_map(
	metadata_path: Optional[str],
	metadata_dir: Optional[str],
	pdf_path: Optional[str],
	pdf_dir: Optional[str],
	include_references: bool = False,
) -> Dict[str, Dict[str, Any]]:
	items = load_crossref_metadata_paths(metadata_path, metadata_dir)
	pdf_index = discover_pdfs(pdf_path, pdf_dir)

	logger.info("Loaded Crossref metadata records: %d", len(items))
	logger.info("Discovered PDFs: %d", len(pdf_index))

	meta_by_path: Dict[str, Dict[str, Any]] = {}
	unmatched = []

	for item in items:
		pdf = match_pdf_for_crossref_item(item, pdf_index)
		if not pdf:
			unmatched.append({
				"doi": item.get("DOI") or item.get("doi"),
				"title": first_or_none(item.get("title")),
			})
			continue
		md = crossref_to_metadata(item, pdf, include_references=include_references)
		meta_by_path[os.path.normpath(os.path.abspath(pdf))] = md

	if unmatched:
		logger.warning("Unmatched metadata records: %d. First unmatched: %s", len(unmatched), unmatched[0])
	if not meta_by_path:
		raise ValueError("No PDF matched to Crossref metadata. Check DOI-based PDF filenames, --pdf_path/--pdf_dir, or explicit filepath fields.")

	return meta_by_path


def ingest_living_reviews(
	meta_by_path: Dict[str, Dict[str, Any]],
	collection_name: str,
	chunk_size: int,
	embedder,
	qdrant_url: str,
	use_safe_embedder: bool = False,
):
	def file_metadata_fn(path: str) -> Dict[str, Any]:
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

	# Avoid injecting large metadata fields into the text passed to the splitter/embedder.
	# The fields remain stored in document/node metadata payloads, but are not prepended
	# to the text content used for chunking/embedding.
	exclude_from_text = [
		"abstract",
		"author_orcids",
		"funders",
		"keywords",
		"license",
		"pdf_url",
		"html_url",
		"download_url",
		"url",
		"metadata_json_path",
		"document_hash",
		"references",
	]

	for d in good_documents:
		d.excluded_embed_metadata_keys = list(
			set(getattr(d, "excluded_embed_metadata_keys", []) + exclude_from_text)
		)
		d.excluded_llm_metadata_keys = list(
			set(getattr(d, "excluded_llm_metadata_keys", []) + exclude_from_text)
		)


	logger.info("Creating qdrant store ...")
	client = qdrant_client.QdrantClient(url=qdrant_url)
	vs = QdrantVectorStore(client=client, collection_name=collection_name)
	storage_context = StorageContext.from_defaults(vector_store=vs)

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

	logger.info("Living Reviews in Solar Physics indexed successfully to Qdrant", collection=collection_name)
	return index


def main():
	parser = argparse.ArgumentParser()

	parser.add_argument("--metadata_json", default=None, help="Single Crossref JSON file, or an aggregate JSON list/API response")
	parser.add_argument("--metadata_dir", default=None, help="Directory containing one Crossref JSON per paper")
	parser.add_argument("--pdf_path", default=None, help="Single PDF file to ingest")
	parser.add_argument("--pdf_dir", default=None, help="Directory containing PDFs")
	parser.add_argument("--collection_name", required=True, help="Qdrant collection name")
	parser.add_argument("--qdrant_url", default="http://localhost:6333")
	parser.add_argument("--embedding_model", default="mixedbread-ai/mxbai-embed-large-v1")
	parser.add_argument("--chunk_size", type=int, default=1024)
	parser.add_argument("--use_safe_embedder", action="store_true", default=False)
	parser.add_argument("--include_references", action="store_true", default=False, help="Store full Crossref reference list in each payload. Usually not recommended because it can make payloads large.")
	parser.add_argument("--dryrun", action="store_true", default=False, help="Build and print metadata map without uploading to Qdrant")
	args = parser.parse_args()

	if not args.metadata_json and not args.metadata_dir:
		raise ValueError("Provide --metadata_json or --metadata_dir")
	if not args.pdf_path and not args.pdf_dir:
		raise ValueError("Provide --pdf_path or --pdf_dir")

	meta_by_path = build_metadata_map(
		metadata_path=args.metadata_json,
		metadata_dir=args.metadata_dir,
		pdf_path=args.pdf_path,
		pdf_dir=args.pdf_dir,
		include_references=args.include_references,
	)

	print("Matched metadata records:")
	for path, md in meta_by_path.items():
		print(json.dumps({
			"filepath": path,
			"title": md.get("title"),
			"doi": md.get("doi"),
			"year": md.get("year"),
			"journal": md.get("journal"),
		}, indent=2, ensure_ascii=False))

	if args.dryrun:
		logger.info("Dry run requested; not uploading to Qdrant.")
		return

	logger.info("Loading embedder model %s ...", args.embedding_model)
	embed_model = HuggingFaceEmbedding(model_name=args.embedding_model, trust_remote_code=True)

	logger.info("Start ingestion ...")
	ingest_living_reviews(
		meta_by_path=meta_by_path,
		collection_name=args.collection_name,
		chunk_size=args.chunk_size,
		embedder=embed_model,
		qdrant_url=args.qdrant_url,
		use_safe_embedder=args.use_safe_embedder,
	)
	logger.info("Living Reviews ingest completed.", collection=args.collection_name)


if __name__ == "__main__":
	main()
