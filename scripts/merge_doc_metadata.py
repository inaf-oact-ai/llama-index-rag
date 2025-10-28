#!/usr/bin/env python
# -*- coding: utf-8 -*-

##########################
##   IMPORT MODULES
##########################
# - Import standard modules
import json, re, os, sys
from typing import Dict, Any, Optional
import structlog
import argparse
from tqdm import tqdm

# - Import Qdrant modules
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

logger = structlog.get_logger()


##########################
##   HELPER METHODS
##########################
def norm_arxiv(s: str) -> Optional[str]:
    if not s: return None
    s = s.strip()
    s = s.replace("arXiv:", "").lower()
    # keep only 4/5-digit new style id (e.g. 2411.12564) or old 7-digit (e.g. 9910196)
    m = re.search(r"(\d{4}\.\d{4,5})(?:v\d+)?", s)
    if m: return m.group(1)
    m = re.search(r"(\d{7})(?:v\d+)?", s)  # old-style year+seq
    if m: return m.group(1)
    return None

def first_author(authors_field: Any) -> Optional[str]:
    # ADS JSON examples show "authors": "Last, First et al."
    if isinstance(authors_field, str):
        s = authors_field.strip()
        # split at " et al." first
        s = re.sub(r"\s+et\s+al\.\s*$", "", s, flags=re.I)
        # if many separated by ; or " and " just take first segment
        s = re.split(r"\s*;\s*|\s+and\s+|&", s)[0]
        return s.strip() or None
    if isinstance(authors_field, list) and authors_field:
        return str(authors_field[0]).strip()
    return None

def title_from(item: Dict[str, Any]) -> Optional[str]:
    t = item.get("title")
    if isinstance(t, list) and t: return t[0]
    if isinstance(t, str): return t
    return None

def parse_bibcode(bib: str) -> Dict[str, Optional[str]]:
    # Example: 2025ApJ...978....5X
    out = {"journal": None, "volume": None, "pages": None, "year": None}
    if not isinstance(bib, str): return out
    m = re.match(r"(?P<year>\d{4})(?P<journal>[A-Za-z\.]{4,9})\.*(?P<vol>\d{1,4})\.*(?P<page>[A-Za-z\d\.]+)", bib)
    if m:
        out["year"] = m.group("year")
        out["journal"] = m.group("journal").replace('.', '')
        out["volume"] = m.group("vol")
        out["pages"]  = m.group("page").strip('.')
    return out

def identifiers_map(item: Dict[str, Any]) -> Dict[str, str]:
    ids = {"arxiv_id": None, "bibcode": None, "doi": None}
    # explicit fields
    if "arxiv_id" in item and isinstance(item["arxiv_id"], str):
        ids["arxiv_id"] = item["arxiv_id"]
    # identifiers often carries bibcode etc.
    for x in item.get("identifiers", []) or []:
        if isinstance(x, str):
            if re.match(r"^\d{4}[A-Za-z\.]{3,}", x): ids["bibcode"] = x
            elif x.lower().startswith("arxiv:"): ids["arxiv_id"] = x
        # also pick up DOIs from the "doi" list
    dois = item.get("doi") or []
    if isinstance(dois, list) and dois:
        ids["doi"] = dois[0]
    elif isinstance(dois, str):
        ids["doi"] = dois
    return ids

######################################
##    ARGS
######################################
# - Parse arguments
def load_args():
    """ Load arguments """
    parser = argparse.ArgumentParser()
    parser.add_argument("-metadata", "--metadata", type=str, required=True, help="Input ADS metadata file (.json)")
    parser.add_argument("-collection_name", "--collection_name", type=str, required=False, default="radiopapers", help="Collection name")
    parser.add_argument("-qdrant_url", "--qdrant_url", type=str, required=False, default="http://localhost:6333", help="QDRant URL")

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
    
    inputfile_metadata= args.metadata
    collection = args.collection_name
    qdrant_url = args.qdrant_url

    # - Load metadata
    logger.info(f"Reading metadata file {inputfile_metadata} ...")
    with open(inputfile_metadata, "r", encoding="utf-8") as f:
        ads = json.load(f)

    # - Build index by normalized arXiv key and also by bibcode
    ads_by_arxiv: Dict[str, Dict[str, Any]] = {}
    ads_by_bib: Dict[str, Dict[str, Any]] = {}

    for it in ads:
        ids = identifiers_map(it)
        narx = norm_arxiv(ids.get("arxiv_id") or "")
        if narx:
            ads_by_arxiv[narx] = it
        bib = ids.get("bibcode")
        if bib:
            ads_by_bib[bib] = it

    # - Connect to Qdrant
    client = QdrantClient(url=qdrant_url)
    
    # - Total points for progress bar
    total_points = client.count(collection, exact=True).count
    pbar = tqdm(total=total_points, desc="Updating payloads", unit="pt")

    # - Scroll all points
    limit = 1000
    offset = None
    updated, scanned = 0, 0

    while True:
        resp = client.scroll(
            collection_name=collection,
            limit=limit,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        points, offset = resp[0], resp[1]
        if not points:
            break

        updates = [] 
        for p in points:
            scanned += 1
            pbar.update(1)
            
            md = p.payload or {}
            file_name = (md.get("file_name") or "") + " " + (md.get("file_path") or "")
            # try to extract arXiv numeric id from file_name/path
            key = norm_arxiv(file_name)
            item = None

            if key and key in ads_by_arxiv:
                item = ads_by_arxiv[key]
            else:
                # fallback: try matching by any bibcode substring in file path (rare)
                for bib in ads_by_bib.keys():
                    if bib in file_name:
                        item = ads_by_bib[bib]
                        break

            if not item:
                continue

            ids = identifiers_map(item)
            bib = ids.get("bibcode") or ""
            b = parse_bibcode(bib)

            payload_update = {
                "title": title_from(item),
                "authors": item.get("authors"),
                "first_author": first_author(item.get("authors")),
                "journal": b.get("journal"),
                "volume": b.get("volume"),
                "pages": b.get("pages"),
                "year": b.get("year"),
                "doi": ids.get("doi"),
                "arxiv_id": ids.get("arxiv_id"),
                "bibcode": bib or None,
            }

            # prune Nones so we only add present fields
            payload_update = {k: v for k, v in payload_update.items() if v}

            if payload_update:
                #existing_meta = (p.payload or {}).get("metadata", {})
                #merged_meta = {**existing_meta, **payload_update}
            
                #client.set_payload(
                #    collection_name=collection,
                #    payload={"metadata": merged_meta},
                #    points=[p.id],
                #)

                #client.set_payload(
                #    collection_name=collection,
                #    payload=payload_update,
                #    points=[p.id],
                #)
                         
                blob = (p.payload or {}).get("_node_content")
                if not blob or not isinstance(blob, str):
                    continue  # nothing we can safely update

                try:
                    node = json.loads(blob)
                except Exception:
                    continue  # skip malformed nodes


                # sanity check: must keep a valid text
                if not isinstance(node.get("text"), str) or node["text"] == "":
                    # don't touch broken nodes; let the checker tell us which IDs to handle
                    continue
     
                # merge metadata
                node_meta = node.get("metadata") or {}
                node_meta.update(payload_update)
                node["metadata"] = node_meta

                updated_blob = json.dumps(node, ensure_ascii=False)

                client.set_payload(
                    collection_name=collection,
                    payload={"_node_content": updated_blob},  # <-- only this field
                    points=[p.id],
                )
    
                updated += 1

        if offset is None:
            break

    print(f"Scanned {scanned} points; updated {updated} payloads in '{collection}'.")
    pbar.close()

########################
##    RUN MAIN
########################
if __name__ == "__main__":
    main()
