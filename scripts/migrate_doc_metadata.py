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

######################################
##    ARGS
######################################
# - Parse arguments
def load_args():
    """ Load arguments """
    parser = argparse.ArgumentParser()
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
    
    COLLECTION = args.collection_name
    QDRANT_URL = args.qdrant_url
    
    # - Connect client    
    client = QdrantClient(url=QDRANT_URL)

    total = client.count(COLLECTION, exact=True).count
    pbar = tqdm(total=total, desc="Merging into _node_content.metadata", unit="pt")

    limit = 1000
    offset = None
    updated = 0

    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION,
            limit=limit,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        if not points:
            break

        updates = []
        for p in points:
            pbar.update(1)
            payload = p.payload or {}
            blob = payload.get("_node_content")
            extra = payload.get("metadata") or {}
            if not blob or not extra:
                continue

            try:
                node = json.loads(blob)
            except Exception:
                continue

            # node["metadata"] exists in LlamaIndex nodes
            node_meta = node.get("metadata") or {}
            merged = {**node_meta, **extra}

            # nothing to do?
            if merged == node_meta:
                continue

            node["metadata"] = merged
            updates.append((p.id, json.dumps(node, ensure_ascii=False)))

        # push updates in batches
        if updates:
            client.set_payload(
                collection_name=COLLECTION,
                payload={"_node_content": None},    # ensure key exists in schema
                points=[u[0] for u in updates],
            )
        
            # Qdrant set_payload allows dict-of-lists too; do one call per field for clarity
            client.set_payload(
                collection_name=COLLECTION,
                payload={"_node_content": [u[1] for u in updates]},
                points=[u[0] for u in updates],
            )
            updated += len(updates)

        if offset is None:
            break

    pbar.close()
    print(f"Updated _node_content for {updated} points.")
    
########################
##    RUN MAIN
########################
if __name__ == "__main__":
    main()
