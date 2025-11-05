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
##    RAG CLASS
#############################
collection_descriptions = {
    "radiopapers": "Scientific papers stored in ArXiV repository with subject keywords related to radio astronomy.",
    "radiobooks": "Text books with subjects related to radio astronomy.",
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

class Response(BaseModel):
    search_result: str
    content_found: bool
    status: int
    sources: list

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
        "file_path": md.get("file_path"),
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
        "file_path": md.get("file_path"),
        "file_name": md.get("file_name"),
        "page_label": md.get("page_label"),
        # - Add custom metadata fields
        "title": md.get("title"),
        "subtitle": md.get("subtitle"),
        "publisher": md.get("publisher"),
        "year": md.get("year"),
        "pages": md.get("pages"),
        "authors": md.get("authors"),
        "first_author": md.get("authors")[0],
        "doi": md.get("doi"),
        "isbn": md.get("isbn"),
        "issn": md.get("issn"),
        "url": md.get("url"),
        "download_url": md.get("download_url"),
        "text": sn.node.get_content(),
    }
    
    return source

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
    parser.add_argument("-collection_names", "--collection_names", type=str, required=False, default="radiopapers", help="Comma-separated list of Qdrant collection names to query across (overrides --collection_name)")
    parser.add_argument("-similarity_thr", "--similarity_thr", type=float, required=False, default=0.5, help="Similarity threshold")
    parser.add_argument("-llm", "--llm", type=str, required=False, default="", help="LLM model name")
    parser.add_argument("-llm_url", "--llm_url", type=str, required=False, default="http://localhost:11434", help="LLM ollama url")
    parser.add_argument("-llm_ctx_window", "--llm_ctx_window", type=int, required=False, default=4096, help="LLM context window")
    parser.add_argument("-llm_timeout", "--llm_timeout", type=int, required=False, default=120, help="LLM response timeout in seconds")
    parser.add_argument("-llm_keep_alive", "--llm_keep_alive", type=str, required=False, default="0s", help="LLM keep alive model option. 0s means loaded for all the time duration. 1h means active for 1h")
    parser.add_argument("--llm_thinking", dest="llm_thinking", action='store_true',help='Enable LLM thinking (default=False)')	
    parser.set_defaults(llm_thinking=False)
    parser.add_argument("-qdrant_url", "--qdrant_url", type=str, required=False, default="http://localhost:6333", help="QDRant URL")
    #parser.add_argument("-select_top_k", "--select_top_k", type=int, required=False, default=10, help="Max number of collection source tools used in multi-source retrieval")
    
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
    logger.info(f"Loading model {args.llm} ...")
    llm = Ollama(
      model=args.llm,
      base_url=args.llm_url,
      request_timeout=args.llm_timeout,
      context_window=args.llm_ctx_window,
      keep_alive=args.llm_keep_alive,
      thinking=args.llm_thinking,
      #additional_kwargs={
      #    "num_ctx": 4096,
      #    "num_batch": 128
      #}
    )

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

    @app.post("/api/search", response_model=Response, status_code=200)
    def search(query: Query):

        logger.info(f"Received query: {query}")
        #try:
        #    query_engine = index.as_query_engine(
        #        similarity_top_k=query.similarity_top_k,
        #        output=Response,
        #        response_mode="tree_summarize",
        #        include_metadata=True,
        #        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=args.similarity_thr)],
        #        verbose=True,
        #    )
        #except Exception as e:
        #    logger.error(f"Failed to retrieve query engine (err={str(e)})!")
        #    err_resp= Response(status=-1, search_result="Failed to retrieve query engine", sources=[], content_found=False)
        #    return err_resp   
            
        try:
            if not multi_mode:
                query_engine = index.as_query_engine(
                    similarity_top_k=query.similarity_top_k,
                    output=Response,
                    response_mode="tree_summarize",
                    include_metadata=True,
                    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=args.similarity_thr)],
                    verbose=True,
                )
            else:
                # Build a tool per collection with the same retrieval settings
                # NB: Some LLM models (e.g. Deepseek does not support tools, so we switched to a different strategy
                try:
                    tools = []
                    for cname, idx in indices.items():
                        eng = idx.as_query_engine(
                            similarity_top_k=query.similarity_top_k,
                            response_mode="tree_summarize",
                            include_metadata=True,
                            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=args.similarity_thr)],
                            verbose=True,
                        )
                        desc = collection_descriptions.get(cname, f"Qdrant collection '{cname}'")
                        tools.append(
                            QueryEngineTool.from_defaults(
                                query_engine=eng,
                                description=desc
                            )
                        )    
        
                    # Let the LLM route/merge across collections; select_top_k lets it pick multiple sources
                    query_engine = RouterQueryEngine(
                        llm=rag.llm,
                        selector=PydanticMultiSelector.from_defaults(),
                        query_engine_tools=tools,
                        verbose=True,
                    )
                except Exception as e:
                    logger.warning(f"Failed to run router query engine (err={str(e)}), trying with query fusion retriever ...")
                    
                    retrievers = [
                        idx.as_retriever(similarity_top_k=query.similarity_top_k)
                        for idx in indices.values()
                    ]

                    # - Fuse results (simple union + dedupe; 1 query => no query expansion)
                    fusion = QueryFusionRetriever(
                        retrievers=retrievers,
                        similarity_top_k=query.similarity_top_k,  # per-retriever top_k
                        num_queries=1,                             # keep 1 to disable expansion
                        # mode="reciprocal_rerank",                # optional: stronger RRF re-ranking
                        use_async=True,
                        verbose=True,
                    )

                    # - Wrap in a QueryEngine
                    query_engine = RetrieverQueryEngine.from_args(     
                        fusion,
                        response_mode="tree_summarize",      # synthesis strategy
                        include_metadata=True,               # preserve node metadata
                        output=Response,
                        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=args.similarity_thr)],
                    )
                    
        except Exception as e:
            logger.error(f"Failed to retrieve query engine (err={str(e)})!")
            err_resp = Response(status=-1, search_result="Failed to retrieve query engine", sources=[], content_found=False)
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
            #search_result= str(response).strip()
            #print("response.metadata")
            #print(response.metadata)
            #sources= [response.metadata[k]["file_path"] for k in response.metadata.keys()]
            #source= sources[0]

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
            response_sources = []
            for sn in getattr(response, "source_nodes", []):
                md = sn.node.metadata or {}
                print("--> md")
                print(md)
                
                # - Check if source is a book
                doctype= md.get("kind")
                source= None
                if doctype:
                    if doctype=="book":
                        # - Book
                        source= get_book_metadata(sn, md)
                    else:
                        logger.warning(f"Unknown doctype parsed ({doctype}), skipping entry ...")
                        continue
                else:
                    # - Arxiv paper
                    source= get_arxiv_metadata(sn, md)
                
                if source is None:
                    logger.warning("Response source parsed is still None, skipping entry ...")
                    continue
                    
                # - Add response source to list
                response_sources.append(source)
                
                # - Set arxiv id
                # --- arXiv extraction & normalization ---
                #arxiv_raw = (
                #    md.get("arxiv_id")
                #    or md.get("arXiv")
                #    or md.get("arxiv")
                #    or md.get("eprint")
                #    or md.get("identifier")
                #)
                #arxiv_norm = None
                #if isinstance(arxiv_raw, str):
                #    import re as _re
                #    # strip leading "arXiv:" and try to capture 2011.07620 (ignore version)
                #    s = _re.sub(r"^arXiv:\s*", "", arxiv_raw.strip(), flags=_re.IGNORECASE)
                #    m = _re.search(r"(\d{4}\.\d{4,5})", s)
                #    arxiv_norm = m.group(1) if m else s

                # fallback: derive from file_name like 2011.07620v2.pdf
                #if not arxiv_norm:
                #    fn = (md.get("file_name") or "").strip()
                #    import re as _re
                #    m = _re.search(r"(\d{4}\.\d{4,5})(?:v\d+)?\.pdf$", fn)
                #    if m:
                #        arxiv_norm = m.group(1)

                #arxiv_abs_url = f"https://arxiv.org/abs/{arxiv_norm}" if arxiv_norm else None
                #arxiv_pdf_url = f"https://arxiv.org/pdf/{arxiv_norm}.pdf" if arxiv_norm else None
                
                # - Set response sources
                #response_sources.append({
                #    "node_id": sn.node.node_id,
                #    "score": sn.score,                     # similarity score
                #    "file_path": md.get("file_path"),
                #    "file_name": md.get("file_name"),
                #    "page_label": md.get("page_label"),
                #    # pass through biblio fields if you have them:
                #    "title": md.get("title"),
                #    "paper_title": md.get("paper_title"),
                #    "document_title": md.get("document_title"),
                #    "authors": md.get("authors"),
                #    "author": md.get("author"),
                #    "first_author": md.get("first_author"),
                #    "journal": md.get("journal"),
                #    "journal_name": md.get("journal_name"),
                #    "container_title": md.get("container_title"),
                #    "publication": md.get("publication"),
                #    "volume": md.get("volume"),
                #    "issue": md.get("issue"),
                #    "number": md.get("number"),
                #    "pages": md.get("pages"),
                #    "page": md.get("page"),
                #    "year": md.get("year"),
                #    "pub_year": md.get("pub_year"),
                #    "date": md.get("date"),
                #    "bibcode": md.get("bibcode"),
                #    "doi": md.get("doi"),
                #    "arxiv_id": arxiv_norm,
                #    "arxiv_abs_url": arxiv_abs_url,
                #    "arxiv_pdf_url": arxiv_pdf_url,
                #    "text": sn.node.get_content(),
                #})
                
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
            err_resp= Response(status=-1, search_result="Failed to parse response", sources=[])
            return err_resp

        return response_object

    # - Running server
    logger.info("Running app ...")
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
