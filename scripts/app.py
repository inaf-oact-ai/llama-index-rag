#!/usr/bin/env python
# -*- coding: utf-8 -*-

##########################
##   IMPORT MODULES
##########################
# - Import standard modules
import os
import time

# - Import streamlit
import requests
import streamlit as st

st.set_page_config(page_title="RAG Tester", page_icon="🔎", layout="centered")

# --- Sidebar config ---
st.sidebar.header("Backend Settings")
API_BASE = st.sidebar.text_input(
    "API base URL",
    value=os.environ.get("RAG_API_URL", "http://localhost:8000"),
    help="Your FastAPI base URL (no trailing slash)",
)
default_topk = int(os.environ.get("RAG_DEFAULT_TOPK", "3"))
st.sidebar.caption("Tip: set RAG_API_URL / RAG_DEFAULT_TOPK env vars to change defaults.")

st.title("🔎 RAG Web Tester")

# --- Form ---
with st.form(key="query_form"):
    prompt = st.text_area("Your question", height=120, placeholder="Ask about radio astronomy papers…")
    top_k = st.slider("Similarity top-k", min_value=1, max_value=5, value=default_topk, step=1)
    submitted = st.form_submit_button("Search")

if submitted:
    if not prompt.strip():
        st.warning("Please enter a question.")
        st.stop()

    url = f"{API_BASE}/api/search"
    payload = {"query": prompt, "similarity_top_k": int(top_k)}

    with st.spinner("Querying RAG service…"):
        try:
            t0 = time.time()
            r = requests.post(url, json=payload, timeout=300)
            latency = time.time() - t0
        except Exception as e:
            st.error(f"Failed to reach backend: {e}")
            st.stop()

    if r.status_code != 200:
        st.error(f"Backend error {r.status_code}: {r.text}")
        st.stop()

    try:
        data = r.json()
    except Exception:
        st.error("Response is not valid JSON.")
        st.stop()

    # --- Main answer ---
    st.subheader("Answer")
    answer = data.get("search_result", "").strip()
    content_found = bool(data.get("content_found", False))
    status = data.get("status", 0)

    if status != 0:
        st.error(f"Service reported an error (status={status}).")
    if not content_found:
        st.info("No relevant content found (below similarity threshold).")

    st.write(answer or "_(empty response)_")
    st.caption(f"Latency: {latency:.2f}s  •  top-k={top_k}")

    # --- References ---
    st.subheader("References (by similarity)")
    sources = data.get("sources", []) or []

    # Sort by score (descending), keep only the fields you asked for:
    def _basename(fp: str | None, fallback: str | None) -> str:
        if isinstance(fp, str) and fp:
            return os.path.basename(fp)
        return (fallback or "").strip() or "(unknown)"

    sources_sorted = sorted(
        sources,
        key=lambda s: (s.get("score") is not None, s.get("score", 0.0)),
        reverse=True,
    )

    if not sources_sorted:
        st.caption("No references.")
    else:
        for i, src in enumerate(sources_sorted, 1):
            meta = src
            file_name = meta.get("file_name")
            file_path = meta.get("file_path")
            page = meta.get("page_label")
            score = meta.get("score")

            display_name = file_name or _basename(file_path, file_name)
            # Show filename (no basedir), page and similarity score
            st.markdown(f"**{i}. {display_name}**  — page {page}, score {score:.3f}" if score is not None else
                        f"**{i}. {display_name}**  — page {page}")
