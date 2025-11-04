#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import re
import requests
import streamlit as st

######################################
##  PAGE CONFIG & STYLE
######################################
st.set_page_config(page_title="Radio RAG", page_icon="🔎", layout="centered")

# [CHANGED] Simplified clean style without blue background box
st.markdown(
    """
    <style>
      body { background-color: #fafafa; }
      .main { padding-top: 1rem; }
      .ref-line { margin-bottom: .35rem; }
      .score-badge { font-weight: 700; padding: 2px 8px; border-radius: 999px; border: 1px solid rgba(0,0,0,0.06); }
      .score-red { background:#ffe5e5; color:#b91c1c; }
      .score-orange { background:#fff0df; color:#c2410c; }
      .score-yellow { background:#fff9db; color:#92400e; }
      .score-green { background:#e7f7e7; color:#166534; }
      .paper-link { text-decoration:none; font-weight:600; border-bottom:1px dashed rgba(2,132,199,0.35); }
    </style>
    """,
    unsafe_allow_html=True,
)

######################################
##  LOGO CONFIG
######################################
# [CHANGED] use horizontal banner
BANNER_LOGO = "share/radioRAG_banner.png"
LOGO_URL = os.environ.get("RAG_LOGO_URL", BANNER_LOGO if os.path.exists(BANNER_LOGO) else None)

######################################
##  HEADER SECTION
######################################
with st.container():
    if LOGO_URL:
        # [CHANGED] Display wide banner at top center
        st.image(LOGO_URL, use_container_width=True)
    else:
        st.markdown("<h1 style='text-align:center;'>Radio RAG</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center;color:gray;'>AI-powered Retrieval-Augmented Generation for Radio Astronomy</p>",
        unsafe_allow_html=True,
    )

######################################
##  SIDEBAR
######################################
st.sidebar.header("Backend Settings")
API_BASE = st.sidebar.text_input(
    "API base URL",
    value=os.environ.get("RAG_API_URL", "http://localhost:8000"),
    help="Your FastAPI base URL (no trailing slash)",
)
default_topk = int(os.environ.get("RAG_DEFAULT_TOPK", "10"))
st.sidebar.caption("Tip: set RAG_API_URL / RAG_DEFAULT_TOPK env vars to change defaults.")

######################################
##  HELPERS
######################################
def _basename(fp: str | None, fallback: str | None) -> str:
    if isinstance(fp, str) and fp:
        return os.path.basename(fp)
    return (fallback or "").strip() or "(unknown)"

def _score_class(score):
    if score is None: return "score-yellow"
    if score < 0.3: return "score-red"
    if score < 0.5: return "score-orange"
    if score < 0.7: return "score-yellow"
    return "score-green"

def _score_label(score):
    if score is None: return "n/a"
    return f"{score:.3f}"

def _arxiv_url(meta):
    for k in ["arxiv_id", "arXiv", "arxiv", "eprint_id", "eprint", "identifier"]:
        val = meta.get(k)
        if isinstance(val, str) and val.strip():
            v = re.sub(r"^arxiv:\s*", "", val.strip(), flags=re.IGNORECASE)
            return f"https://arxiv.org/abs/{v}"
    return None

######################################
##  MAIN APP BODY
######################################
with st.form(key="query_form"):
    prompt = st.text_area("Your question", height=120, placeholder="Ask about radio astronomy papers…")
    top_k = st.slider("Similarity top-k", min_value=1, max_value=10, value=default_topk, step=1)
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

    st.subheader("Answer")
    st.write(data.get("search_result", "_empty response_"))
    st.caption(f"Latency: {latency:.2f}s  •  top-k={top_k}")

    st.subheader("References (by similarity)")
    for i, src in enumerate(data.get("sources", []), 1):
        score = src.get("score")
        meta = src
        file_name = meta.get("file_name") or _basename(meta.get("file_path"), None)
        arxiv_url = _arxiv_url(meta)
        link_html = f"<a class='paper-link' href='{arxiv_url}' target='_blank'>[LINK]</a>" if arxiv_url else ""
        score_html = f"<span class='score-badge {_score_class(score)}'>{_score_label(score)}</span>"
        st.markdown(f"<div class='ref-line'><strong>{i}. {file_name}</strong> • score {score_html} {link_html}</div>", unsafe_allow_html=True)
else:
    st.info("Enter a question above and click *Search* to test your RAG service.")

