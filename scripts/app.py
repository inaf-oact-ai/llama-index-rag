#!/usr/bin/env python
# -*- coding: utf-8 -*-

##########################
##   IMPORT MODULES
##########################
# - Import standard modules
import os
import time
import re

# - Import streamlit
import requests
import streamlit as st

######################################
##  PAGE CONFIG & LIGHT THEMING
######################################
st.set_page_config(page_title="Radio RAG", page_icon="🔎", layout="wide")

# [CHANGED] Global typography and layout improvements
st.markdown(
    """
    <style>
  /* [CHANGED] Larger, more legible base typography */
  /* html, body { font-size: 20px; }  */
  /* .main { padding-top: 1.25rem; font-size: 20px; } */
  html, body { font-size: 17px; }
  .main { padding-top: 1.0rem; font-size: 17px; }
  
  h1, h2, h3 { font-weight: 800; }
  h1 { font-size: 40px !important; }
  h2 { font-size: 24px !important; }
  /* h1 { font-size: 48px !important; } */
  /* h2 { font-size: 28px !important; } */
  .stMarkdown, .stTextInput, .stTextArea, label { font-size: 20px !important; }

  /* [NEW] Larger sidebar text */
  /* section[data-testid="stSidebar"] * { font-size: 18px !important; } */
  section[data-testid="stSidebar"] * { font-size: 16px !important; }


  /* [NEW] Make the prompt textarea light silver */
  .stTextArea textarea {
    background-color: #eef2f5 !important; /* light silver */
    border-radius: 10px !important;
    color: #0f172a !important;
  }

  /* Reference line + badges */
  /* .ref-line { margin-bottom: .65rem; font-size: 19px; } */
  .ref-line { margin-bottom: .65rem; font-size: 16px; }
  .score-badge { font-weight: 700; padding: 6px 12px; border-radius: 999px; border: 1px solid rgba(0,0,0,0.06); font-size: 15px; }
  /* .score-badge { font-weight: 700; padding: 6px 12px; border-radius: 999px; border: 1px solid rgba(0,0,0,0.06); font-size: 18px; } */
  
  .score-red { background:#ffe5e5; color:#b91c1c; }
  .score-orange { background:#fff0df; color:#c2410c; }
  .score-yellow { background:#fff9db; color:#92400e; }
  .score-green { background:#e7f7e7; color:#166534; }
  .paper-link { text-decoration:none; font-weight:600; border-bottom:1px dashed rgba(2,132,199,0.35); }

  /* [NEW] Center the banner and let it grow up to a large max width */
  .banner-wrap { display:flex; justify-content:center; }
  .banner-wrap img { width:100%; max-width: 700px; height:auto; }
  .app-subtitle { text-align:center; color:gray; font-size:22px; margin-top:4px; }
  \* .app-subtitle { text-align:center; color:gray; font-size:19px; margin-top:4px; } */
 
</style>
    """,
    unsafe_allow_html=True,
)

######################################
##  LOGO CONFIG
######################################
# [CHANGED] use larger horizontal banner
#BANNER_LOGO = "share/radioRAG_banner.png"
#LOGO_URL = os.environ.get("RAG_LOGO_URL", BANNER_LOGO if os.path.exists(BANNER_LOGO) else None)
LOGO_URL = os.environ.get(
    "RAG_LOGO_URL",
    "https://raw.githubusercontent.com/inaf-oact-ai/llama-index-rag/main/share/radioRAG_banner_v3.png"
)
print(f"LOGO_URL: {LOGO_URL}")


######################################
##     SIDEBAR CONFIG
######################################

st.sidebar.header("Backend Settings")
API_BASE = st.sidebar.text_input(
    "API base URL",
    value=os.environ.get("RAG_API_URL", "http://localhost:8000"),
    help="Your FastAPI base URL (no trailing slash)",
)

default_topk = int(os.environ.get("RAG_DEFAULT_TOPK", "5"))
st.sidebar.caption("Tip: set RAG_API_URL / RAG_DEFAULT_TOPK env vars to change defaults.")


######################################
##  HEADER SECTION
######################################
#with st.container():
#    cols = st.columns([1, 9])
#    with cols[0]:
#        if LOGO_URL:
#            st.image(LOGO_URL, use_container_width=True)
#        else:
#            st.markdown("<div style='font-size:40px'>🛰️</div>", unsafe_allow_html=True)
#    with cols[1]:
#        st.markdown(
#            """
#            <div class="app-header">
#              <div>
#                <div class="brand-title">Radio RAG </div>
#                <div class="brand-sub">Search your radio RAG backend and inspect retrieved references</div>
#              </div>
#            </div>
#            """,
#            unsafe_allow_html=True,
#        )


with st.container():
    # [CHANGED] Always show a clear title
    #st.markdown("<h1 style='text-align:center;'>Radio RAG</h1>", unsafe_allow_html=True)

    # [CHANGED] Banner centered and allowed to scale wide
    if LOGO_URL:
        st.markdown(f"<div class='banner-wrap'><img src='{LOGO_URL}' alt='Radio RAG banner' /></div>", unsafe_allow_html=True) 
        #st.markdown(f"<div class='banner-wrap'><img src='{LOGO_URL}' alt='Radio RAG banner' /></div>", unsafe_allow_html=True)
        #st.image(LOGO_URL, width=1000)
        #st.image(LOGO_URL, use_container_width=True)
        #st.markdown("<style>img {max-width:1600px !important;}</style>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='font-size:40px'>🛰️</div>", unsafe_allow_html=True)
    
    st.markdown(
        "<p class='app-subtitle'>AI-powered Retrieval-Augmented Generation for Radio Astronomy</p>",
        unsafe_allow_html=True,
    )

######################################
##     HELPER METHODS
######################################

def _basename(fp: str | None, fallback: str | None) -> str:
    if isinstance(fp, str) and fp:
        return os.path.basename(fp)
    return (fallback or "").strip() or "(unknown)"


def _first_author(meta: dict) -> str | None:
    # try a few common keys
    candidates = [
        meta.get("first_author"),
        meta.get("author"),
        meta.get("authors"),
        meta.get("creator"),
    ]
    raw = next((c for c in candidates if isinstance(c, str) and c.strip()), None)
    if not raw:
        return None
    s = raw.strip()

    # split on common separators: semicolon, ' and ', ampersand, commas with "and"
    parts = re.split(r"\s*;\s*|\s+and\s+|&|,\s*(?=[A-Z][a-z])", s)
    first = parts[0].strip() if parts else s

    # if it's "Last, First", normalize to "Last, F." (optional)
    if "," in first:
        last, firsts = [t.strip() for t in first.split(",", 1)]
        initials = " ".join([f[0] + "." for f in firsts.split() if f])
        return f"{last}, {initials}" if initials else last
    return first


def _journal_citation(meta: dict) -> str:
    # Accept a variety of keys
    title   = meta.get("title") or meta.get("paper_title") or meta.get("document_title")
    journal = meta.get("journal") or meta.get("journal_name") or meta.get("container_title") or meta.get("publication")
    volume  = meta.get("volume") or meta.get("vol")
    issue   = meta.get("issue") or meta.get("number") or meta.get("no")
    pages   = meta.get("pages") or meta.get("page") or meta.get("page_range")
    year    = meta.get("year") or meta.get("pub_year") or meta.get("date") or meta.get("publication_year")

    # try to extract a 4-digit year from date strings
    if isinstance(year, str):
        m = re.search(r"(19|20)\d{2}", year)
        year = m.group(0) if m else year

    bits = []
    if title:   bits.append(f"“{title}”")
    if journal: bits.append(journal)
    vol_issue = None
    if volume and issue:
        vol_issue = f"{volume}({issue})"
    elif volume:
        vol_issue = str(volume)
    if vol_issue: bits.append(vol_issue)
    if pages:   bits.append(pages)
    if year:    bits.append(str(year))
    return ", ".join(bits)


def _book_citation(meta: dict) -> str:
    # Accept a variety of keys
    title   = meta.get("title") or meta.get("paper_title") or meta.get("document_title")
    publisher = meta.get("publisher")
    pages   = meta.get("pages")
    year    = meta.get("year")

    # try to extract a 4-digit year from date strings
    if isinstance(year, str):
        m = re.search(r"(19|20)\d{2}", year)
        year = m.group(0) if m else year

    bits = []
    if title:   bits.append(f"“{title}”")
    if publisher: bits.append(publisher)
    if pages:   bits.append(pages)
    if year:    bits.append(str(year))
    return ", ".join(bits)

def _score_class(score: float | None) -> str:
    if score is None:
        return "score-yellow"
    if score < 0.3:
        return "score-red"
    if score < 0.5:
        return "score-orange"
    if score < 0.7:
        return "score-yellow"
    return "score-green"


def _score_label(score: float | None) -> str:
    if score is None:
        return "n/a"
    return f"{score:.3f}"


def _arxiv_id_from_meta(meta: dict) -> str | None:
    # common metadata keys where an arXiv id might live
    for k in ["arxiv_id", "arXiv", "arxiv", "eprint_id", "eprint", "arxivId", "identifier"]:
        val = meta.get(k)
        if isinstance(val, str) and val.strip():
            # normalize something like "arXiv:2101.01234v2" to an abs-URL-friendly id
            v = val.strip()
            # often metadata contains full URLs already
            if v.startswith("http://") or v.startswith("https://"):
                if "arxiv.org" in v:
                    # return id segment after /abs/ or /pdf/
                    m = re.search(r"arxiv\.org/(abs|pdf)/([^?#/]+)", v)
                    if m:
                        return m.group(2).replace(".pdf", "")
                    return None
                # non-arxiv URL is not useful for arxiv link
                continue
            v = re.sub(r"^arxiv:\s*", "", v, flags=re.IGNORECASE)
            return v
    return None


def _arxiv_url(meta: dict) -> str | None:
    aid = _arxiv_id_from_meta(meta)
    if not aid:
        return None
    return f"https://arxiv.org/abs/{aid}"


######################################
##     APP BODY
######################################

# --- Examples (click to auto-fill & submit) ---
EXAMPLES = [
    "Can you explain the difference between ultra ultra-compact (UC) and hyper-compact (HC) HII regions? How many HC and UC HIIs are currently known in radio?",
    "Do you know what is an Odd Radio Circle (ORC)? Describe its morphology and how they are detected in radio surveys.",
    "Can you summarize how many Supernova Remnants (SNR) are currently known, what fraction of them are detected in the radio band?"
]

st.info("Enter a question and click *Search*. Below, you find some examples. *Click* one of them for testing.")
#st.markdown("**Below, you find some example test queries. Click one of them:**")
_cols = st.columns(len(EXAMPLES))
for _i, _ex in enumerate(EXAMPLES):
    # show button label in italic
    if _cols[_i].button(f"_{_ex}_", key=f"ex_{_i}"):
        st.session_state["prompt_seed"] = _ex
        st.session_state["autosubmit"] = True

# --- Form ---
with st.form(key="query_form"):
    #prompt = st.text_area("Your question", height=120, placeholder="Ask about radio astronomy papers…")
    prompt = st.text_area(
        "Your question",
        height=120,
        value=st.session_state.get("prompt_seed", ""),
        placeholder="Ask about radio astronomy papers…"
    )
    
    top_k = st.slider("Similarity top-k", min_value=1, max_value=10, value=default_topk, step=1)
    submitted = st.form_submit_button("Search")
    
    # Auto-submit if an example button was clicked
    if st.session_state.pop("autosubmit", False) and not submitted:
        submitted = True

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

    sources_sorted = sorted(
        sources,
        key=lambda s: (s.get("score") is not None, s.get("score", 0.0)),
        reverse=True,
    )

    if not sources_sorted:
        st.caption("No references.")
    else:
        for i, src in enumerate(sources_sorted, 1):
            score = src.get("score")
            meta  = src  # assuming your backend flattens node.metadata into the source dict

            # - Check source doc type
            doctype= meta.get("doctype")

            # - Get basic file info
            file_name = meta.get("file_name")
            file_path = meta.get("file_path")
            display_name = file_name or _basename(file_path, file_name)

            # - Get page
            page = meta.get("page_label") or meta.get("page")
            
            # - Extract rich bibliographic info
            if doctype=="arxiv":
                # - Extract author & citation
                author = _first_author(meta)
                citation = _journal_citation(meta)  # title, journal, vol(issue), pages, year

                # - Retrieve arXiv URL (if any)
                url= meta.get("arxiv_abs_url")
                if url is None: # try to get url from arxiv_id
                    url = _arxiv_url(meta)
            
                # - Retrieve download link (if available)
                download_url= meta.get("arxiv_pdf_url")
                        
            elif doctype=="book":
                # - Extract author
                author = meta.get("first_author")
                citation = _book_citation(meta)
                
                # - Retrieve book URL
                url= meta.get("url")
                
                # - Retrieve download link (if available)
                download_url= meta.get("download_url")
                
            else:
                print(f"WARN: Unknown doctype retrieved {doctype}, will set empty link field!")
                
            # - Set link fields
            link_html = f"<a class='paper-link' href='{url}' target='_blank'>[LINK]</a>" if url else ""
            download_html = (f"<a class='paper-link' href='{download_url}' target='_blank'>[DOWNLOAD]</a>" if download_url else "")
            
            print(f"meta: {meta}")
            print(f"url: {url}")
            print(f"download_url: {download_url}")

            # - Set score badge
            score_html = f"<span class='score-badge {_score_class(score)}'>{_score_label(score)}</span>"

            # - Set citation
            details = []
            if author or citation:
                who_what = ", ".join([x for x in [author, citation] if x])
                if who_what:
                    details.append(who_what)
            if page:
                details.append(f"p. {page}")

            extra_html = " • ".join(details)
            extra_html = (" — " + extra_html) if extra_html else ""

            
            st.markdown(
                f"<div class='ref-line'><strong>{i}. {display_name}</strong>{extra_html} • "
                f"score {score_html} "
                + (f" • {link_html}" if link_html else "")
                + (f" • {download_html}" if download_html else "")
                + "</div>",
                unsafe_allow_html=True,
            )

#else:
#    st.info("Enter a question above and click *Search* to test your RAG service.")
    
st.markdown(
    "<hr style='margin-top:3em;margin-bottom:1em;'>"
    "<p style='text-align:center;color:black;font-size:18px;'>"
    "© 2025 S. Riggi – INAF"
    "</p>",
    unsafe_allow_html=True,
)

