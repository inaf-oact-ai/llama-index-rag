#!/usr/bin/env python
# -*- coding: utf-8 -*-

##########################
##   IMPORT MODULES
##########################
# - Import standard modules
import os
import time
import re
import argparse

# - Import streamlit
import requests
import streamlit as st

######################################
##  STREAMLIT CLI ARGS
######################################

def load_streamlit_args():
    """Parse command-line options passed after: streamlit run app.py -- ..."""

    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument(
        "--show_collection_summary",
        dest="show_collection_summary",
        action="store_true",
        help="Fetch and show collection statistics on the landing page.",
    )

    parser.add_argument(
        "--no_collection_summary",
        dest="show_collection_summary",
        action="store_false",
        help="Disable collection-statistics fetching on the landing page.",
    )

    parser.set_defaults(
        show_collection_summary=os.environ.get("RAG_SHOW_COLLECTION_SUMMARY", "0").lower()
        in {"1", "true", "yes", "on"}
    )

    args, _ = parser.parse_known_args()
    return args


APP_ARGS = load_streamlit_args()

######################################
##  PAGE CONFIG & LIGHT THEMING
######################################
#st.set_page_config(page_title="Radio RAG", page_icon="🔎", layout="wide")
st.set_page_config(page_title="Research RAG", page_icon="🔎", layout="wide")

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
####BANNER_LOGO = "share/radioRAG_banner.png"
####LOGO_URL = os.environ.get("RAG_LOGO_URL", BANNER_LOGO if os.path.exists(BANNER_LOGO) else None)
#LOGO_URL = os.environ.get(
#    "RAG_LOGO_URL",
#    "https://raw.githubusercontent.com/inaf-oact-ai/llama-index-rag/main/share/radioRAG_banner_v3.png"
#)
#print(f"LOGO_URL: {LOGO_URL}")



######################################
##  DOMAIN CONFIG
######################################

DOMAIN_CONFIG = {
    "radio": {
        "title": "Radio RAG",
        "subtitle": "AI-powered Retrieval-Augmented Generation for Radio Astronomy",
        "page_icon": "📡",
        "logo_url": os.environ.get(
            "RADIO_RAG_LOGO_URL",
            "https://raw.githubusercontent.com/inaf-oact-ai/llama-index-rag/main/share/radioRAG_banner_v3.png",
        ),
        "collections": ["radiopapers", "radiobooks", "annreviews"],
        "examples": [
            "Can you explain the difference between ultra ultra-compact (UC) and hyper-compact (HC) HII regions? How many HC and UC HIIs are currently known in radio?",
            "Do you know what is an Odd Radio Circle (ORC)? Describe its morphology and how they are detected in radio surveys.",
            "Can you summarize how many Supernova Remnants (SNR) are currently known, what fraction of them are detected in the radio band?",
        ],
        "placeholder": "Ask about radio astronomy papers, books, and reviews…",
    },
    "solar": {
        "title": "Solar RAG",
        "subtitle": "AI-powered Retrieval-Augmented Generation for Solar Physics",
        "page_icon": "☀️",
        "logo_url": os.environ.get(
            "SOLAR_RAG_LOGO_URL",
            "https://raw.githubusercontent.com/inaf-oact-ai/llama-index-rag/main/share/solarRAG_banner_v2.png",
        ),
        "collections": ["solar-living-reviews", "solar-papers", "annreviews"],
        "examples": [
            "What are the main observational signatures of solar flares?",
            "How are CMEs related to Forbush decreases?",
            "What features are commonly used for solar flare forecasting?",
        ],
        "placeholder": "Ask about solar physics papers and reviews…",
    },
    "exoplanets": {
        "title": "Exoplanets RAG",
        "subtitle": "AI-powered Retrieval-Augmented Generation for Exoplanet Science",
        "page_icon": "🪐",
        "logo_url": os.environ.get(
            "EXOPLANETS_RAG_LOGO_URL",
            "https://raw.githubusercontent.com/inaf-oact-ai/llama-index-rag/main/share/exoplanetsRAG_banner_v1.png",
        ),
        "collections": ["exoplanets-papers", "annreviews"],
        "examples": [
            "What are the main methods used to detect exoplanets?",
            "How are exoplanet atmospheres characterized from transit spectroscopy?",
            "What are hot Jupiters and why are they important for exoplanet science?",
        ],
        "placeholder": "Ask about exoplanet papers and reviews…",
    },
}


def get_domain_config(domain_key: str) -> dict:
    """Return a domain configuration, falling back to radio."""
    return DOMAIN_CONFIG.get(domain_key, DOMAIN_CONFIG["radio"])


def fetch_collection_summaries(api_base: str, enabled: bool = True) -> dict:
    """Fetch collection summaries from the backend."""

    if not enabled:
        return {}

    url = f"{api_base.rstrip('/')}/api/collections/summary"

    with st.spinner("Loading collection information from the RAG backend..."):
        try:
            r = requests.get(url, timeout=60)

            if r.status_code != 200:
                st.warning(f"Could not fetch collection summaries from {url}: HTTP {r.status_code}")
                st.caption(r.text[:500])
                return {}

            data = r.json()

            if data.get("status", -1) != 0:
                st.warning(
                    f"Backend returned collection-summary status={data.get('status')}: "
                    f"{data.get('message')}"
                )
                return {}

            summaries = {
                item.get("collection"): item
                for item in data.get("collections", [])
                if item.get("collection")
            }

            if not summaries:
                st.warning(f"Collection summary endpoint returned no collections: {url}")

            return summaries

        except Exception as e:
            st.warning(f"Could not fetch collection summaries from {url}: {e}")
            return {}



def format_collection_summary(collection_name: str, summaries: dict) -> str:
    """Format collection summary for the landing-page cards."""

    item = summaries.get(collection_name)
    if not item:
        return (
            f"<li style='margin-bottom:0.45rem;'>"
            f"<code>{collection_name}</code><br>"
            f"<span style='color:#94a3b8;'>summary unavailable</span>"
            f"</li>"
        )

    status = item.get("status", 0)
    if status != 0:
        msg = item.get("message", "unknown backend error")
        return (
            f"<li style='margin-bottom:0.45rem;'>"
            f"<code>{collection_name}</code><br>"
            f"<span style='color:#b91c1c;'>summary error: {msg}</span>"
            f"</li>"
        )

    ndocs = item.get("estimated_documents", 0)
    npoints = item.get("points_count", 0)
    ymin = item.get("year_min")
    ymax = item.get("year_max")

    if ymin is not None and ymax is not None:
        if ymin == ymax:
            year_text = f" · {ymin}"
        else:
            year_text = f" · {ymin}–{ymax}"
    else:
        year_text = ""

    return (
        f"<li style='margin-bottom:0.45rem;'>"
        f"<code>{collection_name}</code><br>"
        f"<span style='color:#475569;'>"
        f"{ndocs:,} documents · {npoints:,} chunks{year_text}"
        f"</span>"
        f"</li>"
    )


def render_domain_landing_page(api_base: str, show_collection_summary: bool = False):
    """Render the landing page used to select the RAG domain."""

    st.markdown("<h1 style='text-align:center;'>Research RAG</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='app-subtitle'>Select the scientific domain you want to query</p>",
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    
    collection_summaries = fetch_collection_summaries(
        api_base=api_base,
        enabled=show_collection_summary,
    )
    
    if show_collection_summary:
	      if collection_summaries:
		        st.caption(f"Loaded summaries for {len(collection_summaries)} collections.")
	  else:
		    st.warning("No collection summaries available. Check backend URL or server status.")

    cols = st.columns(len(DOMAIN_CONFIG))

    for col, (domain_key, cfg) in zip(cols, DOMAIN_CONFIG.items()):
        with col:
            if show_collection_summary:
                collection_items = "".join(
                    format_collection_summary(c, collection_summaries)
                    for c in cfg["collections"]
                )
                collections_html = (
                    "<div style=\"text-align:left;color:#475569;font-size:14px;margin-top:0.8rem;\">"
                    "<strong>Collections</strong>"
                    "<ul style=\"margin-top:0.35rem;padding-left:1.2rem;\">"
                    f"{collection_items}"
                    "</ul>"
                    "</div>"
                )
            else:
                collections_text = ", ".join(cfg["collections"])
                collections_html = (
                    "<p style=\"color:#475569;font-size:14px;margin-top:0.8rem;\">"
                    f"Collections: {collections_text}"
                    "</p>"
                )
	      
	      
	          card_html = (
                "<div style=\""
                "border:1px solid #e5e7eb;"
                "border-radius:18px;"
                "padding:1.25rem;"
                "text-align:center;"
                "background:#ffffff;"
                "box-shadow:0 4px 14px rgba(15,23,42,0.08);"
                "min-height:430px;"
                "\">"
                f"<div style=\"font-size:48px;margin-bottom:0.5rem;\">{cfg['page_icon']}</div>"
                f"<h2 style=\"margin-bottom:0.4rem;\">{cfg['title']}</h2>"
                f"<p style=\"color:#64748b;font-size:17px;\">{cfg['subtitle']}</p>"
                f"{collections_html}"
                "</div>"
            )
	      
            st.markdown(card_html, unsafe_allow_html=True)

            if st.button(f"Open {cfg['title']}", key=f"open_domain_{domain_key}", use_container_width=True):
                st.session_state["selected_domain"] = domain_key
                st.session_state.pop("prompt_seed", None)
                st.session_state.pop("autosubmit", None)
                st.rerun()

######################################
##  DOMAIN SELECTION
######################################
API_BASE_DEFAULT = os.environ.get("RAG_API_URL", "http://localhost:8000")

#if "selected_domain" not in st.session_state:
#    render_domain_landing_page()
#    st.markdown(
#        "<hr style='margin-top:3em;margin-bottom:1em;'>"
#        "<p style='text-align:center;color:black;font-size:18px;'>"
#        "© 2025 S. Riggi – INAF"
#        "</p>",
#        unsafe_allow_html=True,
#    )
#    st.stop()


if "selected_domain" not in st.session_state:
    if APP_ARGS.show_collection_summary:
        API_BASE = st.text_input(
            "Backend API base URL",
            value=API_BASE_DEFAULT,
            help="Your FastAPI base URL, without trailing slash.",
        )
    else:
        API_BASE = API_BASE_DEFAULT

    render_domain_landing_page(
        api_base=API_BASE,
        show_collection_summary=APP_ARGS.show_collection_summary,
    )

    st.markdown(
        "<hr style='margin-top:3em;margin-bottom:1em;'>"
        "<p style='text-align:center;color:black;font-size:18px;'>"
        "© 2026 S. Riggi – INAF"
        "</p>",
        unsafe_allow_html=True,
    )

    st.stop()

DOMAIN_KEY = st.session_state["selected_domain"]
DOMAIN = get_domain_config(DOMAIN_KEY)
LOGO_URL = DOMAIN.get("logo_url")

print(f"DOMAIN_KEY: {DOMAIN_KEY}")
print(f"LOGO_URL: {LOGO_URL}")
print(f"COLLECTIONS: {DOMAIN.get('collections')}")

######################################
##     SIDEBAR CONFIG
######################################
st.sidebar.header("Backend Settings")
#API_BASE = st.sidebar.text_input(
#    "API base URL",
#    value=os.environ.get("RAG_API_URL", "http://localhost:8000"),
#    help="Your FastAPI base URL (no trailing slash)",
#)
API_BASE = st.sidebar.text_input(
    "API base URL",
    value=API_BASE_DEFAULT,
    help="Your FastAPI base URL (no trailing slash)",
)

default_topk = int(os.environ.get("RAG_DEFAULT_TOPK", "5"))
st.sidebar.caption("Tip: set RAG_API_URL / RAG_DEFAULT_TOPK env vars to change defaults.")

st.sidebar.markdown("---")
st.sidebar.caption(f"Current domain: **{DOMAIN['title']}**")
st.sidebar.caption(f"Collections: `{', '.join(DOMAIN['collections'])}`")

if st.sidebar.button("Change RAG domain"):
    st.session_state.pop("selected_domain", None)
    st.session_state.pop("prompt_seed", None)
    st.session_state.pop("autosubmit", None)
    st.rerun()

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
        #st.markdown(f"<div class='banner-wrap'><img src='{LOGO_URL}' alt='Radio RAG banner' /></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='banner-wrap'><img src='{LOGO_URL}' alt='{DOMAIN['title']} banner' /></div>", unsafe_allow_html=True)
        
        ###st.markdown(f"<div class='banner-wrap'><img src='{LOGO_URL}' alt='Radio RAG banner' /></div>", unsafe_allow_html=True)
        ###st.image(LOGO_URL, width=1000)
        ###st.image(LOGO_URL, use_container_width=True)
        ###st.markdown("<style>img {max-width:1600px !important;}</style>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='font-size:40px'>🛰️</div>", unsafe_allow_html=True)
    
    #st.markdown(
    #    "<p class='app-subtitle'>AI-powered Retrieval-Augmented Generation for Radio Astronomy</p>",
    #    unsafe_allow_html=True,
    #)
    st.markdown(
        f"<p class='app-subtitle'>{DOMAIN['subtitle']}</p>",
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

def format_book_authors(authors: list[str]) -> str:
    """
    Convert a list of author strings from book-style metadata to arXiv-style citation.

    Example:
        ["M.D. Filipovic", "T. Caio"] -> "Filipovic, M.D. et al."
        ["M.D. Filipovic"] -> "Filipovic, M.D."
    """
    if not authors:
        return ""

    def normalize_name(name: str) -> str:
        # Split by whitespace and strip punctuation
        parts = name.strip().replace(",", "").split()
        if not parts:
            return ""
        surname = parts[-1]
        initials = "".join(parts[:-1])
        return f"{surname}, {initials}"

    formatted_first = normalize_name(authors[0])

    if len(authors) == 1:
        return formatted_first
    else:
        return f"{formatted_first} et al."


def format_annreview_author(name: str) -> str:
    """
    Format a single author string like:
        "Matt A. White" -> "M.A. White"
        "John D. P. Smith" -> "J.D.P. Smith"
        "M. D. Filipovic" -> "M.D. Filipovic"
        "Maria J. de Souza" -> "M.J. de Souza"
    """
    if not name or not isinstance(name, str):
        return ""

    parts = name.strip().split()
    if len(parts) == 0:
        return ""

    # Assume surname is last token; handle lowercase prefixes like "de", "van", "von"
    surname_parts = []
    given_parts = []
    for token in reversed(parts):
        if token[0].islower():  # part of surname prefix
            surname_parts.insert(0, token)
        elif not surname_parts:  # first uppercase-start word from the end → start surname
            surname_parts.insert(0, token)
        else:
            given_parts.insert(0, token)

    surname = " ".join(surname_parts)
    initials = ""
    for g in given_parts:
        # extract first letter if alphabetic
        if len(g) > 0 and g[0].isalpha():
            initials += g[0].upper() + "."
        # handle tokens already like "A." (keep as "A.")
        elif re.match(r"^[A-Z]\.$", g):
            initials += g
    return f"{initials} {surname}".strip()

def format_annreview_authors(authors: list[str]) -> str:
    """ Format annual review authors """
    if not authors:
        return ""
    authors_bookformat = [format_annreview_author(a) for a in authors if a]
    authors_formatted= format_book_authors(authors_bookformat)

    return authors_formatted

######################################
##     APP BODY
######################################

# --- Examples (click to auto-fill & submit) ---
#EXAMPLES = [
#    "Can you explain the difference between ultra ultra-compact (UC) and hyper-compact (HC) HII regions? How many HC and UC HIIs are currently known in radio?",
#    "Do you know what is an Odd Radio Circle (ORC)? Describe its morphology and how they are detected in radio surveys.",
#    "Can you summarize how many Supernova Remnants (SNR) are currently known, what fraction of them are detected in the radio band?"
#]

EXAMPLES = DOMAIN["examples"]

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
        #placeholder="Ask about radio astronomy papers…"
        placeholder=DOMAIN["placeholder"]
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
    #payload = {"query": prompt, "similarity_top_k": int(top_k)}
    payload = {
        "query": prompt,
        "similarity_top_k": int(top_k),
        "domain": DOMAIN_KEY,
        "collections": DOMAIN["collections"],
    }

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
            authors = meta.get("authors") or meta.get("author") or meta.get("first_author") or display_name
            citation = _journal_citation(meta)
            url = meta.get("url")
            download_url = meta.get("download_url")
            
            if doctype=="arxiv":
                # - Extract author & citation
                author = _first_author(meta)
                authors = meta.get("authors")
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
                authors= format_book_authors(meta.get("authors")) 
                citation = _book_citation(meta)
                
                # - Retrieve book URL
                url= meta.get("url")
                
                # - Retrieve download link (if available)
                download_url= meta.get("download_url")
                
            elif doctype=="annual-review":
                # - Extract author & citation
                author = meta.get("first_author")
                authors = format_annreview_authors(meta.get("authors"))
                citation = _journal_citation(meta)  # title, journal, vol(issue), pages, year

                # - Retrieve URL
                url= meta.get("url")
                
                # - Retrieve download link (if available)
                download_url= meta.get("download_url")  
                
            else:
                #print(f"WARN: Unknown doctype retrieved {doctype}, will set empty link field!")
                print(f"WARN: Unknown doctype retrieved {doctype}, using generic metadata formatting.")
                
            # - Set link fields
            link_html = f"<a class='paper-link' href='{url}' target='_blank'>[LINK]</a>" if url else ""
            download_html = (f"<a class='paper-link' href='{download_url}' target='_blank'>[DOWNLOAD]</a>" if download_url else "")
            
            print(f"meta: {meta}")
            print(f"url: {url}")
            print(f"download_url: {download_url}")

            # - Set score badge
            score_html = f"<span class='score-badge {_score_class(score)}'>{_score_label(score)}</span>"

            # - Set citation
            #details = []
            #if author or citation:
            #    who_what = ", ".join([x for x in [author, citation] if x])
            #    if who_what:
            #        details.append(who_what)
            
            details = [citation]
            if page:
                details.append(f"p. {page}")

            extra_html = " • ".join(details)
            #extra_html = (" — " + extra_html) if extra_html else ""

            
            st.markdown(
                #f"<div class='ref-line'><strong>{i}. {display_name}</strong>{extra_html} • "
                f"<div class='ref-line'>[{i}] <strong> {authors} </strong>, {extra_html} — "
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

