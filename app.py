"""Marketing Analytics RAG Agent — by Aneela Veldi"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from PIL import Image
from src.vector_store import index_exists, index_size, build_index
from src.agent import retrieve, ask_stream, get_followups
from src.data_loader import load_corpus
from src.embeddings import embed_texts

_favicon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "favicon.png")
_favicon = Image.open(_favicon_path) if os.path.exists(_favicon_path) else "◆"

st.set_page_config(
    page_title="Marketing Analytics AI",
    page_icon=_favicon,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Space+Grotesk:wght@500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }

.stApp {
  background: #04040d;
  color: #ddddf5;
  font-family: 'Inter', sans-serif;
}
.block-container { padding-top: 0 !important; max-width: 1200px; }
header[data-testid="stHeader"] { display: none !important; }
section[data-testid="stSidebar"] > div:first-child { padding-top: 0; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: #07071a !important;
  border-right: 1px solid rgba(124,58,237,0.12) !important;
}
[data-testid="stSidebar"] * { color: #b0b0d8 !important; }

/* ── Sidebar logo area ── */
.sb-logo {
  background: linear-gradient(135deg, rgba(124,58,237,0.15), rgba(79,70,229,0.08));
  border-bottom: 1px solid rgba(124,58,237,0.12);
  padding: 24px 20px 20px;
  text-align: center;
  margin-bottom: 0;
}
.sb-logo-icon { font-size: 2.2rem; display: block; margin-bottom: 10px; }
.sb-logo-title {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 0.95rem; font-weight: 700;
  color: #a78bfa !important;
  letter-spacing: 0.3px;
}
.sb-logo-sub { font-size: 0.68rem; color: #3d3d70 !important; margin-top: 3px; }

/* ── Stats grid ── */
.sb-stats {
  display: grid; grid-template-columns: 1fr 1fr 1fr;
  gap: 6px; padding: 14px;
}
.sb-stat {
  background: rgba(124,58,237,0.07);
  border: 1px solid rgba(124,58,237,0.12);
  border-radius: 8px; padding: 10px 6px; text-align: center;
}
.sb-stat-n {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 1.4rem; font-weight: 700; color: #ede9fe !important;
}
.sb-stat-l { font-size: 0.62rem; color: #44446a !important;
             text-transform: uppercase; letter-spacing: 0.8px; margin-top: 1px; }

/* ── Section label ── */
.sb-label {
  font-size: 0.65rem; font-weight: 700; color: #44446a !important;
  text-transform: uppercase; letter-spacing: 1.2px;
  padding: 0 14px; margin: 12px 0 6px;
}

/* ── Topic pills ── */
.pill-wrap { padding: 0 12px 10px; }
.pill {
  display: inline-block;
  background: rgba(124,58,237,0.08);
  border: 1px solid rgba(124,58,237,0.14);
  border-radius: 5px; padding: 2px 8px; margin: 2px;
  font-size: 0.7rem; color: #7c6ab0 !important;
}
.pill.arxiv { border-color: rgba(236,72,153,0.2); color: #c084fc !important;
               background: rgba(236,72,153,0.06); }

/* ── Buttons ── */
.stButton > button {
  background: rgba(124,58,237,0.09) !important;
  border: 1px solid rgba(124,58,237,0.22) !important;
  color: #a78bfa !important;
  border-radius: 8px !important;
  font-size: 0.82rem !important;
  font-weight: 500 !important;
  padding: 6px 12px !important;
  transition: all 0.15s ease !important;
  width: 100%;
}
.stButton > button:hover {
  background: rgba(124,58,237,0.2) !important;
  border-color: rgba(124,58,237,0.45) !important;
  color: #ede9fe !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 4px 16px rgba(124,58,237,0.2) !important;
}

/* ── Slider ── */
[data-testid="stSlider"] { padding: 0 14px; }
.stSlider [data-testid="stSliderThumb"] { background: #7c3aed !important; }
[data-baseweb="slider"] [data-testid="stSliderTrackFill"] { background: #7c3aed !important; }

/* ── Checkbox ── */
.stCheckbox { padding: 0 14px; }
.stCheckbox label p { font-size: 0.82rem !important; color: #7070a0 !important; }

/* ── Hero header ── */
.hero {
  position: relative;
  border-radius: 18px;
  overflow: hidden;
  margin: 24px 0 28px;
  padding: 40px 44px;
  background: linear-gradient(135deg, #0d0820 0%, #100a28 35%, #07071a 70%, #04040d 100%);
  border: 1px solid rgba(124,58,237,0.2);
  box-shadow:
    0 0 0 1px rgba(124,58,237,0.06),
    0 20px 60px rgba(124,58,237,0.08),
    inset 0 1px 0 rgba(255,255,255,0.04);
}
.hero::before {
  content: '';
  position: absolute;
  top: -40%; left: 30%;
  width: 400px; height: 400px;
  background: radial-gradient(circle, rgba(124,58,237,0.12) 0%, transparent 65%);
  pointer-events: none;
}
.hero::after {
  content: '';
  position: absolute;
  bottom: -30%; right: 15%;
  width: 300px; height: 300px;
  background: radial-gradient(circle, rgba(79,70,229,0.08) 0%, transparent 65%);
  pointer-events: none;
}
.hero-badge {
  display: inline-flex; align-items: center; gap: 6px;
  background: rgba(124,58,237,0.12);
  border: 1px solid rgba(124,58,237,0.25);
  border-radius: 20px;
  padding: 4px 12px;
  font-size: 0.72rem; font-weight: 600; color: #a78bfa;
  text-transform: uppercase; letter-spacing: 1px;
  margin-bottom: 16px;
}
.hero-title {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 2.4rem; font-weight: 700;
  color: #f5f3ff;
  line-height: 1.15;
  margin-bottom: 10px;
}
.hero-title span { color: #a78bfa; }
.hero-sub { font-size: 0.92rem; color: #5a5a90; max-width: 520px; line-height: 1.65; }
.hero-pills { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 20px; }
.hero-pill {
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 6px; padding: 4px 10px;
  font-size: 0.73rem; color: #6060a0;
}

/* ── Empty state cards ── */
.eq-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; margin-bottom: 28px; }
.eq-card {
  background: rgba(124,58,237,0.05);
  border: 1px solid rgba(124,58,237,0.14);
  border-radius: 12px;
  padding: 18px 16px;
  transition: all 0.18s ease;
  cursor: pointer;
}
.eq-card:hover {
  background: rgba(124,58,237,0.1);
  border-color: rgba(124,58,237,0.32);
  transform: translateY(-3px);
  box-shadow: 0 10px 30px rgba(124,58,237,0.12);
}
.eq-icon { font-size: 1.5rem; margin-bottom: 10px; }
.eq-cat {
  font-size: 0.67rem; font-weight: 700; color: #7c3aed;
  text-transform: uppercase; letter-spacing: 1px; margin-bottom: 5px;
}
.eq-text { font-size: 0.86rem; color: #9090c0; line-height: 1.5; }

/* ── Chat messages ── */
[data-testid="stChatMessage"] { background: transparent !important; padding: 2px 0; }
[data-testid="stChatMessageContent"] {
  background: rgba(124,58,237,0.06) !important;
  border: 1px solid rgba(124,58,237,0.12) !important;
  border-radius: 14px !important;
  padding: 18px 22px !important;
  font-size: 0.93rem !important;
  line-height: 1.78 !important;
}
[data-testid="stChatMessageContent"] p { color: #ddddf5 !important; margin: 0 0 10px; }
[data-testid="stChatMessageContent"] p:last-child { margin-bottom: 0; }
[data-testid="stChatMessageContent"] ul,
[data-testid="stChatMessageContent"] ol { color: #c8c8e8 !important; padding-left: 20px; }
[data-testid="stChatMessageContent"] li { margin: 4px 0; }
[data-testid="stChatMessageContent"] strong { color: #ede9fe !important; }
[data-testid="stChatMessageContent"] code {
  background: rgba(124,58,237,0.18) !important;
  color: #c4b5fd !important;
  border-radius: 4px; padding: 1px 6px; font-size: 0.87em;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
  background: rgba(79,70,229,0.08) !important;
  border-color: rgba(79,70,229,0.18) !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] textarea {
  background: rgba(124,58,237,0.06) !important;
  border: 1px solid rgba(124,58,237,0.22) !important;
  border-radius: 12px !important;
  color: #ddddf5 !important;
  font-size: 0.93rem !important;
}
[data-testid="stChatInput"] textarea:focus {
  border-color: rgba(124,58,237,0.5) !important;
  box-shadow: 0 0 0 3px rgba(124,58,237,0.08) !important;
}
[data-testid="stChatInput"] button {
  background: #7c3aed !important;
  border-radius: 8px !important;
  border: none !important;
}

/* ── Source cards ── */
.src-wrap { display: flex; flex-direction: column; gap: 8px; margin: 4px 0; }
.src-card {
  background: rgba(8,8,24,0.85);
  border: 1px solid rgba(124,58,237,0.14);
  border-left: 3px solid #7c3aed;
  border-radius: 10px;
  padding: 12px 16px;
  backdrop-filter: blur(8px);
  transition: border-color 0.15s;
}
.src-card.arxiv { border-left-color: #c084fc; }
.src-card:hover { border-color: rgba(124,58,237,0.35); }
.src-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }
.src-topic {
  font-family: 'Space Grotesk', sans-serif;
  font-weight: 600; font-size: 0.8rem; color: #a78bfa;
}
.src-badge {
  font-size: 0.63rem; font-weight: 600; padding: 2px 7px;
  border-radius: 4px; text-transform: uppercase; letter-spacing: 0.5px;
}
.src-badge.wiki { background: rgba(59,130,246,0.12); color: #60a5fa; border: 1px solid rgba(59,130,246,0.2); }
.src-badge.arxiv { background: rgba(192,132,252,0.1); color: #c084fc; border: 1px solid rgba(192,132,252,0.2); }
.src-bar-bg {
  height: 3px; background: rgba(255,255,255,0.05);
  border-radius: 2px; margin: 6px 0;
}
.src-bar { height: 100%; border-radius: 2px; transition: width 0.3s ease; }
.src-pct { font-size: 0.7rem; color: #44446a; margin-bottom: 6px; }
.src-snippet { font-size: 0.78rem; color: #6060a0; line-height: 1.55; }

/* ── Expander ── */
[data-testid="stExpander"] {
  background: rgba(8,8,24,0.5) !important;
  border: 1px solid rgba(124,58,237,0.12) !important;
  border-radius: 10px !important;
  margin: 8px 0 !important;
}
[data-testid="stExpander"] summary {
  color: #7060a0 !important;
  font-size: 0.82rem !important;
}

/* ── Follow-up chips ── */
.fup-section { margin-top: 14px; }
.fup-label {
  font-size: 0.67rem; font-weight: 600; color: #3d3d6a;
  text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(124,58,237,0.25); border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

# ── Data ──────────────────────────────────────────────────────────────────────

EXAMPLES = [
    {"cat": "Attribution",   "q": "How does multi-touch attribution work and which model is most accurate?"},
    {"cat": "Churn",         "q": "What ML models best predict customer churn and what features matter most?"},
    {"cat": "Influencer ROI","q": "How do you measure real influencer marketing ROI beyond vanity metrics?"},
    {"cat": "A/B Testing",   "q": "How do you design an A/B test that's actually statistically valid?"},
    {"cat": "Segmentation",  "q": "What clustering approaches work best for customer segmentation?"},
    {"cat": "LLMs",          "q": "How are LLMs being used in marketing analytics today?"},
]

WIKI_PILLS  = ["A/B Testing","Attribution","Churn","CLV","CRO","MMM","Influencer",
               "Email","NPS","Cohort","Digital Mktg","Funnel","ROMI","CAC","Predictive","Sentiment"]
ARXIV_PILLS = ["ML Churn","Deep Attribution","NLP Reviews","RL Optimization",
               "Transformers","GNN Reco","Causal Inference","LLM Analytics","CLV Neural","Multi-touch ML"]

# ── State ─────────────────────────────────────────────────────────────────────

for k, v in [("messages",[]),("meta",{}),("followups",{}),("pending",None)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div class='sb-logo'>
      <span class='sb-logo-icon'>◆</span>
      <div class='sb-logo-title'>Marketing Analytics AI</div>
      <div class='sb-logo-sub'>RAG Agent · Grok-3 · by Aneela Veldi</div>
    </div>
    """, unsafe_allow_html=True)

    ready    = index_exists()
    n_chunks = index_size() if ready else 0
    n_asked  = len([m for m in st.session_state.messages if m["role"] == "user"])

    st.markdown(f"""
    <div class='sb-stats'>
      <div class='sb-stat'><div class='sb-stat-n'>{n_chunks}</div><div class='sb-stat-l'>Chunks</div></div>
      <div class='sb-stat'><div class='sb-stat-n'>45</div><div class='sb-stat-l'>Sources</div></div>
      <div class='sb-stat'><div class='sb-stat-n'>{n_asked}</div><div class='sb-stat-l'>Asked</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='sb-label'>Knowledge Base</div>", unsafe_allow_html=True)
    wiki_html  = "".join(f"<span class='pill'>{p}</span>" for p in WIKI_PILLS)
    arxiv_html = "".join(f"<span class='pill arxiv'>{p}</span>" for p in ARXIV_PILLS)
    st.markdown(f"<div class='pill-wrap'>{wiki_html}</div>", unsafe_allow_html=True)
    st.markdown("<div class='sb-label'>arXiv Papers</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='pill-wrap'>{arxiv_html}</div>", unsafe_allow_html=True)

    st.markdown("<div class='sb-label' style='margin-top:14px'>Settings</div>", unsafe_allow_html=True)
    top_k        = st.slider("Sources", 3, 8, 6)
    show_sources = st.checkbox("Show retrieved sources", value=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    if st.button("Clear conversation", use_container_width=True):
        for k in ["messages","meta","followups","pending"]:
            st.session_state[k] = {} if k in ("meta","followups") else ([] if k=="messages" else None)
        st.rerun()

# ── Auto-build index if missing ───────────────────────────────────────────────

if not ready:
    with st.status("Building knowledge base for the first time. This takes a few minutes...", expanded=True) as status:
        st.write("Fetching Wikipedia articles and arXiv research papers...")
        chunks = load_corpus(use_cache=False)
        st.write(f"Embedding {len(chunks)} chunks...")
        embeddings = embed_texts([c["text"] for c in chunks])
        st.write("Building FAISS index...")
        build_index(chunks, embeddings)
        status.update(label="Knowledge base ready.", state="complete")
    st.rerun()

if not os.getenv("XAI_API_KEY"):
    st.error("Add XAI_API_KEY to Streamlit Cloud secrets (Settings > Secrets).")
    st.stop()

# ── Hero ──────────────────────────────────────────────────────────────────────

st.markdown("""
<div class='hero'>
  <div class='hero-badge'>◆ AI Agent &nbsp;·&nbsp; RAG &nbsp;·&nbsp; Grok-3</div>
  <div class='hero-title'>Marketing Analytics<br><span>Intelligence Agent</span></div>
  <div class='hero-sub'>
    Ask anything grounded in real research. Wikipedia knowledge base + arXiv papers
    on ML for marketing, attribution, churn, segmentation, and NLP.
  </div>
  <div class='hero-pills'>
    <span class='hero-pill'>Wikipedia KB</span>
    <span class='hero-pill'>arXiv Research Papers</span>
    <span class='hero-pill'>Grok-3</span>
    <span class='hero-pill'>FAISS Semantic Search</span>
    <span class='hero-pill'>sentence-transformers</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────

def bar_color(score: float) -> str:
    if score > 0.68: return "linear-gradient(90deg,#22c55e,#16a34a)"
    if score > 0.50: return "linear-gradient(90deg,#f59e0b,#d97706)"
    return "linear-gradient(90deg,#6366f1,#4f46e5)"

def render_sources(chunks: list[dict], scores: list[float]) -> None:
    st.markdown("<div class='src-wrap'>", unsafe_allow_html=True)
    for chunk, score in zip(chunks, scores):
        pct    = int(score * 100)
        stype  = chunk.get("source_type", "Wikipedia")
        is_ax  = stype == "arXiv Research Paper"
        cls    = "src-card arxiv" if is_ax else "src-card"
        badge  = f"<span class='src-badge arxiv'>arXiv</span>" if is_ax else "<span class='src-badge wiki'>Wikipedia</span>"
        snip   = chunk["text"][:180].replace("\n"," ").strip() + "..."
        st.markdown(f"""
        <div class='{cls}'>
          <div class='src-head'>
            <div class='src-topic'>{chunk['source'][:55]}</div>
            {badge}
          </div>
          <div class='src-bar-bg'><div class='src-bar' style='width:{pct}%;background:{bar_color(score)}'></div></div>
          <div class='src-pct'>{pct}% relevance match</div>
          <div class='src-snippet'>{snip}</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

def render_followups(followups: list[str], prefix: str) -> None:
    if not followups: return
    st.markdown("<div class='fup-label'>You might also ask</div>", unsafe_allow_html=True)
    cols = st.columns(len(followups))
    for i, (col, fup) in enumerate(zip(cols, followups)):
        with col:
            if st.button(fup, key=f"{prefix}_{i}", use_container_width=True):
                st.session_state.pending = fup
                st.rerun()

# ── Empty state ───────────────────────────────────────────────────────────────

if not st.session_state.messages:
    cols = st.columns(3)
    for i, ex in enumerate(EXAMPLES):
        with cols[i % 3]:
            if st.button(
                f"**{ex['cat']}**\n\n{ex['q']}",
                key=f"eg_{i}",
                use_container_width=True,
            ):
                st.session_state.pending = ex["q"]
                st.rerun()
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

# ── Chat history ──────────────────────────────────────────────────────────────

for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            meta = st.session_state.meta.get(i, {})
            if show_sources and meta.get("chunks"):
                with st.expander(f"◈ {len(meta['chunks'])} sources retrieved", expanded=False):
                    render_sources(meta["chunks"], meta["scores"])
            if meta.get("followups"):
                render_followups(meta["followups"], f"h{i}")

# ── Input ─────────────────────────────────────────────────────────────────────

prompt = st.chat_input("Ask about marketing analytics, attribution, churn, segmentation...") \
         or st.session_state.pop("pending", None)

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        chunks, scores = retrieve(prompt, k=top_k)

        if show_sources:
            with st.expander(f"◈ {len(chunks)} sources retrieved", expanded=True):
                render_sources(chunks, scores)

        history = [{"role": m["role"], "content": m["content"]}
                   for m in st.session_state.messages[:-1]]

        full_answer = st.write_stream(ask_stream(prompt, history, chunks))

    followups = get_followups(prompt, full_answer)
    render_followups(followups, f"n{len(st.session_state.messages)}")

    idx = len(st.session_state.messages)
    st.session_state.meta[idx] = {"chunks": chunks, "scores": scores, "followups": followups}
    st.session_state.messages.append({"role": "assistant", "content": full_answer})
