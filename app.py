import os
import time
import tempfile
import threading
import streamlit as st

from main import get_collection
from src.embed import embed_query
from src.keyword_search import build_bm25_index
from src.stage3_rerank import rerank
from src.generate import generate_answer
from src.fusion import hybrid_retrieve_chroma, expand_chunks
from src.parse import check_file_limits
from src.llm import stream_llm, rewrite_query
from src.stage5_trust import check_grounding

st.set_page_config(page_title="Sourcerer", page_icon="📄", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;0,900;1,700&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 70% 55% at 25% 100%, #8B1A00 0%, transparent 60%),
        radial-gradient(ellipse 45% 40% at 0% 80%, #C8340A 0%, transparent 45%),
        #080400 !important;
    min-height: 100vh;
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="block-container"] {
    padding-top: 60px !important;
    padding-left: 5vw !important;
    padding-right: 5vw !important;
    max-width: 1600px !important;
}

/* Geometric art */
.geo-art {
    position: fixed;
    right: -60px;
    top: 0;
    width: 50vw;
    height: 100vh;
    z-index: 0;
    pointer-events: none;
    opacity: 0.7;
}

/* Hero */
.eyebrow {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: rgba(240,230,204,0.35);
    margin-bottom: 16px;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2.8rem, 5.5vw, 5.5rem);
    font-weight: 900;
    line-height: 0.95;
    letter-spacing: -0.02em;
    color: #F0E6CC;
    margin-bottom: 20px;
}
.hero-title em {
    font-style: italic;
    color: #C8340A;
}
.divider-line {
    width: 50px;
    height: 1px;
    background: rgba(240,230,204,0.2);
    margin-bottom: 20px;
}
.hero-tagline {
    font-family: 'Playfair Display', serif;
    font-style: italic;
    font-size: clamp(0.85rem, 1.5vw, 1.05rem);
    color: rgba(240,230,204,0.5);
    margin-bottom: 10px;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.hero-desc {
    font-size: 13px;
    font-weight: 300;
    color: rgba(240,230,204,0.32);
    line-height: 1.7;
    max-width: 380px;
    margin-bottom: 0;
}

/* Nav buttons — pill style */
[data-testid="stButton"] button {
    background: transparent !important;
    border: 1px solid rgba(240,230,204,0.22) !important;
    border-radius: 999px !important;
    color: #F0E6CC !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    padding: 5px 14px !important;
    letter-spacing: 0.04em !important;
    white-space: nowrap !important;
    transition: all 0.2s ease !important;
}
[data-testid="stButton"] button:hover {
    border-color: rgba(240,230,204,0.55) !important;
    background: rgba(240,230,204,0.05) !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    border: 1px solid rgba(240,230,204,0.12) !important;
    border-radius: 10px !important;
    background: rgba(8,4,0,0.75) !important;
    padding: 12px !important;
    max-width: 460px !important;
    margin-top: 8px !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(200,52,10,0.4) !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] {
    color: rgba(240,230,204,0.35) !important;
}

/* Text input — solid background */
[data-testid="stTextInput"] input {
    border: 1px solid rgba(240,230,204,0.15) !important;
    border-radius: 8px !important;
    background: rgba(8,4,0,0.85) !important;
    color: #F0E6CC !important;
    font-size: 0.9rem !important;
    padding: 0.7rem 1rem !important;
    max-width: 460px !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: rgba(200,52,10,0.5) !important;
    box-shadow: 0 0 0 2px rgba(200,52,10,0.1) !important;
}
[data-testid="stTextInput"] input::placeholder {
    color: rgba(240,230,204,0.2) !important;
}

/* Expander */
[data-testid="stExpander"] {
    border: 1px solid rgba(240,230,204,0.1) !important;
    border-radius: 10px !important;
    background: rgba(8,4,0,0.8) !important;
}

/* Alerts */
[data-testid="stAlert"] {
    border-radius: 8px !important;
    border: none !important;
    max-width: 460px !important;
    background: rgba(8,4,0,0.8) !important;
}

/* Answer text */
[data-testid="stMarkdownContainer"] p {
    color: #F0E6CC;
    line-height: 1.75;
}

/* Confidence badges */
.badge-high {
    display: inline-block;
    background: rgba(0,180,80,0.07);
    color: #4ADE80;
    border: 1px solid rgba(0,180,80,0.18);
    border-radius: 999px;
    padding: 0.2rem 0.75rem;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    margin-top: 0.75rem;
    text-transform: uppercase;
}
.badge-low {
    display: inline-block;
    background: rgba(200,52,10,0.07);
    color: #FB923C;
    border: 1px solid rgba(200,52,10,0.2);
    border-radius: 999px;
    padding: 0.2rem 0.75rem;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    margin-top: 0.75rem;
    text-transform: uppercase;
}

/* Info panel */
.info-section { margin-bottom: 1.1rem; }
.info-section h3 {
    color: rgba(200,52,10,0.9);
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.info-section p, .info-section li {
    color: rgba(240,230,204,0.55);
    font-size: 0.82rem;
    line-height: 1.6;
}
.info-section ul { padding-left: 1rem; margin: 0; }
.tech-pill {
    display: inline-block;
    background: rgba(240,230,204,0.04);
    border: 1px solid rgba(240,230,204,0.12);
    border-radius: 999px;
    padding: 2px 9px;
    font-size: 0.68rem;
    color: rgba(240,230,204,0.4);
    margin: 2px;
}

/* Pipeline animation */
@keyframes flow {
    0%   { stroke-dashoffset: 20; opacity: 0.2; }
    50%  { opacity: 0.7; }
    100% { stroke-dashoffset: 0; opacity: 0.2; }
}
.pipe  { stroke-dasharray: 5 5; animation: flow 1.6s linear infinite 0.0s; }
.pipe2 { stroke-dasharray: 5 5; animation: flow 1.6s linear infinite 0.3s; }
.pipe3 { stroke-dasharray: 5 5; animation: flow 1.6s linear infinite 0.6s; }
.pipe4 { stroke-dasharray: 5 5; animation: flow 1.6s linear infinite 0.9s; }
.pipe5 { stroke-dasharray: 5 5; animation: flow 1.6s linear infinite 1.2s; }
.pipe6 { stroke-dasharray: 5 5; animation: flow 1.6s linear infinite 1.5s; }

hr { border-color: rgba(240,230,204,0.08) !important; margin: 1.2rem 0 !important; }
</style>

<!-- Geometric art -->
<svg class="geo-art" viewBox="0 0 700 900" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
  <polygon points="280,80 480,80 480,480 280,480" fill="none" stroke="#F0E6CC" stroke-width="1.2" opacity="0.4"/>
  <polygon points="380,120 580,120 580,520 380,520" fill="none" stroke="#F0E6CC" stroke-width="1.2" opacity="0.25"/>
  <line x1="480" y1="480" x2="700" y2="500" stroke="#F0E6CC" stroke-width="0.6" opacity="0.25"/>
  <line x1="480" y1="480" x2="700" y2="524" stroke="#F0E6CC" stroke-width="0.6" opacity="0.23"/>
  <line x1="480" y1="480" x2="700" y2="548" stroke="#F0E6CC" stroke-width="0.6" opacity="0.21"/>
  <line x1="480" y1="480" x2="700" y2="572" stroke="#F0E6CC" stroke-width="0.6" opacity="0.19"/>
  <line x1="480" y1="480" x2="700" y2="596" stroke="#F0E6CC" stroke-width="0.6" opacity="0.17"/>
  <line x1="480" y1="480" x2="700" y2="620" stroke="#F0E6CC" stroke-width="0.6" opacity="0.15"/>
  <line x1="480" y1="480" x2="700" y2="644" stroke="#F0E6CC" stroke-width="0.6" opacity="0.13"/>
  <line x1="480" y1="480" x2="700" y2="668" stroke="#F0E6CC" stroke-width="0.6" opacity="0.11"/>
  <line x1="480" y1="480" x2="700" y2="692" stroke="#F0E6CC" stroke-width="0.6" opacity="0.09"/>
  <line x1="480" y1="480" x2="680" y2="900" stroke="#F0E6CC" stroke-width="0.6" opacity="0.07"/>
  <line x1="480" y1="480" x2="620" y2="900" stroke="#F0E6CC" stroke-width="0.6" opacity="0.06"/>
  <line x1="480" y1="480" x2="560" y2="900" stroke="#F0E6CC" stroke-width="0.6" opacity="0.05"/>
  <line x1="580" y1="520" x2="700" y2="540" stroke="#F0E6CC" stroke-width="0.5" opacity="0.16"/>
  <line x1="580" y1="520" x2="700" y2="570" stroke="#F0E6CC" stroke-width="0.5" opacity="0.13"/>
  <line x1="580" y1="520" x2="700" y2="600" stroke="#F0E6CC" stroke-width="0.5" opacity="0.10"/>
  <line x1="480" y1="80" x2="700" y2="80" stroke="#F0E6CC" stroke-width="0.8" opacity="0.35"/>
</svg>
""", unsafe_allow_html=True)

# ─── Session state defaults ─────────────────────────────────────────────────
for key, default in {
    "thread": None,
    "collection": None,
    "ingest_error": None,
    "last_filename": None,
    "result_container": None,
    "chat_history": [],
    "show_info": False,
    "show_pipeline": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─── Layout: left content + right panel ─────────────────────────────────────
left, right = st.columns([5, 2])

with left:
    # Hero
    st.markdown("""
<p class="eyebrow">Document intelligence</p>
<div class="hero-title">Hi, I'm<br><em>Sourcerer</em></div>
<div class="divider-line"></div>
<p class="hero-tagline">Your PDF, actually answered.</p>
<p class="hero-desc">Upload any PDF. Ask anything. Sourcerer finds the right passage, verifies the answer is grounded in the source, and streams it back with a confidence badge.</p>
""", unsafe_allow_html=True)

    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

    # ── File upload ──────────────────────────────────────────────────────────
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

    if uploaded_file is not None:

        if uploaded_file.name != st.session_state.last_filename:
            st.session_state.collection = None
            st.session_state.ingest_error = None
            st.session_state.thread = None
            st.session_state.result_container = None
            st.session_state.last_filename = uploaded_file.name
            st.session_state.chat_history = []

        collection_ready = st.session_state.collection is not None
        thread_running = (
            st.session_state.thread is not None
            and st.session_state.thread.is_alive()
        )
        thread_finished = (
            st.session_state.thread is not None
            and not st.session_state.thread.is_alive()
            and not collection_ready
            and st.session_state.ingest_error is None
        )

        if not collection_ready and not thread_running and st.session_state.thread is None:

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getbuffer())
                temp_path = tmp.name

            ok, error_msg = check_file_limits(temp_path)
            if not ok:
                st.error(f"❌ {error_msg}")
                os.unlink(temp_path)
                st.stop()

            result_container = {"collection": None, "error": None}
            st.session_state.result_container = result_container

            def ingest_worker(path, container):
                try:
                    container["collection"] = get_collection(path)
                except Exception as e:
                    container["error"] = str(e)
                finally:
                    os.unlink(path)

            t = threading.Thread(
                target=ingest_worker,
                args=(temp_path, result_container),
                daemon=True
            )
            t.start()
            st.session_state.thread = t
            st.rerun()

        elif thread_running:
            with st.spinner("Processing document..."):
                time.sleep(1)
            st.rerun()

        elif thread_finished:
            container = st.session_state.result_container
            if container["error"]:
                st.session_state.ingest_error = container["error"]
            else:
                st.session_state.collection = container["collection"]
            st.rerun()

        elif st.session_state.ingest_error is not None:
            st.error(f"❌ Failed to process PDF: {st.session_state.ingest_error}")
            st.stop()

        elif collection_ready:
            collection = st.session_state.collection

            all_data = collection.get(include=["documents"])
            chunks = all_data["documents"]
            bm25_index = build_bm25_index(chunks)

            st.success(f"✅ {len(chunks)} chunks loaded from {uploaded_file.name}")

            question = st.text_input("Ask a question about the document")

            if question:

                rewritten = rewrite_query(question, st.session_state.chat_history)
                if rewritten != question:
                    st.caption(f"🔍 Interpreted as: *{rewritten}*")

                with st.spinner("Searching..."):
                    candidates = hybrid_retrieve_chroma(
                        rewritten, embed_query(rewritten), collection, chunks,
                        bm25_index, top_k=12, candidate_k=12
                    )
                    top_results = rerank(rewritten, candidates, top_k=3)
                    expanded = expand_chunks(top_results, chunks, neighbors=1)
                    context_chunks = [chunk for _, chunk in expanded]
                    top_index = top_results[0][0]
                    score = top_results[0][2]

                st.markdown("### Answer")

                if isinstance(context_chunks, list):
                    context = "\n\n---\n\n".join(context_chunks)
                else:
                    context = context_chunks

                prompt = (
                    "Answer the question using only the following context. "
                    "If the answer isn't in the context, say so clearly instead of guessing.\n\n"
                    f"Context:\n{context}\n\n"
                    f"Question: {rewritten}"
                )

                streamed_text = st.write_stream(stream_llm(prompt))

                verdict = check_grounding(rewritten, context_chunks, streamed_text)

                if verdict == "SUPPORTED":
                    st.markdown('<span class="badge-high">✦ HIGH CONFIDENCE</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="badge-low">◈ LOW CONFIDENCE</span>', unsafe_allow_html=True)

                st.session_state.chat_history.append((question, streamed_text))
                st.session_state.chat_history = st.session_state.chat_history[-4:]

                with st.expander(f"Source chunks — top #{top_index}, score {score:.4f}"):
                    st.write(f"**{len(context_chunks)} chunks used as context:**")
                    for chunk_idx, chunk_text in expanded:
                        st.write(f"**Chunk #{chunk_idx}:**")
                        st.write(chunk_text)
                        st.divider()

    else:
        st.markdown('<p style="color:rgba(240,230,204,0.25);font-size:0.85rem;margin-top:0.25rem;">Upload a PDF to get started.</p>', unsafe_allow_html=True)

# ─── Right panel: buttons + panels ──────────────────────────────────────────
with right:
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    
    b1, b2 = st.columns(2)
    with b1:
        if st.button("⚡ Pipeline"):
            st.session_state.show_pipeline = not st.session_state.show_pipeline
            st.session_state.show_info = False
    with b2:
        if st.button("ℹ Info"):
            st.session_state.show_info = not st.session_state.show_info
            st.session_state.show_pipeline = False

    # ── Info panel ───────────────────────────────────────────────────────────
    if st.session_state.show_info:
        st.markdown("""
<div style="margin-top:16px;padding:16px;background:rgba(8,4,0,0.85);border:1px solid rgba(240,230,204,0.1);border-radius:10px;">
<div class="info-section">
  <h3>What it does</h3>
  <p>Upload any PDF and ask questions. Sourcerer finds the most relevant passages, verifies grounding, and streams the answer with a confidence badge.</p>
</div>
<div class="info-section">
  <h3>12 build stages</h3>
  <ul>
    <li><b style="color:rgba(240,230,204,0.8)">S0</b> — Basic RAG pipeline</li>
    <li><b style="color:rgba(240,230,204,0.8)">S1</b> — Baseline report card</li>
    <li><b style="color:rgba(240,230,204,0.8)">S2</b> — BM25 + RRF hybrid search</li>
    <li><b style="color:rgba(240,230,204,0.8)">S3</b> — Cross-encoder reranking</li>
    <li><b style="color:rgba(240,230,204,0.8)">S4</b> — HyDE query expansion</li>
    <li><b style="color:rgba(240,230,204,0.8)">S5</b> — LLM trust check</li>
    <li><b style="color:rgba(240,230,204,0.8)">S6</b> — Chroma vector DB</li>
    <li><b style="color:rgba(240,230,204,0.8)">S7</b> — Messy PDF handling</li>
    <li><b style="color:rgba(240,230,204,0.8)">S8</b> — Background threading</li>
    <li><b style="color:rgba(240,230,204,0.8)">S9</b> — Streaming answers</li>
    <li><b style="color:rgba(240,230,204,0.8)">S9.5</b> — Conversation memory</li>
    <li><b style="color:rgba(240,230,204,0.8)">S10</b> — Deployed</li>
  </ul>
</div>
<div class="info-section">
  <h3>Stack</h3>
  <span class="tech-pill">Python</span>
  <span class="tech-pill">Streamlit</span>
  <span class="tech-pill">Chroma</span>
  <span class="tech-pill">BGE reranker</span>
  <span class="tech-pill">BM25+RRF</span>
  <span class="tech-pill">Gemini</span>
</div>
</div>
""", unsafe_allow_html=True)

    # ── Pipeline panel — monochrome vertical ─────────────────────────────────
    if st.session_state.show_pipeline:
        st.markdown("""
<div style="margin-top:16px;padding:16px;background:rgba(8,4,0,0.85);border:1px solid rgba(240,230,204,0.1);border-radius:10px;">
<p style="font-size:10px;font-weight:600;letter-spacing:0.12em;color:rgba(240,230,204,0.3);text-transform:uppercase;margin-bottom:12px">Pipeline</p>
<svg width="100%" viewBox="0 0 200 640" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="va" viewBox="0 0 10 10" refX="5" refY="8" markerWidth="5" markerHeight="5" orient="auto">
      <path d="M2 2L5 8L8 2" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
  </defs>

  <!-- INGEST label -->
  <text x="100" y="18" text-anchor="middle" style="fill:rgba(240,230,204,0.25);font-size:8px;font-family:Inter,sans-serif;font-weight:600;letter-spacing:0.1em">INGEST</text>

  <!-- PDF -->
  <rect x="40" y="26" width="120" height="36" rx="6" fill="rgba(200,52,10,0.1)" stroke="rgba(200,52,10,0.4)" stroke-width="0.8"/>
  <text x="100" y="40" text-anchor="middle" style="fill:rgba(240,230,204,0.85);font-size:11px;font-family:Inter,sans-serif;font-weight:600">PDF upload</text>
  <text x="100" y="54" text-anchor="middle" style="fill:rgba(240,230,204,0.3);font-size:9px;font-family:Inter,sans-serif">parse · chunk · embed</text>

  <line class="pipe" x1="100" y1="63" x2="100" y2="83" stroke="rgba(240,230,204,0.25)" stroke-width="1" fill="none" marker-end="url(#va)"/>

  <!-- Chroma -->
  <rect x="40" y="86" width="120" height="36" rx="6" fill="rgba(240,230,204,0.04)" stroke="rgba(240,230,204,0.18)" stroke-width="0.8"/>
  <text x="100" y="100" text-anchor="middle" style="fill:rgba(240,230,204,0.85);font-size:11px;font-family:Inter,sans-serif;font-weight:600">Chroma</text>
  <text x="100" y="114" text-anchor="middle" style="fill:rgba(240,230,204,0.3);font-size:9px;font-family:Inter,sans-serif">vector store</text>

  <!-- Divider -->
  <line x1="20" y1="142" x2="180" y2="142" stroke="rgba(240,230,204,0.08)" stroke-width="0.5" stroke-dasharray="3 3"/>
  <text x="100" y="156" text-anchor="middle" style="fill:rgba(240,230,204,0.2);font-size:8px;font-family:Inter,sans-serif;font-weight:600;letter-spacing:0.1em">QUERY</text>

  <!-- Question -->
  <rect x="40" y="164" width="120" height="36" rx="6" fill="rgba(240,230,204,0.04)" stroke="rgba(240,230,204,0.18)" stroke-width="0.8"/>
  <text x="100" y="178" text-anchor="middle" style="fill:rgba(240,230,204,0.85);font-size:11px;font-family:Inter,sans-serif;font-weight:600">Question</text>
  <text x="100" y="192" text-anchor="middle" style="fill:rgba(240,230,204,0.3);font-size:9px;font-family:Inter,sans-serif">user input</text>

  <line class="pipe2" x1="100" y1="201" x2="100" y2="221" stroke="rgba(240,230,204,0.25)" stroke-width="1" fill="none" marker-end="url(#va)"/>

  <!-- Rewrite -->
  <rect x="40" y="224" width="120" height="36" rx="6" fill="rgba(240,230,204,0.04)" stroke="rgba(240,230,204,0.18)" stroke-width="0.8"/>
  <text x="100" y="238" text-anchor="middle" style="fill:rgba(240,230,204,0.85);font-size:11px;font-family:Inter,sans-serif;font-weight:600">Rewrite</text>
  <text x="100" y="252" text-anchor="middle" style="fill:rgba(240,230,204,0.3);font-size:9px;font-family:Inter,sans-serif">query expansion</text>

  <line class="pipe3" x1="100" y1="261" x2="100" y2="281" stroke="rgba(240,230,204,0.25)" stroke-width="1" fill="none" marker-end="url(#va)"/>

  <!-- Retrieve -->
  <rect x="40" y="284" width="120" height="36" rx="6" fill="rgba(240,230,204,0.04)" stroke="rgba(240,230,204,0.18)" stroke-width="0.8"/>
  <text x="100" y="298" text-anchor="middle" style="fill:rgba(240,230,204,0.85);font-size:11px;font-family:Inter,sans-serif;font-weight:600">Retrieve</text>
  <text x="100" y="312" text-anchor="middle" style="fill:rgba(240,230,204,0.3);font-size:9px;font-family:Inter,sans-serif">BM25 + vector + RRF</text>

  <line class="pipe4" x1="100" y1="321" x2="100" y2="341" stroke="rgba(240,230,204,0.25)" stroke-width="1" fill="none" marker-end="url(#va)"/>

  <!-- Rerank -->
  <rect x="40" y="344" width="120" height="36" rx="6" fill="rgba(240,230,204,0.04)" stroke="rgba(240,230,204,0.18)" stroke-width="0.8"/>
  <text x="100" y="358" text-anchor="middle" style="fill:rgba(240,230,204,0.85);font-size:11px;font-family:Inter,sans-serif;font-weight:600">Rerank</text>
  <text x="100" y="372" text-anchor="middle" style="fill:rgba(240,230,204,0.3);font-size:9px;font-family:Inter,sans-serif">cross-encoder</text>

  <line class="pipe5" x1="100" y1="381" x2="100" y2="401" stroke="rgba(240,230,204,0.25)" stroke-width="1" fill="none" marker-end="url(#va)"/>

  <!-- Generate -->
  <rect x="40" y="404" width="120" height="36" rx="6" fill="rgba(240,230,204,0.04)" stroke="rgba(240,230,204,0.18)" stroke-width="0.8"/>
  <text x="100" y="418" text-anchor="middle" style="fill:rgba(240,230,204,0.85);font-size:11px;font-family:Inter,sans-serif;font-weight:600">Gemini</text>
  <text x="100" y="432" text-anchor="middle" style="fill:rgba(240,230,204,0.3);font-size:9px;font-family:Inter,sans-serif">stream answer</text>

  <line class="pipe6" x1="100" y1="441" x2="100" y2="461" stroke="rgba(240,230,204,0.25)" stroke-width="1" fill="none" marker-end="url(#va)"/>

  <!-- Trust + Answer -->
  <rect x="40" y="464" width="120" height="36" rx="6" fill="rgba(200,52,10,0.08)" stroke="rgba(200,52,10,0.3)" stroke-width="0.8"/>
  <text x="100" y="478" text-anchor="middle" style="fill:rgba(240,230,204,0.85);font-size:11px;font-family:Inter,sans-serif;font-weight:600">Answer</text>
  <text x="100" y="492" text-anchor="middle" style="fill:rgba(240,230,204,0.3);font-size:9px;font-family:Inter,sans-serif">trust check · confidence</text>

</svg>
</div>
""", unsafe_allow_html=True)