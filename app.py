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
        radial-gradient(ellipse 70% 55% at 20% 100%, #8B1A00 0%, transparent 60%),
        radial-gradient(ellipse 45% 40% at 0% 80%, #C8340A 0%, transparent 45%),
        #080400 !important;
    min-height: 100vh;
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="block-container"] {
    padding-top: 48px !important;
    padding-left: 5vw !important;
    padding-right: 5vw !important;
    max-width: 1600px !important;
}

/* Remove default column gap bloat */
[data-testid="stHorizontalBlock"] {
    gap: 2rem !important;
    align-items: flex-start !important;
}

/* Geo art */
.geo-art {
    position: fixed;
    right: -60px;
    top: 0;
    width: 50vw;
    height: 100vh;
    z-index: 0;
    pointer-events: none;
    opacity: 0.65;
}

/* Hero */
.eyebrow {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: rgba(240,230,204,0.3);
    margin-bottom: 14px;
    margin-top: 0;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2.6rem, 5vw, 5rem);
    font-weight: 900;
    line-height: 0.95;
    letter-spacing: -0.02em;
    color: #F0E6CC;
    margin-bottom: 18px;
}
.hero-title em { font-style: italic; color: #C8340A; }
.divider-line {
    width: 44px; height: 1px;
    background: rgba(240,230,204,0.18);
    margin-bottom: 16px;
}
.hero-tagline {
    font-family: 'Playfair Display', serif;
    font-style: italic;
    font-size: 0.95rem;
    color: rgba(240,230,204,0.45);
    margin-bottom: 8px;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.hero-desc {
    font-size: 12.5px;
    font-weight: 300;
    color: rgba(240,230,204,0.28);
    line-height: 1.65;
    max-width: 360px;
    margin-bottom: 0;
}

/* Buttons */
[data-testid="stButton"] button {
    background: transparent !important;
    border: 1px solid rgba(240,230,204,0.2) !important;
    border-radius: 999px !important;
    color: #F0E6CC !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    padding: 5px 14px !important;
    letter-spacing: 0.04em !important;
    white-space: nowrap !important;
    transition: all 0.2s !important;
    width: 100% !important;
}
[data-testid="stButton"] button:hover {
    border-color: rgba(240,230,204,0.5) !important;
    background: rgba(240,230,204,0.04) !important;
}

/* Remove column padding on button row */
[data-testid="stHorizontalBlock"] > div:first-child,
[data-testid="stHorizontalBlock"] > div:last-child {
    padding: 0 !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    border: 1px solid rgba(240,230,204,0.1) !important;
    border-radius: 8px !important;
    background: rgba(8,4,0,0.8) !important;
    padding: 8px !important;
    max-width: 440px !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(200,52,10,0.35) !important;
}

/* Text input */
[data-testid="stTextInput"] input {
    border: 1px solid rgba(240,230,204,0.12) !important;
    border-radius: 7px !important;
    background: rgba(8,4,0,0.88) !important;
    color: #F0E6CC !important;
    font-size: 0.88rem !important;
    padding: 0.65rem 0.9rem !important;
    max-width: 440px !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: rgba(200,52,10,0.45) !important;
    box-shadow: 0 0 0 2px rgba(200,52,10,0.08) !important;
}
[data-testid="stTextInput"] input::placeholder {
    color: rgba(240,230,204,0.18) !important;
}

/* Expander */
[data-testid="stExpander"] {
    border: 1px solid rgba(240,230,204,0.08) !important;
    border-radius: 8px !important;
    background: rgba(8,4,0,0.82) !important;
}

/* Alerts */
[data-testid="stAlert"] {
    border-radius: 7px !important;
    border: none !important;
    max-width: 440px !important;
}

/* Answer text */
[data-testid="stMarkdownContainer"] p { color: #F0E6CC; line-height: 1.75; }

/* Badges */
.badge-high {
    display: inline-block;
    background: rgba(0,180,80,0.07);
    color: #4ADE80;
    border: 1px solid rgba(0,180,80,0.16);
    border-radius: 999px;
    padding: 0.18rem 0.7rem;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    margin-top: 0.6rem;
    text-transform: uppercase;
}
.badge-low {
    display: inline-block;
    background: rgba(200,52,10,0.07);
    color: #FB923C;
    border: 1px solid rgba(200,52,10,0.18);
    border-radius: 999px;
    padding: 0.18rem 0.7rem;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    margin-top: 0.6rem;
    text-transform: uppercase;
}

/* Right panel card */
.panel-card {
    margin-top: 12px;
    padding: 14px;
    background: rgba(8,4,0,0.88);
    border: 1px solid rgba(240,230,204,0.09);
    border-radius: 10px;
}
.panel-label {
    font-size: 9px;
    font-weight: 600;
    letter-spacing: 0.14em;
    color: rgba(240,230,204,0.22);
    text-transform: uppercase;
    margin-bottom: 10px;
}
.info-row {
    display: flex;
    gap: 6px;
    margin-bottom: 5px;
    align-items: baseline;
}
.info-stage {
    font-size: 9px;
    font-weight: 600;
    color: rgba(200,52,10,0.8);
    min-width: 28px;
    letter-spacing: 0.04em;
}
.info-text {
    font-size: 10.5px;
    color: rgba(240,230,204,0.5);
    line-height: 1.4;
}
.tech-pill {
    display: inline-block;
    background: rgba(240,230,204,0.03);
    border: 1px solid rgba(240,230,204,0.1);
    border-radius: 999px;
    padding: 2px 8px;
    font-size: 9px;
    color: rgba(240,230,204,0.35);
    margin: 2px;
}

/* Pipeline animation */
@keyframes flow {
    0%   { stroke-dashoffset: 18; opacity: 0.15; }
    50%  { opacity: 0.6; }
    100% { stroke-dashoffset: 0; opacity: 0.15; }
}
.p1 { stroke-dasharray: 4 4; animation: flow 1.8s linear infinite 0.0s; }
.p2 { stroke-dasharray: 4 4; animation: flow 1.8s linear infinite 0.3s; }
.p3 { stroke-dasharray: 4 4; animation: flow 1.8s linear infinite 0.6s; }
.p4 { stroke-dasharray: 4 4; animation: flow 1.8s linear infinite 0.9s; }
.p5 { stroke-dasharray: 4 4; animation: flow 1.8s linear infinite 1.2s; }
.p6 { stroke-dasharray: 4 4; animation: flow 1.8s linear infinite 1.5s; }

hr { border-color: rgba(240,230,204,0.07) !important; margin: 1rem 0 !important; }
</style>

<svg class="geo-art" viewBox="0 0 700 900" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
  <polygon points="280,80 480,80 480,480 280,480" fill="none" stroke="#F0E6CC" stroke-width="1.1" opacity="0.38"/>
  <polygon points="380,120 580,120 580,520 380,520" fill="none" stroke="#F0E6CC" stroke-width="1.1" opacity="0.22"/>
  <line x1="480" y1="480" x2="700" y2="500" stroke="#F0E6CC" stroke-width="0.5" opacity="0.22"/>
  <line x1="480" y1="480" x2="700" y2="526" stroke="#F0E6CC" stroke-width="0.5" opacity="0.20"/>
  <line x1="480" y1="480" x2="700" y2="552" stroke="#F0E6CC" stroke-width="0.5" opacity="0.18"/>
  <line x1="480" y1="480" x2="700" y2="578" stroke="#F0E6CC" stroke-width="0.5" opacity="0.16"/>
  <line x1="480" y1="480" x2="700" y2="604" stroke="#F0E6CC" stroke-width="0.5" opacity="0.14"/>
  <line x1="480" y1="480" x2="700" y2="630" stroke="#F0E6CC" stroke-width="0.5" opacity="0.12"/>
  <line x1="480" y1="480" x2="700" y2="656" stroke="#F0E6CC" stroke-width="0.5" opacity="0.10"/>
  <line x1="480" y1="480" x2="680" y2="900" stroke="#F0E6CC" stroke-width="0.5" opacity="0.08"/>
  <line x1="480" y1="480" x2="610" y2="900" stroke="#F0E6CC" stroke-width="0.5" opacity="0.06"/>
  <line x1="580" y1="520" x2="700" y2="545" stroke="#F0E6CC" stroke-width="0.4" opacity="0.14"/>
  <line x1="580" y1="520" x2="700" y2="578" stroke="#F0E6CC" stroke-width="0.4" opacity="0.11"/>
  <line x1="580" y1="520" x2="700" y2="614" stroke="#F0E6CC" stroke-width="0.4" opacity="0.08"/>
  <line x1="480" y1="80" x2="700" y2="80" stroke="#F0E6CC" stroke-width="0.7" opacity="0.32"/>
</svg>
""", unsafe_allow_html=True)

# ─── Session state ───────────────────────────────────────────────────────────
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

# ─── Two-column layout ───────────────────────────────────────────────────────
left, right = st.columns([5, 2], gap="large")

with left:
    st.markdown("""
<p class="eyebrow">Document intelligence</p>
<div class="hero-title">Hi, I'm<br><em>Sourcerer</em></div>
<div class="divider-line"></div>
<p class="hero-tagline">Your PDF, actually answered.</p>
<p class="hero-desc">Upload any PDF. Ask anything. Sourcerer finds the right passage, verifies the answer is grounded in the source, and streams it back with a confidence badge.</p>
""", unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

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

            t = threading.Thread(target=ingest_worker, args=(temp_path, result_container), daemon=True)
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
            st.error(f"❌ {st.session_state.ingest_error}")
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

                context = "\n\n---\n\n".join(context_chunks) if isinstance(context_chunks, list) else context_chunks
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
        st.markdown('<p style="color:rgba(240,230,204,0.22);font-size:0.82rem;margin-top:0.2rem;">Upload a PDF to get started.</p>', unsafe_allow_html=True)

# ─── Right column ────────────────────────────────────────────────────────────
with right:
    b1, b2 = st.columns(2, gap="small")
    with b1:
        if st.button("⚡ Pipeline"):
            st.session_state.show_pipeline = not st.session_state.show_pipeline
            st.session_state.show_info = False
    with b2:
        if st.button("ℹ Info"):
            st.session_state.show_info = not st.session_state.show_info
            st.session_state.show_pipeline = False

    if st.session_state.show_info:
        st.markdown("""
<div class="panel-card">
  <div class="panel-label">About Sourcerer</div>
  <div style="margin-bottom:10px">
    <div class="info-row"><span class="info-stage">S0</span><span class="info-text">Basic RAG — parse, chunk, embed, retrieve</span></div>
    <div class="info-row"><span class="info-stage">S1</span><span class="info-text">Baseline report card, 4 test sets</span></div>
    <div class="info-row"><span class="info-stage">S2</span><span class="info-text">BM25 + vector hybrid + RRF fusion</span></div>
    <div class="info-row"><span class="info-stage">S3</span><span class="info-text">Cross-encoder reranking (BGE)</span></div>
    <div class="info-row"><span class="info-stage">S4</span><span class="info-text">HyDE query expansion (tested)</span></div>
    <div class="info-row"><span class="info-stage">S5</span><span class="info-text">LLM-as-judge trust check</span></div>
    <div class="info-row"><span class="info-stage">S6</span><span class="info-text">Chroma vector database</span></div>
    <div class="info-row"><span class="info-stage">S7</span><span class="info-text">Messy PDF handling + guardrails</span></div>
    <div class="info-row"><span class="info-stage">S8</span><span class="info-text">Background threading</span></div>
    <div class="info-row"><span class="info-stage">S9</span><span class="info-text">Streaming answers</span></div>
    <div class="info-row"><span class="info-stage">S9.5</span><span class="info-text">4-turn conversation memory</span></div>
    <div class="info-row"><span class="info-stage">S10</span><span class="info-text">Deployed to Streamlit Cloud</span></div>
  </div>
  <div style="border-top:1px solid rgba(240,230,204,0.07);padding-top:10px">
    <span class="tech-pill">Python</span>
    <span class="tech-pill">Streamlit</span>
    <span class="tech-pill">Chroma</span>
    <span class="tech-pill">BGE reranker</span>
    <span class="tech-pill">BM25+RRF</span>
    <span class="tech-pill">Gemini</span>
    <span class="tech-pill">MiniLM</span>
  </div>
</div>
""", unsafe_allow_html=True)

    if st.session_state.show_pipeline:
        st.markdown("""
<div class="panel-card">
  <div class="panel-label">Pipeline</div>
  <svg width="100%" viewBox="0 0 160 560" xmlns="http://www.w3.org/2000/svg">

    <text x="80" y="14" text-anchor="middle" style="fill:rgba(240,230,204,0.2);font-size:7px;font-family:Inter,sans-serif;font-weight:600;letter-spacing:0.1em">INGEST</text>

    <rect x="20" y="20" width="120" height="30" rx="5" fill="rgba(200,52,10,0.1)" stroke="rgba(200,52,10,0.35)" stroke-width="0.7"/>
    <text x="80" y="32" text-anchor="middle" style="fill:rgba(240,230,204,0.82);font-size:10px;font-family:Inter,sans-serif;font-weight:600">PDF upload</text>
    <text x="80" y="44" text-anchor="middle" style="fill:rgba(240,230,204,0.28);font-size:8px;font-family:Inter,sans-serif">parse · chunk · embed</text>

    <line class="p1" x1="80" y1="51" x2="80" y2="68" stroke="rgba(240,230,204,0.22)" stroke-width="0.8"/>
    <polygon points="76,66 80,74 84,66" fill="rgba(240,230,204,0.22)"/>

    <rect x="20" y="76" width="120" height="30" rx="5" fill="rgba(240,230,204,0.03)" stroke="rgba(240,230,204,0.15)" stroke-width="0.7"/>
    <text x="80" y="88" text-anchor="middle" style="fill:rgba(240,230,204,0.82);font-size:10px;font-family:Inter,sans-serif;font-weight:600">Chroma</text>
    <text x="80" y="100" text-anchor="middle" style="fill:rgba(240,230,204,0.28);font-size:8px;font-family:Inter,sans-serif">vector store</text>

    <line x1="20" y1="124" x2="140" y2="124" stroke="rgba(240,230,204,0.07)" stroke-width="0.5" stroke-dasharray="3 3"/>
    <text x="80" y="136" text-anchor="middle" style="fill:rgba(240,230,204,0.18);font-size:7px;font-family:Inter,sans-serif;font-weight:600;letter-spacing:0.1em">QUERY</text>

    <rect x="20" y="142" width="120" height="30" rx="5" fill="rgba(240,230,204,0.03)" stroke="rgba(240,230,204,0.15)" stroke-width="0.7"/>
    <text x="80" y="154" text-anchor="middle" style="fill:rgba(240,230,204,0.82);font-size:10px;font-family:Inter,sans-serif;font-weight:600">Question</text>
    <text x="80" y="166" text-anchor="middle" style="fill:rgba(240,230,204,0.28);font-size:8px;font-family:Inter,sans-serif">user input</text>

    <line class="p2" x1="80" y1="173" x2="80" y2="190" stroke="rgba(240,230,204,0.22)" stroke-width="0.8"/>
    <polygon points="76,188 80,196 84,188" fill="rgba(240,230,204,0.22)"/>

    <rect x="20" y="198" width="120" height="30" rx="5" fill="rgba(240,230,204,0.03)" stroke="rgba(240,230,204,0.15)" stroke-width="0.7"/>
    <text x="80" y="210" text-anchor="middle" style="fill:rgba(240,230,204,0.82);font-size:10px;font-family:Inter,sans-serif;font-weight:600">Rewrite</text>
    <text x="80" y="222" text-anchor="middle" style="fill:rgba(240,230,204,0.28);font-size:8px;font-family:Inter,sans-serif">query expansion</text>

    <line class="p3" x1="80" y1="229" x2="80" y2="246" stroke="rgba(240,230,204,0.22)" stroke-width="0.8"/>
    <polygon points="76,244 80,252 84,244" fill="rgba(240,230,204,0.22)"/>

    <rect x="20" y="254" width="120" height="30" rx="5" fill="rgba(240,230,204,0.03)" stroke="rgba(240,230,204,0.15)" stroke-width="0.7"/>
    <text x="80" y="266" text-anchor="middle" style="fill:rgba(240,230,204,0.82);font-size:10px;font-family:Inter,sans-serif;font-weight:600">Retrieve</text>
    <text x="80" y="278" text-anchor="middle" style="fill:rgba(240,230,204,0.28);font-size:8px;font-family:Inter,sans-serif">BM25 + vector + RRF</text>

    <line class="p4" x1="80" y1="285" x2="80" y2="302" stroke="rgba(240,230,204,0.22)" stroke-width="0.8"/>
    <polygon points="76,300 80,308 84,300" fill="rgba(240,230,204,0.22)"/>

    <rect x="20" y="310" width="120" height="30" rx="5" fill="rgba(240,230,204,0.03)" stroke="rgba(240,230,204,0.15)" stroke-width="0.7"/>
    <text x="80" y="322" text-anchor="middle" style="fill:rgba(240,230,204,0.82);font-size:10px;font-family:Inter,sans-serif;font-weight:600">Rerank</text>
    <text x="80" y="334" text-anchor="middle" style="fill:rgba(240,230,204,0.28);font-size:8px;font-family:Inter,sans-serif">cross-encoder BGE</text>

    <line class="p5" x1="80" y1="341" x2="80" y2="358" stroke="rgba(240,230,204,0.22)" stroke-width="0.8"/>
    <polygon points="76,356 80,364 84,356" fill="rgba(240,230,204,0.22)"/>

    <rect x="20" y="366" width="120" height="30" rx="5" fill="rgba(240,230,204,0.03)" stroke="rgba(240,230,204,0.15)" stroke-width="0.7"/>
    <text x="80" y="378" text-anchor="middle" style="fill:rgba(240,230,204,0.82);font-size:10px;font-family:Inter,sans-serif;font-weight:600">Gemini</text>
    <text x="80" y="390" text-anchor="middle" style="fill:rgba(240,230,204,0.28);font-size:8px;font-family:Inter,sans-serif">stream answer</text>

    <line class="p6" x1="80" y1="397" x2="80" y2="414" stroke="rgba(240,230,204,0.22)" stroke-width="0.8"/>
    <polygon points="76,412 80,420 84,412" fill="rgba(240,230,204,0.22)"/>

    <rect x="20" y="422" width="120" height="30" rx="5" fill="rgba(200,52,10,0.08)" stroke="rgba(200,52,10,0.28)" stroke-width="0.7"/>
    <text x="80" y="434" text-anchor="middle" style="fill:rgba(240,230,204,0.82);font-size:10px;font-family:Inter,sans-serif;font-weight:600">Answer</text>
    <text x="80" y="446" text-anchor="middle" style="fill:rgba(240,230,204,0.28);font-size:8px;font-family:Inter,sans-serif">trust check · confidence</text>

  </svg>
</div>
""", unsafe_allow_html=True)