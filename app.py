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

st.set_page_config(page_title="Sourcerer", page_icon="📄", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }

.hero-title {
    font-size: 3rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    background: linear-gradient(90deg, #FF6B00, #FF9A00);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.25rem;
}
.hero-sub {
    font-size: 1rem;
    color: #666;
    font-weight: 400;
    margin-bottom: 1rem;
}
[data-testid="stFileUploader"] {
    border: 1.5px dashed #FF6B00 !important;
    border-radius: 12px !important;
    padding: 1rem !important;
    background: #111 !important;
    transition: all 0.2s ease;
}
[data-testid="stFileUploader"]:hover {
    border-color: #FF9A00 !important;
    background: #1a1a1a !important;
}
[data-testid="stTextInput"] input {
    border: 1.5px solid #333 !important;
    border-radius: 10px !important;
    background: #111 !important;
    color: #fff !important;
    font-size: 1rem !important;
    padding: 0.75rem 1rem !important;
    transition: border 0.2s ease;
}
[data-testid="stTextInput"] input:focus {
    border-color: #FF6B00 !important;
    box-shadow: 0 0 0 3px rgba(255,107,0,0.15) !important;
}
[data-testid="stExpander"] {
    border: 1px solid #222 !important;
    border-radius: 10px !important;
    background: #111 !important;
}
[data-testid="stButton"] button {
    background: #111 !important;
    border: 1px solid #333 !important;
    border-radius: 8px !important;
    color: #888 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    padding: 6px 14px !important;
    transition: all 0.2s ease !important;
}
[data-testid="stButton"] button:hover {
    border-color: #FF6B00 !important;
    color: #FF6B00 !important;
}
.badge-high {
    display: inline-block;
    background: rgba(0,200,100,0.1);
    color: #00C864;
    border: 1px solid rgba(0,200,100,0.3);
    border-radius: 999px;
    padding: 0.25rem 0.85rem;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    margin-top: 0.75rem;
}
.badge-low {
    display: inline-block;
    background: rgba(255,107,0,0.1);
    color: #FF6B00;
    border: 1px solid rgba(255,107,0,0.3);
    border-radius: 999px;
    padding: 0.25rem 0.85rem;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    margin-top: 0.75rem;
}
.info-section { margin-bottom: 1.25rem; }
.info-section h3 {
    color: #FF6B00;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.info-section p, .info-section li {
    color: #aaa;
    font-size: 0.9rem;
    line-height: 1.6;
}
.info-section ul { padding-left: 1.2rem; margin: 0; }
.tech-pill {
    display: inline-block;
    background: #1a1a1a;
    border: 1px solid #333;
    border-radius: 999px;
    padding: 3px 10px;
    font-size: 0.75rem;
    color: #aaa;
    margin: 3px;
}
@keyframes flow {
    0% { stroke-dashoffset: 24; opacity: 0.3; }
    50% { opacity: 1; }
    100% { stroke-dashoffset: 0; opacity: 0.3; }
}
.pipe  { stroke-dasharray: 6 6; animation: flow 1.4s linear infinite 0.0s; }
.pipe2 { stroke-dasharray: 6 6; animation: flow 1.4s linear infinite 0.2s; }
.pipe3 { stroke-dasharray: 6 6; animation: flow 1.4s linear infinite 0.4s; }
.pipe4 { stroke-dasharray: 6 6; animation: flow 1.4s linear infinite 0.6s; }
.pipe5 { stroke-dasharray: 6 6; animation: flow 1.4s linear infinite 0.8s; }
.pipe6 { stroke-dasharray: 6 6; animation: flow 1.4s linear infinite 1.0s; }
hr { border-color: #222 !important; margin: 1.5rem 0 !important; }
</style>
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

# ─── Hero + top buttons ──────────────────────────────────────────────────────
col1, col2, col3 = st.columns([6, 1, 1])
with col1:
    st.markdown('<div class="hero-title">Sourcerer</div><div class="hero-sub">Upload a PDF. Ask anything.</div>', unsafe_allow_html=True)
with col2:
    if st.button("⚡ Pipeline"):
        st.session_state.show_pipeline = not st.session_state.show_pipeline
        st.session_state.show_info = False
with col3:
    if st.button("? Info"):
        st.session_state.show_info = not st.session_state.show_info
        st.session_state.show_pipeline = False

# ─── Info panel ──────────────────────────────────────────────────────────────
if st.session_state.show_info:
    with st.expander("About Sourcerer", expanded=True):
        st.markdown("""
<div class="info-section">
  <h3>What it does</h3>
  <p>Upload any PDF and ask questions about it. Sourcerer finds the most relevant passages,
  verifies the answer is grounded in the source, and streams it back word by word —
  with a confidence badge telling you how much to trust it.</p>
</div>

<div class="info-section">
  <h3>How it was built — 12 stages</h3>
  <ul>
    <li><b style="color:#fff">Stage 0</b> — Basic RAG pipeline: parse, chunk, embed, retrieve, generate</li>
    <li><b style="color:#fff">Stage 1</b> — Baseline report card across 4 test sets (easy / hard / hardest / ambiguous)</li>
    <li><b style="color:#fff">Stage 2</b> — Hybrid BM25 + vector search with RRF fusion</li>
    <li><b style="color:#fff">Stage 3</b> — Cross-encoder reranking (bge-reranker-v2-m3)</li>
    <li><b style="color:#fff">Stage 4</b> — HyDE query expansion (tested, not in final pipeline)</li>
    <li><b style="color:#fff">Stage 5</b> — LLM-as-judge trust check for grounding verification</li>
    <li><b style="color:#fff">Stage 6</b> — Chroma vector database replacing pickle storage</li>
    <li><b style="color:#fff">Stage 7</b> — Messy PDF handling, scanned PDF detection, guardrails</li>
    <li><b style="color:#fff">Stage 8</b> — Non-blocking background ingestion with threading</li>
    <li><b style="color:#fff">Stage 9</b> — Streaming answers token by token</li>
    <li><b style="color:#fff">Stage 9.5</b> — Conversation memory with 4-turn query rewriting</li>
    <li><b style="color:#fff">Stage 10</b> — Deployed to Streamlit Community Cloud</li>
  </ul>
</div>

<div class="info-section">
  <h3>Tech stack</h3>
  <span class="tech-pill">Python</span>
  <span class="tech-pill">Streamlit</span>
  <span class="tech-pill">Chroma</span>
  <span class="tech-pill">sentence-transformers</span>
  <span class="tech-pill">bge-reranker-v2-m3</span>
  <span class="tech-pill">BM25 + RRF</span>
  <span class="tech-pill">Gemini</span>
  <span class="tech-pill">Google Colab</span>
</div>
""", unsafe_allow_html=True)

# ─── Pipeline panel ──────────────────────────────────────────────────────────
if st.session_state.show_pipeline:
    with st.expander("Pipeline — how your PDF becomes an answer", expanded=True):
        st.markdown("""
<svg width="100%" viewBox="0 0 680 480" style="margin-top:0.5rem">
  <defs>
    <marker id="arr2" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
  </defs>

  <text x="340" y="28" text-anchor="middle" style="fill:#555;font-size:11px;font-family:Inter,sans-serif;font-weight:600;letter-spacing:0.08em">INGEST TIME</text>

  <rect x="20" y="42" width="80" height="56" rx="8" fill="#1a0d00" stroke="#FF6B00" stroke-width="0.5"/>
  <text x="60" y="65" text-anchor="middle" style="fill:#FF9A00;font-size:13px;font-family:Inter,sans-serif;font-weight:600">PDF</text>
  <text x="60" y="83" text-anchor="middle" style="fill:#664400;font-size:11px;font-family:Inter,sans-serif">upload</text>

  <line class="pipe" x1="102" y1="70" x2="136" y2="70" stroke="#FF6B00" stroke-width="1.5" fill="none" marker-end="url(#arr2)"/>

  <rect x="138" y="42" width="88" height="56" rx="8" fill="#1a1200" stroke="#BA7517" stroke-width="0.5"/>
  <text x="182" y="65" text-anchor="middle" style="fill:#EF9F27;font-size:13px;font-family:Inter,sans-serif;font-weight:600">Parse</text>
  <text x="182" y="83" text-anchor="middle" style="fill:#664400;font-size:11px;font-family:Inter,sans-serif">extract text</text>

  <line class="pipe2" x1="228" y1="70" x2="262" y2="70" stroke="#BA7517" stroke-width="1.5" fill="none" marker-end="url(#arr2)"/>

  <rect x="264" y="42" width="88" height="56" rx="8" fill="#1a1200" stroke="#BA7517" stroke-width="0.5"/>
  <text x="308" y="65" text-anchor="middle" style="fill:#EF9F27;font-size:13px;font-family:Inter,sans-serif;font-weight:600">Chunk</text>
  <text x="308" y="83" text-anchor="middle" style="fill:#664400;font-size:11px;font-family:Inter,sans-serif">300 words</text>

  <line class="pipe3" x1="354" y1="70" x2="388" y2="70" stroke="#BA7517" stroke-width="1.5" fill="none" marker-end="url(#arr2)"/>

  <rect x="390" y="42" width="88" height="56" rx="8" fill="#0d0a1a" stroke="#534AB7" stroke-width="0.5"/>
  <text x="434" y="65" text-anchor="middle" style="fill:#AFA9EC;font-size:13px;font-family:Inter,sans-serif;font-weight:600">Embed</text>
  <text x="434" y="83" text-anchor="middle" style="fill:#3C3489;font-size:11px;font-family:Inter,sans-serif">MiniLM</text>

  <line class="pipe4" x1="480" y1="70" x2="514" y2="70" stroke="#534AB7" stroke-width="1.5" fill="none" marker-end="url(#arr2)"/>

  <rect x="516" y="42" width="90" height="56" rx="8" fill="#0d0a1a" stroke="#534AB7" stroke-width="0.5"/>
  <text x="561" y="65" text-anchor="middle" style="fill:#AFA9EC;font-size:13px;font-family:Inter,sans-serif;font-weight:600">Chroma</text>
  <text x="561" y="83" text-anchor="middle" style="fill:#3C3489;font-size:11px;font-family:Inter,sans-serif">vector store</text>

  <line x1="20" y1="126" x2="660" y2="126" stroke="#222" stroke-width="0.5" stroke-dasharray="4 4"/>
  <text x="340" y="144" text-anchor="middle" style="fill:#444;font-size:11px;font-family:Inter,sans-serif;font-weight:600;letter-spacing:0.08em">QUERY TIME</text>

  <rect x="20" y="160" width="88" height="56" rx="8" fill="#111" stroke="#444" stroke-width="0.5"/>
  <text x="64" y="183" text-anchor="middle" style="fill:#aaa;font-size:13px;font-family:Inter,sans-serif;font-weight:600">Question</text>
  <text x="64" y="201" text-anchor="middle" style="fill:#555;font-size:11px;font-family:Inter,sans-serif">user input</text>

  <line class="pipe5" x1="110" y1="188" x2="144" y2="188" stroke="#666" stroke-width="1.5" fill="none" marker-end="url(#arr2)"/>

  <rect x="146" y="160" width="88" height="56" rx="8" fill="#001a14" stroke="#0F6E56" stroke-width="0.5"/>
  <text x="190" y="183" text-anchor="middle" style="fill:#5DCAA5;font-size:13px;font-family:Inter,sans-serif;font-weight:600">Rewrite</text>
  <text x="190" y="201" text-anchor="middle" style="fill:#085041;font-size:11px;font-family:Inter,sans-serif">query expand</text>

  <line class="pipe5" x1="236" y1="188" x2="270" y2="188" stroke="#0F6E56" stroke-width="1.5" fill="none" marker-end="url(#arr2)"/>

  <rect x="272" y="160" width="100" height="56" rx="8" fill="#001a14" stroke="#0F6E56" stroke-width="0.5"/>
  <text x="322" y="183" text-anchor="middle" style="fill:#5DCAA5;font-size:13px;font-family:Inter,sans-serif;font-weight:600">Retrieve</text>
  <text x="322" y="201" text-anchor="middle" style="fill:#085041;font-size:11px;font-family:Inter,sans-serif">BM25 + vector</text>

  <path class="pipe4" d="M561 100 L561 143 L322 143 L322 158" stroke="#534AB7" stroke-width="1.5" fill="none" marker-end="url(#arr2)"/>

  <line class="pipe6" x1="374" y1="188" x2="408" y2="188" stroke="#0F6E56" stroke-width="1.5" fill="none" marker-end="url(#arr2)"/>

  <rect x="410" y="160" width="88" height="56" rx="8" fill="#00111a" stroke="#185FA5" stroke-width="0.5"/>
  <text x="454" y="183" text-anchor="middle" style="fill:#85B7EB;font-size:13px;font-family:Inter,sans-serif;font-weight:600">Rerank</text>
  <text x="454" y="201" text-anchor="middle" style="fill:#0C447C;font-size:11px;font-family:Inter,sans-serif">cross-encoder</text>

  <line class="pipe6" x1="500" y1="188" x2="534" y2="188" stroke="#185FA5" stroke-width="1.5" fill="none" marker-end="url(#arr2)"/>

  <rect x="536" y="160" width="88" height="56" rx="8" fill="#00111a" stroke="#185FA5" stroke-width="0.5"/>
  <text x="580" y="183" text-anchor="middle" style="fill:#85B7EB;font-size:13px;font-family:Inter,sans-serif;font-weight:600">Gemini</text>
  <text x="580" y="201" text-anchor="middle" style="fill:#0C447C;font-size:11px;font-family:Inter,sans-serif">generate</text>

  <path class="pipe6" d="M580 218 L580 308 L534 308" stroke="#185FA5" stroke-width="1.5" fill="none" marker-end="url(#arr2)"/>

  <rect x="272" y="280" width="100" height="56" rx="8" fill="#001a00" stroke="#3B6D11" stroke-width="0.5"/>
  <text x="322" y="303" text-anchor="middle" style="fill:#97C459;font-size:13px;font-family:Inter,sans-serif;font-weight:600">Trust check</text>
  <text x="322" y="321" text-anchor="middle" style="fill:#27500A;font-size:11px;font-family:Inter,sans-serif">LLM judge</text>

  <rect x="430" y="280" width="100" height="56" rx="8" fill="#1a0900" stroke="#FF6B00" stroke-width="0.5"/>
  <text x="480" y="303" text-anchor="middle" style="fill:#FF9A00;font-size:13px;font-family:Inter,sans-serif;font-weight:600">Answer</text>
  <text x="480" y="321" text-anchor="middle" style="fill:#664400;font-size:11px;font-family:Inter,sans-serif">streamed</text>

  <path class="pipe6" d="M374 308 L428 308" stroke="#3B6D11" stroke-width="1.5" fill="none" marker-end="url(#arr2)"/>

  <text x="20" y="390" style="fill:#444;font-size:10px;font-family:Inter,sans-serif;font-weight:600;letter-spacing:0.08em">LEGEND</text>
  <rect x="20" y="400" width="10" height="10" rx="2" fill="#1a0d00" stroke="#FF6B00" stroke-width="0.5"/>
  <text x="36" y="410" style="fill:#555;font-size:10px;font-family:Inter,sans-serif">Input/Output</text>
  <rect x="120" y="400" width="10" height="10" rx="2" fill="#1a1200" stroke="#BA7517" stroke-width="0.5"/>
  <text x="136" y="410" style="fill:#555;font-size:10px;font-family:Inter,sans-serif">Processing</text>
  <rect x="220" y="400" width="10" height="10" rx="2" fill="#0d0a1a" stroke="#534AB7" stroke-width="0.5"/>
  <text x="236" y="410" style="fill:#555;font-size:10px;font-family:Inter,sans-serif">Vectors</text>
  <rect x="300" y="400" width="10" height="10" rx="2" fill="#001a14" stroke="#0F6E56" stroke-width="0.5"/>
  <text x="316" y="410" style="fill:#555;font-size:10px;font-family:Inter,sans-serif">Retrieval</text>
  <rect x="390" y="400" width="10" height="10" rx="2" fill="#00111a" stroke="#185FA5" stroke-width="0.5"/>
  <text x="406" y="410" style="fill:#555;font-size:10px;font-family:Inter,sans-serif">LLM</text>
  <rect x="450" y="400" width="10" height="10" rx="2" fill="#001a00" stroke="#3B6D11" stroke-width="0.5"/>
  <text x="466" y="410" style="fill:#555;font-size:10px;font-family:Inter,sans-serif">Trust</text>
</svg>
""", unsafe_allow_html=True)

# ─── File upload ────────────────────────────────────────────────────────────
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
    st.markdown('<p style="color:#555;font-size:0.95rem;">Upload a PDF to get started.</p>', unsafe_allow_html=True)