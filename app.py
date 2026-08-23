import os
import time
import tempfile
import threading
import streamlit as st

from main import get_collection
from src.embed import embed_query
from src.keyword_search import build_bm25_index
from src.stage3_rerank import rerank
from src.stage5_trust import generate_trusted_answer
from src.generate import generate_answer
from src.fusion import hybrid_retrieve_chroma, expand_chunks
from src.parse import check_file_limits

# ─── Page config ───────────────────────────────────────────────────────────
st.set_page_config(page_title="Sourcerer", page_icon="📄")
st.title("Sourcerer")
st.write("Upload a PDF and ask questions about it.")

# ─── Session state defaults ─────────────────────────────────────────────────
# These keys must exist before any conditional reads below.
for key, default in {
    "thread": None,
    "collection": None,
    "ingest_error": None,
    "last_filename": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─── File upload ────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:

    # ─── Detect new file ────────────────────────────────────────────────────
    # If the user uploads a different file mid-session, reset everything.
    if uploaded_file.name != st.session_state.last_filename:
        st.session_state.collection = None
        st.session_state.ingest_error = None
        st.session_state.thread = None
        st.session_state.last_filename = uploaded_file.name

    # ─── Start background thread (only once) ────────────────────────────────
    thread_idle = st.session_state.thread is None
    thread_done = (
        st.session_state.thread is not None
        and not st.session_state.thread.is_alive()
    )
    collection_ready = st.session_state.collection is not None

    if not collection_ready and thread_idle:
        # Save PDF to a temp file. delete=False because the thread
        # will read it after this block closes.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getbuffer())
            temp_path = tmp.name

        # Validate file limits before even starting the thread.
        ok, error_msg = check_file_limits(temp_path)
        if not ok:
            st.error(f"❌ {error_msg}")
            st.stop()

        # Worker function — runs in background thread.
        # Writes result or error into session_state when done.
        def ingest_worker(path):
            try:
                result = get_collection(path)
                st.session_state.collection = result
            except Exception as e:
                st.session_state.ingest_error = str(e)
            finally:
                os.unlink(path)  # clean up temp file regardless

        t = threading.Thread(target=ingest_worker, args=(temp_path,), daemon=True)
        t.start()
        st.session_state.thread = t

    # ─── Poll while thread is running ───────────────────────────────────────
    if st.session_state.thread is not None and st.session_state.thread.is_alive():
        st.spinner("Processing document...")
        time.sleep(1)
        st.rerun()

    # ─── Handle thread failure ───────────────────────────────────────────────
    elif st.session_state.ingest_error is not None:
        st.error(f"❌ Failed to process PDF: {st.session_state.ingest_error}")
        st.stop()

    # ─── Collection is ready ─────────────────────────────────────────────────
    elif st.session_state.collection is not None:
        collection = st.session_state.collection

        all_data = collection.get(include=["documents"])
        chunks = all_data["documents"]
        bm25_index = build_bm25_index(chunks)

        st.success(f"✅ Loaded {len(chunks)} chunks from {uploaded_file.name}")

        question = st.text_input("Ask a question about the document")

        if question:
            with st.spinner("Searching..."):
                candidates = hybrid_retrieve_chroma(
                    question, embed_query(question), collection, chunks,
                    bm25_index, top_k=12, candidate_k=12
                )
                top_results = rerank(question, candidates, top_k=3)
                expanded = expand_chunks(top_results, chunks, neighbors=1)
                context_chunks = [chunk for _, chunk in expanded]
                top_index = top_results[0][0]
                score = top_results[0][2]

                answer, confidence, verdict = generate_trusted_answer(
                    question, context_chunks, generate_answer
                )

            st.markdown("### Answer")
            if verdict == "SUPPORTED":
                st.success(f"🟢 {confidence}")
            else:
                st.warning(f"🟡 {confidence}")
            st.write(answer)

            with st.expander(f"Show source chunks (top chunk #{top_index}, reranker score: {score:.4f})"):
                st.write(f"**{len(context_chunks)} chunks used as context:**")
                for chunk_idx, chunk_text in expanded:
                    st.write(f"**Chunk #{chunk_idx}:**")
                    st.write(chunk_text)
                    st.divider()

else:
    st.info("Upload a PDF to get started.")