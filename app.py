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
from src.llm import stream_llm
from src.stage5_trust import check_grounding

# ─── Page config ───────────────────────────────────────────────────────────
st.set_page_config(page_title="Sourcerer", page_icon="📄")
st.title("Sourcerer")
st.write("Upload a PDF and ask questions about it.")

# ─── Session state defaults ─────────────────────────────────────────────────
for key, default in {
    "thread": None,
    "collection": None,
    "ingest_error": None,
    "last_filename": None,
    "result_container": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─── File upload ────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:

    # ─── Detect new file ────────────────────────────────────────────────────
    if uploaded_file.name != st.session_state.last_filename:
        st.session_state.collection = None
        st.session_state.ingest_error = None
        st.session_state.thread = None
        st.session_state.result_container = None
        st.session_state.last_filename = uploaded_file.name

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

    # ─── Start background thread (only once) ────────────────────────────────
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

    # ─── Poll while thread is running ───────────────────────────────────────
    elif thread_running:
        with st.spinner("Processing document..."):
            time.sleep(1)
        st.rerun()

    # ─── Thread just finished — read result into session state ───────────────
    elif thread_finished:
        container = st.session_state.result_container
        if container["error"]:
            st.session_state.ingest_error = container["error"]
        else:
            st.session_state.collection = container["collection"]
        st.rerun()

    # ─── Handle failure ──────────────────────────────────────────────────────
    elif st.session_state.ingest_error is not None:
        st.error(f"❌ Failed to process PDF: {st.session_state.ingest_error}")
        st.stop()

    # ─── Collection ready — show Q&A ─────────────────────────────────────────
    elif collection_ready:
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

            st.markdown("### Answer")

            # ─── Stream the answer ───────────────────────────────────────────
            if isinstance(context_chunks, list):
                context = "\n\n---\n\n".join(context_chunks)
            else:
                context = context_chunks

            prompt = (
                "Answer the question using only the following context. "
                "If the answer isn't in the context, say so clearly instead of guessing.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {question}"
            )

            streamed_text = st.write_stream(stream_llm(prompt))

            # ─── Trust check on completed answer ────────────────────────────
            verdict = check_grounding(question, context_chunks, streamed_text)

            if verdict == "SUPPORTED":
                st.success("🟢 HIGH CONFIDENCE")
            else:
                st.warning("🟡 LOW CONFIDENCE")

            # ─── Source chunks expander ──────────────────────────────────────
            with st.expander(f"Show source chunks (top chunk #{top_index}, reranker score: {score:.4f})"):
                st.write(f"**{len(context_chunks)} chunks used as context:**")
                for chunk_idx, chunk_text in expanded:
                    st.write(f"**Chunk #{chunk_idx}:**")
                    st.write(chunk_text)
                    st.divider()

else:
    st.info("Upload a PDF to get started.")