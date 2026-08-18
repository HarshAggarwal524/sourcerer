import os
import tempfile
import streamlit as st

from main import get_collection, MODEL_NAME
from src.embed import embed_query
from src.keyword_search import build_bm25_index
from src.stage3_rerank import rerank
from src.stage5_trust import generate_trusted_answer
from src.generate import generate_answer
from src.fusion import hybrid_retrieve_chroma, expand_chunks

# ─── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Sourcerer", page_icon="📄")
st.title("Sourcerer")
st.write("Upload a PDF and ask questions about it.")

# ─── File upload ───────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:

    # ─── Save to temp file ─────────────────────────────────────────────────────
    # delete=False: keep the file alive after the block closes since
    # get_collection() needs to open it again by path afterward.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        temp_path = tmp_file.name

    # ─── Load or build Chroma collection ───────────────────────────────────────
    # get_collection() checks Chroma for an existing collection keyed by
    # hash(PDF bytes + model name). First run: parses → chunks → embeds → stores.
    # Subsequent runs: loads instantly from Chroma.
    with st.spinner("Processing document..."):
        try:
            from src.parse import check_file_limits
            ok, error_msg = check_file_limits(temp_path)
            if not ok:
                st.error(f"❌ {error_msg}")
                st.stop()

            collection = get_collection(temp_path)
        except ValueError as e:
            st.error(f"❌ {str(e)}")
            st.stop()
        except Exception as e:
            st.error(f"❌ Failed to process PDF: {e}")
            st.stop()

    # ─── Fetch chunks and build BM25 index ─────────────────────────────────────
    # Chunks are fetched from Chroma (no numpy embeddings needed anymore).
    # BM25 index is rebuilt each run — fast enough for this document size.
    all_data = collection.get(include=["documents"])
    chunks = all_data["documents"]
    bm25_index = build_bm25_index(chunks)

    st.success(f"Loaded {len(chunks)} chunks from {uploaded_file.name}")

    # ─── Question input ─────────────────────────────────────────────────────────
    question = st.text_input("Ask a question about the document")

    if question:
        with st.spinner("Searching..."):

            # ─── Hybrid retrieval (Stage 2+6) ───────────────────────────────────
            # Chroma vector search + BM25 keyword search → RRF fusion
            candidates = hybrid_retrieve_chroma(
                question, embed_query(question), collection, chunks,
                bm25_index, top_k=12, candidate_k=12
            )

            # ─── Reranking (Stage 3) ────────────────────────────────────────────
            # Cross-encoder re-scores top 12 candidates, keeps top 3
            top_results = rerank(question, candidates, top_k=3)

            # ─── Context expansion (Stage 7) ────────────────────────────────────
            # Each top chunk gets 1 neighbor on each side for boundary context.
            # Deduplicates overlaps, sorts by document order.
            expanded = expand_chunks(top_results, chunks, neighbors=1)
            context_chunks = [chunk for _, chunk in expanded]

            top_index = top_results[0][0]
            score = top_results[0][2]

            # ─── Generate + trust check (Stage 5) ──────────────────────────────
            # Generates answer from context, then LLM judge verifies grounding.
            answer, confidence, verdict = generate_trusted_answer(
                question, context_chunks, generate_answer
            )

        # ─── Display answer ─────────────────────────────────────────────────────
        st.markdown("### Answer")

        if verdict == "SUPPORTED":
            st.success(f"🟢 {confidence}")
        else:
            st.warning(f"🟡 {confidence}")

        st.write(answer)

        # ─── Source chunks expander ──────────────────────────────────────────────
        with st.expander(f"Show source chunks (top chunk #{top_index}, reranker score: {score:.4f})"):
            st.write(f"**{len(context_chunks)} chunks used as context:**")
            for chunk_idx, chunk_text in expanded:
                st.write(f"**Chunk #{chunk_idx}:**")
                st.write(chunk_text)
                st.divider()

else:
    st.info("Upload a PDF to get started.")