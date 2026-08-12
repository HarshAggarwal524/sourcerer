import os
import tempfile
import streamlit as st

from main import load_or_build, MODEL_NAME
from src.embed import embed_query
from src.keyword_search import build_bm25_index
from src.fusion import hybrid_retrieve
from src.stage3_rerank import rerank
from src.stage5_trust import generate_trusted_answer
from src.generate import generate_answer

# ─── Page config ───────────────────────────────────────────────────────────────
# Sets the browser tab title and icon. Must be the first Streamlit call.
st.set_page_config(page_title="Sourcerer", page_icon="📄")
st.title("Sourcerer")
st.write("Upload a PDF and ask questions about it.")


# ─── File upload ───────────────────────────────────────────────────────────────
# Streamlit's file uploader returns an in-memory file object, not a path on disk.
# Our pipeline functions (load_or_build, etc.) expect a real file path, so we
# save the uploaded bytes to a temp file first.
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:

    # ─── Save to temp file ─────────────────────────────────────────────────────
    # delete=False: keep the file alive after the 'with' block closes, since
    # load_or_build() needs to open it again by path afterward.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        temp_path = tmp_file.name

    # ─── Parse, chunk, embed (or load from cache) ──────────────────────────────
    # load_or_build() checks for a cached pickle keyed by hash(PDF bytes + model).
    # First run: parses PDF → chunks → embeds → saves cache.
    # Subsequent runs: loads instantly from cache.
    # Streamlit reruns the whole script on every interaction, but since the cache
    # file persists on disk, this always hits the fast path after the first upload.
    with st.spinner("Processing document..."):
        try:
            chunks, embeddings = load_or_build(temp_path)
        except Exception as e:
            st.error(f"Failed to process PDF: {e}")
            st.stop()  # halt execution — no point showing a text input with no data

    # ─── Build BM25 index ──────────────────────────────────────────────────────
    # BM25 index is built over the same chunks each time. Fast enough that
    # we don't need to cache it separately — takes milliseconds.
    bm25_index = build_bm25_index(chunks)

    st.success(f"Loaded {len(chunks)} chunks from {uploaded_file.name}")

    # ─── Question input ────────────────────────────────────────────────────────
    # st.text_input re-renders every time the user types and hits Enter.
    # The 'if question:' guard prevents running retrieval on an empty string.
    question = st.text_input("Ask a question about the document")

    if question:
        with st.spinner("Searching..."):

            # ─── Stage 3 retrieval pipeline ────────────────────────────────────
            # Step 1: hybrid_retrieve — vector search + BM25 keyword search,
            #         merged via Reciprocal Rank Fusion. Returns top 12 candidates.
            # Step 2: rerank — cross-encoder reads question+chunk pairs together,
            #         re-scores candidates for genuine relevance. Returns top 1.
            candidates = hybrid_retrieve(
                question, chunks, embeddings, bm25_index,
                top_k=12, candidate_k=12
            )
            results = rerank(question, candidates, top_k=1)
            top_index, top_chunk, score = results[0]

            # ─── Stage 5 trust check ───────────────────────────────────────────
            # generate_trusted_answer():
            #   1. Generates an answer from the retrieved chunk (via Groq LLM).
            #   2. Makes a second LLM call to judge: is the answer fully supported
            #      by the source chunk, with no outside information added?
            #   3. If SUPPORTED → returns answer + "HIGH CONFIDENCE"
            #   4. If NOT_SUPPORTED → returns refusal message + "LOW CONFIDENCE"
            answer, confidence, verdict = generate_trusted_answer(
                question, top_chunk, generate_answer
            )

        # ─── Display answer ────────────────────────────────────────────────────
        st.markdown("### Answer")

        # Show a colored confidence badge above the answer.
        # st.success = green banner, st.warning = yellow banner.
        if verdict == "SUPPORTED":
            st.success(f"🟢 {confidence}")
        else:
            st.warning(f"🟡 {confidence}")

        st.write(answer)

        # ─── Source chunk expander ─────────────────────────────────────────────
        # Shows the raw chunk the answer was generated from, so the user can
        # manually verify the answer against the source text — the human-readable
        # equivalent of what Stage 5's LLM judge does automatically.
        # Score here is the reranker's cross-encoder score, not cosine similarity.
        with st.expander(f"Show source chunk (reranker score: {score:.4f})"):
            st.write(f"**Chunk #{top_index}**")
            st.write(top_chunk)

else:
    # ─── Empty state ───────────────────────────────────────────────────────────
    # Shown before any file is uploaded — gives the user a clear starting prompt.
    st.info("Upload a PDF to get started.")