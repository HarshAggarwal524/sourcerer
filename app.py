import os
import tempfile
import streamlit as st

from main import load_or_build, MODEL_NAME
from src.embed import embed_query
from src.retrieve import retrieve
from src.generate import generate_answer

st.set_page_config(page_title="Sourcerer", page_icon="📄")
st.title("Sourcerer")
st.write("Upload a PDF and ask questions about it.")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    # Save uploaded file to a temp path so existing functions (which expect
    # a file path, not an in-memory object) work unchanged.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        temp_path = tmp_file.name

    with st.spinner("Processing document..."):
        try:
            chunks, embeddings = load_or_build(temp_path)
        except Exception as e:
            st.error(f"Failed to process PDF: {e}")
            st.stop()

    st.success(f"Loaded {len(chunks)} chunks from {uploaded_file.name}")

    question = st.text_input("Ask a question about the document")

    if question:
        with st.spinner("Searching..."):
            query_vec = embed_query(question)
            results = retrieve(query_vec, embeddings, chunks, top_k=1)
            top_index, top_chunk, score = results[0]
            answer = generate_answer(question, top_chunk)

        st.markdown("### Answer")
        st.write(answer)

        with st.expander(f"Show source chunk (similarity score: {score:.4f})"):
            st.write(top_chunk)
else:
    st.info("Upload a PDF to get started.")