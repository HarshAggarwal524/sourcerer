import os
import hashlib
from src.parse import extract_text
from src.chunking import chunk_text
from src.embed import embed_chunks, embed_query
from src.keyword_search import build_bm25_index
from src.stage3_rerank import rerank
from src.stage5_trust import generate_trusted_answer
from src.generate import generate_answer
from src.stage6_store import ingest_document, load_collection
from src.fusion import hybrid_retrieve_chroma, expand_chunks

MODEL_NAME = "all-MiniLM-L6-v2"


def process_document(pdf_path, model_name=MODEL_NAME):
    """
    Parses, chunks, embeds and stores a PDF into Chroma.
    Safe to call multiple times — skips if already ingested.
    Returns the Chroma collection.
    """
    text = extract_text(pdf_path)
    if not text:
        raise ValueError(f"Could not extract text from {pdf_path}")

    chunks = chunk_text(text, chunk_size=300)
    embeddings = embed_chunks(chunks)
    collection = ingest_document(pdf_path, chunks, embeddings, model_name)
    return collection


def get_collection(pdf_path, model_name=MODEL_NAME):
    """
    Returns the Chroma collection for this PDF, processing it first if needed.
    Single entry point for both main.py and app.py.
    """
    collection = load_collection(pdf_path, model_name)
    if collection is None:
        print("No existing collection found — processing document...")
        collection = process_document(pdf_path, model_name)
    else:
        print(f"Loaded existing collection for {pdf_path}")
    return collection


def main():
    pdf_path = "data/testpdf.pdf"

    try:
        collection = get_collection(pdf_path)
    except Exception as e:
        print(f"Failed to load document: {e}")
        return

    all_data = collection.get(include=["documents"])
    chunks = all_data["documents"]
    bm25_index = build_bm25_index(chunks)

    print(f"\nReady. Loaded {len(chunks)} chunks.")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("Ask a question: ")
        if question.strip().lower() == "exit":
            print("Goodbye.")
            break

        query_vec = embed_query(question)

        # Stage 2+6: hybrid retrieval via Chroma + BM25 + RRF
        candidates = hybrid_retrieve_chroma(
            question, query_vec, collection, chunks,
            bm25_index, top_k=12, candidate_k=12
        )

        # Stage 3: rerank, keep top 3
        top_results = rerank(question, candidates, top_k=3)

        # Stage 7: expand each top chunk with 1 neighbor on each side
        expanded = expand_chunks(top_results, chunks, neighbors=1)
        context_chunks = [chunk for _, chunk in expanded]

        top_index = top_results[0][0]
        score = top_results[0][2]

        # Stage 5: generate answer + trust check
        answer, confidence, verdict = generate_trusted_answer(
            question, context_chunks, generate_answer
        )

        print(f"\n(top chunk #{top_index}, reranker score {score:.4f})")
        print(f"Context: {len(context_chunks)} chunks used")
        print(f"[{confidence}] {answer}\n")


if __name__ == "__main__":
    main()