import os
import hashlib
from src.parse import extract_text
from src.chunking import chunk_text
from src.embed import embed_chunks, embed_query
from src.keyword_search import build_bm25_index
from src.fusion import hybrid_retrieve
from src.stage3_rerank import rerank
from src.stage5_trust import generate_trusted_answer
from src.generate import generate_answer
from src.stage6_store import ingest_document, load_collection, chroma_retrieve
from src.fusion import hybrid_retrieve_chroma

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
    This is the single entry point both main.py and app.py should use.
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

    # BM25 still needs raw chunks — fetch them from Chroma
    all_data = collection.get(include=["documents", "metadatas"])
    chunks = all_data["documents"]
    chunk_embeddings = None  # not needed — Chroma handles vector search now

    bm25_index = build_bm25_index(chunks)

    print(f"\nReady. Loaded {len(chunks)} chunks.")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("Ask a question: ")
        if question.strip().lower() == "exit":
            print("Goodbye.")
            break

        query_vec = embed_query(question)

        # Stage 6: use Chroma for initial vector retrieval
        candidates = hybrid_retrieve_chroma(question, query_vec, collection, chunks, bm25_index, top_k=12, candidate_k=12)

        # Stage 3: rerank the candidates
        results = rerank(question, candidates, top_k=1)
        top_index, top_chunk, score = results[0]

        # Stage 5: generate + trust check
        answer, confidence, verdict = generate_trusted_answer(
            question, top_chunk, generate_answer
        )

        print(f"\n(chunk #{top_index}, reranker score {score:.4f})")
        print(f"[{confidence}] {answer}\n")


if __name__ == "__main__":
    main()