import os
import pickle
import hashlib

from parse import extract_text
from chunking import chunk_text
from embed import embed_chunks, embed_query
from retrieve import retrieve
from generate import generate_answer

MODEL_NAME = "all-MiniLM-L6-v2"


def get_store_path(pdf_path, model_name):
    """
    Builds a cache filename based on a hash of (PDF content + model name).
    Same PDF + same model -> same hash -> cache reused.
    Same PDF + different model -> different hash -> treated as new.
    """
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    combined = pdf_bytes + model_name.encode("utf-8")
    hash_value = hashlib.sha256(combined).hexdigest()[:16]

    return f"store_{hash_value}.pkl"


def load_or_build(pdf_path, model_name=MODEL_NAME):
    """
    Loads cached chunks + embeddings if they exist for this (PDF, model) pair.
    Otherwise runs parse -> chunk -> embed and saves the result.
    """
    store_path = get_store_path(pdf_path, model_name)

    if os.path.exists(store_path):
        print(f"Found existing cache: {store_path}")
        with open(store_path, "rb") as f:
            data = pickle.load(f)
        return data["chunks"], data["embeddings"]

    print("No cache found for this PDF + model — processing for the first time...")
    text = extract_text(pdf_path)
    if not text:
        raise ValueError("Could not extract any text from this PDF.")

    chunks = chunk_text(text, chunk_size=300)
    embeddings = embed_chunks(chunks)

    with open(store_path, "wb") as f:
        pickle.dump(
            {"chunks": chunks, "embeddings": embeddings, "model_name": model_name},
            f,
        )
    print(f"Saved new cache: {store_path}")

    return chunks, embeddings


def main():
    pdf_path = "/Users/harshaggarwal/Projects_4/sourcerer/sampel.pdf"  # replace with your real path

    try:
        chunks, embeddings = load_or_build(pdf_path)
    except Exception as e:
        print(f"Failed to load document: {e}")
        return

    print(f"\nReady. Loaded {len(chunks)} chunks from {pdf_path}.")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("Ask a question: ")
        if question.strip().lower() == "exit":
            print("Goodbye.")
            break

        query_vec = embed_query(question)
        results = retrieve(query_vec, embeddings, chunks, top_k=1)
        top_chunk, score = results[0]

        answer = generate_answer(question, top_chunk)

        print(f"\n(retrieved chunk, score {score:.4f})")
        print(f"Answer: {answer}\n")


if __name__ == "__main__":
    main()