import os
import pickle
import hashlib

from src.parse import extract_text
from src.chunking import chunk_text
from src.embed import embed_chunks, embed_query
from src.retrieve import retrieve
from src.generate import generate_answer
from src.stage5_trust import generate_trusted_answer



MODEL_NAME = "all-MiniLM-L6-v2"


def get_store_path(pdf_path, model_name):
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    combined = pdf_bytes + model_name.encode("utf-8")
    hash_value = hashlib.sha256(combined).hexdigest()[:16]
    return f"cache/store_{hash_value}.pkl"


def load_or_build(pdf_path, model_name=MODEL_NAME):
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
    pdf_path = "/Users/harshaggarwal/Projects_4/sourcerer/data/testpdf.pdf"

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
        top_index, top_chunk, score = results[0]   # <-- unpacking 3 values now

        answer, confidence, verdict = generate_trusted_answer(question, top_chunk, generate_answer)
        print(f"\n(retrieved chunk #{top_index}, score {score:.4f})")
        print(f"[{confidence}] {answer}\n")


if __name__ == "__main__":
    main()