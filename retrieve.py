import numpy as np
from sentence_transformers import util
from parse import extract_text
from chunking import chunk_text
from embed import embed_chunks, embed_query
from store import save_data, load_data

def retrieve(query_embedding, chunk_embeddings, chunks, top_k=1):
    """
    Returns the top_k chunks most similar to the query embedding,
    along with their similarity scores.
    """
    # cos_sim expects 2D inputs; util.cos_sim handles the broadcasting for us
    scores = util.cos_sim(query_embedding, chunk_embeddings)[0]  # shape: (num_chunks,)
    scores = scores.numpy()

    # indices of the top_k highest scores, descending order
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = [(chunks[i], float(scores[i])) for i in top_indices]
    return results


if __name__ == "__main__":
    test_path = "/Users/harshaggarwal/Projects_4/sourcerer/sampel.pdf"  # replace with your real path

    data = load_data()
    if data is None:
        text = extract_text(test_path)
        chunks = chunk_text(text, chunk_size=300)
        embeddings = embed_chunks(chunks)
        save_data(chunks, embeddings)
    else:
        chunks = data["chunks"]
        embeddings = data["embeddings"]

    question = " phase 7"  # replace with a real question
    query_vec = embed_query(question)
    
    print("Question asked:", question)
    print("Query vector (first 5 values):", query_vec[:5])

    results = retrieve(query_vec, embeddings, chunks, top_k=len(chunks))
    for chunk, score in results:
        print(f"Score: {score:.4f} | Chunk preview: {chunk[:80]}")