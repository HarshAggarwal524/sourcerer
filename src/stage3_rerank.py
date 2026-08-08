from sentence_transformers import CrossEncoder

# Loaded once at module level, same reasoning as your embedding model in Stage 0
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")


def rerank(query, candidates, top_k=5):
    """
    Re-scores a list of candidate chunks against the query using a cross-encoder,
    which reads the query and chunk together for more accurate relevance judgment.

    candidates: list of (index, chunk, score) triples — e.g. output of hybrid_retrieve()
    Returns: top_k (index, chunk, rerank_score) triples, sorted by new score.
    """
    if not candidates:
        return []

    # Cross-encoder needs [query, chunk_text] pairs as input
    pairs = [[query, chunk] for idx, chunk, score in candidates]
    rerank_scores = reranker.predict(pairs)

    # Attach new scores back to their original (index, chunk) pairs
    rescored = [
        (idx, chunk, float(rerank_scores[i]))
        for i, (idx, chunk, old_score) in enumerate(candidates)
    ]

    # Sort by the NEW cross-encoder score, descending
    rescored.sort(key=lambda x: x[2], reverse=True)

    return rescored[:top_k]


if __name__ == "__main__":
    from main import load_or_build
    from src.keyword_search import build_bm25_index
    from src.fusion import hybrid_retrieve

    pdf_path = "/Users/harshaggarwal/Projects_4/sourcerer/data/testpdf.pdf"
    chunks, embeddings = load_or_build(pdf_path)
    bm25 = build_bm25_index(chunks)

    query = "Robespierre guillotine"

    # Get a WIDE pool of candidates first (Stage 2's hybrid search)
    candidates = hybrid_retrieve(query, chunks, embeddings, bm25, top_k=30, candidate_k=30)

    # Then rerank down to a focused top 5
    reranked = rerank(query, candidates, top_k=5)

    print(f"Query: {query}\n")
    for idx, chunk, score in reranked:
        print(f"Chunk #{idx} (rerank score {score:.4f}):")
        print(chunk[:200])
        print()