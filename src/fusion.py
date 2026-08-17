def reciprocal_rank_fusion(vector_results, keyword_results, k=60, top_k=5):
    """
    Merges two ranked lists (vector search + keyword search) into one,
    using Reciprocal Rank Fusion (RRF).

    vector_results: list of (index, chunk, score) from retrieve()
    keyword_results: list of (index, chunk, score) from keyword_search()
    k: RRF constant (60 is the standard default)
    top_k: how many final fused results to return
    """
    rrf_scores = {}   # index -> accumulated RRF score
    chunk_lookup = {} # index -> chunk text (so we can return it later)

    # Process vector search results
    for rank, (idx, chunk, score) in enumerate(vector_results, start=1):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (k + rank)
        chunk_lookup[idx] = chunk

    # Process keyword search results
    for rank, (idx, chunk, score) in enumerate(keyword_results, start=1):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (k + rank)
        chunk_lookup[idx] = chunk

    # Sort all chunks that appeared in EITHER list by their combined RRF score
    sorted_indices = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)

    fused_results = [
        (idx, chunk_lookup[idx], rrf_scores[idx])
        for idx in sorted_indices[:top_k]
    ]
    return fused_results

from src.embed import embed_query
from src.retrieve import retrieve
from src.keyword_search import keyword_search


def hybrid_retrieve(query, chunks, embeddings, bm25_index, top_k=5, candidate_k=10):
    """
    Combines vector search + BM25 keyword search using RRF.
    Returns (index, chunk, rrf_score) triples.
    """
    query_vec = embed_query(query)

    vector_results = retrieve(query_vec, embeddings, chunks, top_k=candidate_k)
    keyword_results = keyword_search(query, chunks, bm25_index, top_k=candidate_k)

    fused = reciprocal_rank_fusion(vector_results, keyword_results, k=60, top_k=top_k)
    return fused

def hybrid_retrieve_chroma(query, query_embedding, collection, chunks, bm25_index, top_k=5, candidate_k=12):
    """
    Same as hybrid_retrieve() but uses Chroma for vector search
    instead of numpy-based retrieve().
    """
    from src.stage6_store import chroma_retrieve
    from src.keyword_search import keyword_search

    # vector search via Chroma
    vector_results = chroma_retrieve(query_embedding, collection, top_k=candidate_k)

    # keyword search via BM25 (still uses raw chunks in memory)
    keyword_results = keyword_search(query, chunks, bm25_index, top_k=candidate_k)

    # fuse both ranked lists via RRF
    fused = reciprocal_rank_fusion(vector_results, keyword_results, k=60, top_k=top_k)
    return fused

def expand_chunks(top_results, all_chunks, neighbors=1):
    """
    Takes top reranked results and expands each with neighboring chunks.
    For each top chunk, includes `neighbors` chunks on each side.
    Deduplicates and preserves order (sorted by chunk index).
    
    top_results: list of (index, chunk, score) from rerank()
    all_chunks: full list of all chunks (from collection.get())
    neighbors: how many neighbors on each side (default 1)
    
    Returns: list of (index, chunk_text) tuples, sorted by index
    """
    total_chunks = len(all_chunks)
    seen_indices = set()
    expanded = []

    for idx, chunk, score in top_results:
        # get window around this chunk
        start = max(0, idx - neighbors)
        end = min(total_chunks - 1, idx + neighbors)

        for i in range(start, end + 1):
            if i not in seen_indices:
                seen_indices.add(i)
                expanded.append((i, all_chunks[i]))

    # sort by chunk index so context reads in document order
    expanded.sort(key=lambda x: x[0])
    return expanded

if __name__ == "__main__":
    from main import load_or_build
    from src.embed import embed_query
    from src.retrieve import retrieve
    from src.keyword_search import build_bm25_index, keyword_search

    pdf_path = "/Users/harshaggarwal/Projects_4/sourcerer/data/testpdf.pdf"
    chunks, embeddings = load_or_build(pdf_path)
    bm25 = build_bm25_index(chunks)

    query = "Robespierre guillotine"

    query_vec = embed_query(query)
    vector_results = retrieve(query_vec, embeddings, chunks, top_k=10)
    keyword_results = keyword_search(query, chunks, bm25, top_k=10)

    fused = reciprocal_rank_fusion(vector_results, keyword_results, k=60, top_k=5)

    print(f"Query: {query}\n")
    for idx, chunk, score in fused:
        print(f"Chunk #{idx} (RRF score {score:.5f}):")
        print(chunk[:200])
        print()