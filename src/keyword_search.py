from rank_bm25 import BM25Okapi
import re

def tokenize(text):
    """
    Lowercase and strip punctuation, then split into words.
    Keeps BM25 matching case-insensitive and punctuation-insensitive.
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    return text.split()


def build_bm25_index(chunks):
    """
    Builds a BM25 index over a list of text chunks.
    Returns the BM25 object, which you'll reuse for every query.
    """
    tokenized_chunks = [tokenize(chunk) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    return bm25


def keyword_search(query, chunks, bm25_index, top_k=3):
    """
    Returns the top_k chunks most relevant to the query by BM25 keyword score,
    as (index, chunk_text, score) triples — same shape as vector retrieve().
    """
    tokenized_query = tokenize(query)
    scores = bm25_index.get_scores(tokenized_query)  # one score per chunk

    # Get indices of top_k highest scores, descending
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    results = [(i, chunks[i], float(scores[i])) for i in top_indices if scores[i] > 0]
    return results


if __name__ == "__main__":
    from main import load_or_build

    pdf_path = "/Users/harshaggarwal/Projects_4/sourcerer/data/testpdf.pdf"
    chunks, embeddings = load_or_build(pdf_path)

    bm25 = build_bm25_index(chunks)

    query = "Robespierre guillotine"  # distinctive keywords to test
    results = keyword_search(query, chunks, bm25, top_k=3)

    print(f"Query: {query}\n")
    for idx, chunk, score in results:
        print(f"Chunk #{idx} (score {score:.4f}):")
        print(chunk[:200])
        print()