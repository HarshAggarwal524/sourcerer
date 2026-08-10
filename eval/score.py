import json
from main import load_or_build
from src.retrieve import retrieve
from src.keyword_search import build_bm25_index
from src.fusion import hybrid_retrieve
from src.stage3_rerank import rerank
from src.stage4_hyde import hyde_retrieve
from src.embed import embed_query

PDF_PATH = "data/testpdf.pdf"
RERANK_POOL = 12  # fair, consistent candidate pool for any rerank variant


def load_testset(path):
    with open(path, "r") as f:
        return json.load(f)


def get_results(mode, question, chunks, embeddings, bm25_index, k):
    """
    mode: "vector" (Stage 1), "hybrid" (Stage 2), "hybrid_rerank" (Stage 3),
          "hyde" (Stage 4a), "hyde_rerank" (Stage 4b)
    """
    if mode == "vector":
        query_vec = embed_query(question)
        return retrieve(query_vec, embeddings, chunks, top_k=k)

    elif mode == "hybrid":
        return hybrid_retrieve(question, chunks, embeddings, bm25_index, top_k=k, candidate_k=k)

    elif mode == "hybrid_rerank":
        candidates = hybrid_retrieve(question, chunks, embeddings, bm25_index, top_k=RERANK_POOL, candidate_k=RERANK_POOL)
        return rerank(question, candidates, top_k=k)

    elif mode == "hyde":
        return hyde_retrieve(question, chunks, embeddings, bm25_index, top_k=k, candidate_k=k)

    elif mode == "hyde_rerank":
        candidates = hyde_retrieve(question, chunks, embeddings, bm25_index, top_k=RERANK_POOL, candidate_k=RERANK_POOL)
        return rerank(question, candidates, top_k=k)

    else:
        raise ValueError(f"Unknown mode: {mode}")


def evaluate_recall_at_k(testset, chunks, embeddings, bm25_index, mode, k=3):
    hits = 0
    for item in testset:
        results = get_results(mode, item["question"], chunks, embeddings, bm25_index, k)
        retrieved_indices = [idx for idx, chunk, score in results]
        if item["correct_chunk_index"] in retrieved_indices:
            hits += 1
    return hits / len(testset)


def evaluate_mrr(testset, chunks, embeddings, bm25_index, mode, top_k=10):
    reciprocal_ranks = []
    for item in testset:
        results = get_results(mode, item["question"], chunks, embeddings, bm25_index, top_k)
        retrieved_indices = [idx for idx, chunk, score in results]
        if item["correct_chunk_index"] in retrieved_indices:
            rank = retrieved_indices.index(item["correct_chunk_index"]) + 1
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0)
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def run_eval(testset_path, chunks, embeddings, bm25_index, mode, label):
    testset = load_testset(testset_path)
    recall_3 = evaluate_recall_at_k(testset, chunks, embeddings, bm25_index, mode, k=3)
    mrr = evaluate_mrr(testset, chunks, embeddings, bm25_index, mode, top_k=10)
    print(f"--- {label} ({len(testset)} questions) ---")
    print(f"Recall@3: {recall_3:.4f}")
    print(f"MRR: {mrr:.4f}\n")
    return {"label": label, "recall@3": recall_3, "mrr": mrr}


if __name__ == "__main__":
    chunks, embeddings = load_or_build(PDF_PATH)
    bm25_index = build_bm25_index(chunks)
    print(f"Loaded {len(chunks)} chunks.\n")

    testsets = [
        ("eval/testset_easy.json", "Easy"),
        ("eval/testset_hard.json", "Hard"),
        ("eval/testset_hardest.json", "Hardest"),
        ("eval/testset_ambiguous_experimental.json", "Ambiguous"),
    ]

    stages = [
        ("vector", "Stage 1 (Vector only)"),
        ("hybrid", "Stage 2 (Hybrid BM25+RRF)"),
        ("hybrid_rerank", "Stage 3 (Hybrid+Rerank)"),
        ("hyde", "Stage 4a (HyDE only)"),
        ("hyde_rerank", "Stage 4b (HyDE+Rerank)"),
    ]

    all_results = []
    for mode, stage_label in stages:
        print(f"========== {stage_label} ==========\n")
        for path, set_label in testsets:
            label = f"{set_label} — {stage_label}"
            all_results.append(run_eval(path, chunks, embeddings, bm25_index, mode, label))

    print("\n========== FULL SUMMARY ==========")
    print(f"{'Label':<50} {'Recall@3':<12} {'MRR':<12}")
    for r in all_results:
        print(f"{r['label']:<50} {r['recall@3']:<12.4f} {r['mrr']:<12.4f}")