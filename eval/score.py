import json
from main import load_or_build
from src.keyword_search import build_bm25_index
from src.fusion import hybrid_retrieve
from src.stage3_rerank import rerank
from src.stage4_hyde import hyde_retrieve

PDF_PATH = "data/testpdf.pdf"


def load_testset(path):
    with open(path, "r") as f:
        return json.load(f)


def get_results(question, chunks, embeddings, bm25_index, k, use_hyde=False, use_rerank=False):
    """
    Central retrieval dispatcher — picks the right pipeline based on flags.
    """
    candidate_pool = 12 if use_rerank else k

    if use_hyde:
        candidates = hyde_retrieve(question, chunks, embeddings, bm25_index, top_k=candidate_pool, candidate_k=candidate_pool)
    else:
        candidates = hybrid_retrieve(question, chunks, embeddings, bm25_index, top_k=candidate_pool, candidate_k=candidate_pool)

    if use_rerank:
        return rerank(question, candidates, top_k=k)
    else:
        return candidates[:k]


def evaluate_recall_at_k(testset, chunks, embeddings, bm25_index, k=3, use_hyde=False, use_rerank=False):
    hits = 0
    for item in testset:
        question = item["question"]
        correct_index = item["correct_chunk_index"]

        results = get_results(question, chunks, embeddings, bm25_index, k, use_hyde, use_rerank)
        retrieved_indices = [idx for idx, chunk, score in results]

        if correct_index in retrieved_indices:
            hits += 1

    return hits / len(testset)


def evaluate_mrr(testset, chunks, embeddings, bm25_index, top_k=10, use_hyde=False, use_rerank=False):
    reciprocal_ranks = []

    for item in testset:
        question = item["question"]
        correct_index = item["correct_chunk_index"]

        results = get_results(question, chunks, embeddings, bm25_index, top_k, use_hyde, use_rerank)
        retrieved_indices = [idx for idx, chunk, score in results]

        if correct_index in retrieved_indices:
            rank = retrieved_indices.index(correct_index) + 1
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0)

    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def run_eval(testset_path, chunks, embeddings, bm25_index, label, use_hyde=False, use_rerank=False):
    testset = load_testset(testset_path)
    recall_3 = evaluate_recall_at_k(testset, chunks, embeddings, bm25_index, k=3, use_hyde=use_hyde, use_rerank=use_rerank)
    mrr = evaluate_mrr(testset, chunks, embeddings, bm25_index, top_k=10, use_hyde=use_hyde, use_rerank=use_rerank)

    print(f"--- {label} ({len(testset)} questions) ---")
    print(f"Recall@3: {recall_3:.4f}")
    print(f"MRR: {mrr:.4f}")
    print()

    return {"label": label, "recall@3": recall_3, "mrr": mrr}


if __name__ == "__main__":
    chunks, embeddings = load_or_build(PDF_PATH)
    bm25_index = build_bm25_index(chunks)
    print(f"Loaded {len(chunks)} chunks.\n")

    testsets = [
        ("eval/testset_easy.json", "Easy set"),
        ("eval/testset_hard.json", "Hard set"),
        ("eval/testset_hardest.json", "Hardest set"),
        ("eval/testset_ambiguous_experimental.json", "Ambiguous Experimental set"),
    ]

    print("========== STAGE 4a: HyDE alone (compare to Stage 2) ==========\n")
    for path, label in testsets:
        run_eval(path, chunks, embeddings, bm25_index, f"{label} (HyDE only)", use_hyde=True, use_rerank=False)

    print("========== STAGE 4b: HyDE + Reranking (compare to Stage 3) ==========\n")
    for path, label in testsets:
        run_eval(path, chunks, embeddings, bm25_index, f"{label} (HyDE + Rerank)", use_hyde=True, use_rerank=True)