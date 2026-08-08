import json
from main import load_or_build
from src.keyword_search import build_bm25_index
from src.fusion import hybrid_retrieve

PDF_PATH = "/Users/harshaggarwal/Projects_4/sourcerer/data/testpdf.pdf"


def load_testset(path):
    with open(path, "r") as f:
        return json.load(f)


def evaluate_recall_at_k(testset, chunks, embeddings, bm25_index, k=3):
    hits = 0
    for item in testset:
        question = item["question"]
        correct_index = item["correct_chunk_index"]

        results = hybrid_retrieve(question, chunks, embeddings, bm25_index, top_k=k)
        retrieved_indices = [idx for idx, chunk, score in results]

        if correct_index in retrieved_indices:
            hits += 1

    return hits / len(testset)


def evaluate_mrr(testset, chunks, embeddings, bm25_index, top_k=10):
    reciprocal_ranks = []

    for item in testset:
        question = item["question"]
        correct_index = item["correct_chunk_index"]

        results = hybrid_retrieve(question, chunks, embeddings, bm25_index, top_k=top_k)
        retrieved_indices = [idx for idx, chunk, score in results]

        if correct_index in retrieved_indices:
            rank = retrieved_indices.index(correct_index) + 1
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0)

    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def run_eval(testset_path, chunks, embeddings, bm25_index, label):
    testset = load_testset(testset_path)
    recall_3 = evaluate_recall_at_k(testset, chunks, embeddings, bm25_index, k=3)
    mrr = evaluate_mrr(testset, chunks, embeddings, bm25_index, top_k=10)

    print(f"--- {label} ({len(testset)} questions) ---")
    print(f"Recall@3: {recall_3:.4f}")
    print(f"MRR: {mrr:.4f}")
    print()

    return {"label": label, "recall@3": recall_3, "mrr": mrr}


if __name__ == "__main__":
    chunks, embeddings = load_or_build(PDF_PATH)
    bm25_index = build_bm25_index(chunks)
    print(f"Loaded {len(chunks)} chunks.\n")

    results = []
    results.append(run_eval("eval/testset_easy.json", chunks, embeddings, bm25_index, "Easy set"))
    results.append(run_eval("eval/testset_hard.json", chunks, embeddings, bm25_index, "Hard set"))
    results.append(run_eval("eval/testset_hardest.json", chunks, embeddings, bm25_index, "Hardest set"))
    results.append(run_eval("eval/testset_ambiguous_experimental.json", chunks, embeddings, bm25_index, "Ambiguous Experimental set"))