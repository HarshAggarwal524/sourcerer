import json
from main import load_or_build
from src.retrieve import retrieve
from src.keyword_search import build_bm25_index
from src.fusion import hybrid_retrieve
from src.stage3_rerank import rerank
from src.stage4_hyde import hyde_retrieve
from src.embed import embed_query
from src.generate import generate_answer
from src.stage5_trust import check_grounding

PDF_PATH = "data/testpdf.pdf"
RERANK_POOL = 12


def load_testset(path):
    with open(path, "r") as f:
        return json.load(f)


def get_results(mode, question, chunks, embeddings, bm25_index, k):
    """
    Central retrieval dispatcher.
    mode: "vector" | "hybrid" | "hybrid_rerank" | "hyde" | "hyde_rerank"
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


def evaluate_grounding_rate(testset, chunks, embeddings, bm25_index, mode):
    """
    For each question in the test set:
    1. Retrieve the top 1 chunk using the given pipeline mode.
    2. Generate an answer from that chunk.
    3. Ask the LLM judge whether the answer is supported by the chunk.
    Returns the fraction of answers judged SUPPORTED.

    Note: this makes one Groq API call per question (generation) plus
    one more (grounding check) — more expensive than Recall/MRR scoring.
    Run on a subset if cost/time is a concern.
    """
    supported_count = 0
    for item in testset:
        question = item["question"]

        # retrieve top 1 chunk only — grounding is about the single answer given
        results = get_results(mode, question, chunks, embeddings, bm25_index, k=1)
        _, top_chunk, _ = results[0]

        # generate answer from that chunk
        answer = generate_answer(question, top_chunk)
        if not answer:
            continue  # skip if generation failed

        # judge whether it's supported
        verdict = check_grounding(question, top_chunk, answer)
        if verdict == "SUPPORTED":
            supported_count += 1

    return supported_count / len(testset)


def run_eval(testset_path, chunks, embeddings, bm25_index, mode, label, include_grounding=False):
    testset = load_testset(testset_path)
    recall_3 = evaluate_recall_at_k(testset, chunks, embeddings, bm25_index, mode, k=3)
    mrr = evaluate_mrr(testset, chunks, embeddings, bm25_index, mode, top_k=10)

    print(f"--- {label} ({len(testset)} questions) ---")
    print(f"Recall@3:  {recall_3:.4f}")
    print(f"MRR:       {mrr:.4f}")

    grounding = None
    if include_grounding:
        grounding = evaluate_grounding_rate(testset, chunks, embeddings, bm25_index, mode)
        print(f"Grounding: {grounding:.4f}")

    print()
    return {"label": label, "recall@3": recall_3, "mrr": mrr, "grounding": grounding}


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
        ("vector",        "Stage 1 (Vector only)"),
        ("hybrid",        "Stage 2 (Hybrid BM25+RRF)"),
        ("hybrid_rerank", "Stage 3 (Hybrid+Rerank)"),
        ("hyde",          "Stage 4a (HyDE only)"),
        ("hyde_rerank",   "Stage 4b (HyDE+Rerank)"),
    ]

    all_results = []

'''  # ── Recall + MRR for all stages ──────────────────────────────────────────
    for mode, stage_label in stages:
        print(f"========== {stage_label} ==========\n")
        for path, set_label in testsets:
            label = f"{set_label} — {stage_label}"
            all_results.append(
                run_eval(path, chunks, embeddings, bm25_index, mode, label, include_grounding=False)
            ) '''

# ── Grounding rate: Stage 3 and Stage 4b ─────────────────────────────────
print("========== STAGE 5: Grounding Rate ==========\n")
grounding_results = []

print("--- Stage 3 pipeline (hybrid + rerank, no HyDE) ---\n")
for path, set_label in testsets:
    label = f"{set_label} — Stage 3 (Grounding)"
    grounding_results.append(
        run_eval(path, chunks, embeddings, bm25_index, "hybrid_rerank", label, include_grounding=True)
    )

print("--- Stage 4b pipeline (HyDE + rerank) ---\n")
for path, set_label in testsets:
    label = f"{set_label} — Stage 4b (Grounding)"
    grounding_results.append(
        run_eval(path, chunks, embeddings, bm25_index, "hyde_rerank", label, include_grounding=True)
    )

# ── Summary table ─────────────────────────────────────────────────────────
print("\n========== FULL SUMMARY (Recall + MRR) ==========")
print(f"{'Label':<50} {'Recall@3':<12} {'MRR':<12}")
for r in all_results:
    print(f"{r['label']:<50} {r['recall@3']:<12.4f} {r['mrr']:<12.4f}")

print("\n========== STAGE 5 SUMMARY (Grounding Rate) ==========")
print(f"{'Label':<50} {'Grounding':<12}")
for r in grounding_results:
    print(f"{r['label']:<50} {r['grounding']:<12.4f}")