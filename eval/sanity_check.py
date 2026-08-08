import json
import random
from main import load_or_build

TESTSET_PATH = "eval/testset.json"
PDF_PATH = "/Users/harshaggarwal/Projects_4/sourcerer/data/testpdf.pdf"


def load_testset(path=TESTSET_PATH):
    with open(path, "r") as f:
        return json.load(f)


def fake_random_retrieve(num_chunks, top_k=3):
    """Returns top_k RANDOM indices, ignoring the actual question entirely."""
    return random.sample(range(num_chunks), top_k)


def evaluate_recall_at_k_random(testset, num_chunks, k=3):
    hits = 0
    for item in testset:
        correct_index = item["correct_chunk_index"]
        retrieved_indices = fake_random_retrieve(num_chunks, top_k=k)
        if correct_index in retrieved_indices:
            hits += 1
    return hits / len(testset)


if __name__ == "__main__":
    testset = load_testset()
    chunks, embeddings = load_or_build(PDF_PATH)

    recall = evaluate_recall_at_k_random(testset, len(chunks), k=3)
    print(f"Random-baseline Recall@3: {recall:.4f}")
    print(f"Expected by pure chance: ~{3/len(chunks):.4f}")