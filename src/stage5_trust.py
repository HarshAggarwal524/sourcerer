import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def check_grounding(question, context_chunk, answer, model="groq/compound-mini"):
    """
    LLM-as-judge: checks whether the generated answer is fully supported
    by the source chunk, with no information added from outside it.
    Returns "SUPPORTED" or "NOT_SUPPORTED".
    """
    prompt = (
        "You are a strict fact-checker. Your job is to determine whether an answer "
        "is fully supported by a given source passage.\n\n"
        "Rules:\n"
        "- Reply SUPPORTED only if every specific claim in the answer is directly "
        "stated in the source passage.\n"
        "- Reply NOT_SUPPORTED if the answer contains any information, inference, "
        "or detail not explicitly present in the source passage.\n"
        "- Reply NOT_SUPPORTED if the answer says there is insufficient information "
        "or that the question cannot be answered.\n"
        "- Reply with ONLY the word SUPPORTED or NOT_SUPPORTED. Nothing else.\n\n"
        f"Source passage:\n{context_chunk}\n\n"
        f"Question:\n{question}\n\n"
        f"Answer:\n{answer}"
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,  # deterministic — this is a binary classification task
        )
        verdict = response.choices[0].message.content.strip().upper()
        if verdict not in ("SUPPORTED", "NOT_SUPPORTED"):
            # if the model doesn't follow instructions, default to cautious
            return "NOT_SUPPORTED"
        return verdict
    except Exception as e:
        print(f"[stage5_trust.py] Grounding check failed: {e}")
        return "NOT_SUPPORTED"


def generate_trusted_answer(question, context_chunks, generate_fn, model="groq/compound-mini"):
    """
    Generates an answer and then verifies it against the source chunk.
    Returns: (final_answer, confidence_tag, verdict)
    - confidence_tag: "HIGH CONFIDENCE" or "LOW CONFIDENCE"
    - verdict: "SUPPORTED" or "NOT_SUPPORTED"
    """
    answer = generate_fn(question, context_chunk)

    if not answer:
        return "Could not generate an answer.", "LOW CONFIDENCE", "NOT_SUPPORTED"

    verdict = check_grounding(question, context_chunk, answer)

    if verdict == "SUPPORTED":
        return answer, "HIGH CONFIDENCE", verdict
    else:
        return (
            "I could not find sufficient information in the source text to answer this question confidently.",
            "LOW CONFIDENCE",
            verdict,
        )


if __name__ == "__main__":
    from main import load_or_build
    from src.keyword_search import build_bm25_index
    from src.fusion import hybrid_retrieve
    from src.stage3_rerank import rerank
    from src.generate import generate_answer

    pdf_path = "data/testpdf.pdf"
    chunks, embeddings = load_or_build(pdf_path)
    bm25 = build_bm25_index(chunks)

    # Test 1: a question the document CAN answer
    question_good = "What was the name of the cooperative community Robert Owen sought to build?"
    candidates = hybrid_retrieve(question_good, chunks, embeddings, bm25, top_k=12, candidate_k=12)
    results = rerank(question_good, candidates, top_k=1)
    idx, top_chunk, score = results[0]

    answer, confidence, verdict = generate_trusted_answer(question_good, top_chunk, generate_answer)
    print(f"Q: {question_good}")
    print(f"Chunk #{idx} (score {score:.4f})")
    print(f"Answer: {answer}")
    print(f"Tag: {confidence} ({verdict})\n")

    # Test 2: a question the document CANNOT answer
    question_bad = "What is the capital of Australia?"
    candidates = hybrid_retrieve(question_bad, chunks, embeddings, bm25, top_k=12, candidate_k=12)
    results = rerank(question_bad, candidates, top_k=1)
    idx, top_chunk, score = results[0]

    answer, confidence, verdict = generate_trusted_answer(question_bad, top_chunk, generate_answer)
    print(f"Q: {question_bad}")
    print(f"Chunk #{idx} (score {score:.4f})")
    print(f"Answer: {answer}")
    print(f"Tag: {confidence} ({verdict})")