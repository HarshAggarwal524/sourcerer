import os
from dotenv import load_dotenv
from groq import Groq

from src.embed import embed_query
from src.keyword_search import build_bm25_index
from src.fusion import hybrid_retrieve

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def generate_hypothetical_answer(question, model="groq/compound-mini"):
    """
    Asks the LLM to write a short, plausible passage that would answer
    the question, styled like an excerpt from the source document.
    Does not need to be factually correct — just stylistically similar
    to real document prose, so embedding it matches document chunks better.
    """
    prompt = (
        "Write a short paragraph (2-4 sentences) that would plausibly answer "
        "the following question, written in the style of a factual historical "
        "document excerpt. It does not need to be accurate — just write it as "
        "if it could be a passage from a history textbook.\n\n"
        f"Question: {question}"
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"[stage4_hyde.py] LLM call failed: {e}")
        return None


def hyde_retrieve(question, chunks, embeddings, bm25_index, top_k=5, candidate_k=10):
    """
    HyDE retrieval: generates a hypothetical answer to the question,
    embeds THAT instead of the raw question, then runs hybrid retrieval.
    Returns (index, chunk, score) triples, same shape as other retrieve functions.
    """
    hypothetical = generate_hypothetical_answer(question)

    if not hypothetical:
        # Fallback: if generation fails, just use the original question
        hypothetical = question

    print(f"[HyDE] Original question: {question}")
    print(f"[HyDE] Hypothetical passage: {hypothetical}\n")

    query_vec = embed_query(hypothetical)

    # Reuse hybrid_retrieve's internals, but we already have our own query vector,
    # so we call retrieve() + keyword_search() + fusion directly instead of
    # hybrid_retrieve() (which re-embeds the raw question internally)
    from src.retrieve import retrieve
    from src.keyword_search import keyword_search
    from src.fusion import reciprocal_rank_fusion

    vector_results = retrieve(query_vec, embeddings, chunks, top_k=candidate_k)
    keyword_results = keyword_search(hypothetical, chunks, bm25_index, top_k=candidate_k)

    fused = reciprocal_rank_fusion(vector_results, keyword_results, k=60, top_k=top_k)
    return fused


if __name__ == "__main__":
    from main import load_or_build

    pdf_path = "data/testpdf.pdf"
    chunks, embeddings = load_or_build(pdf_path)
    bm25 = build_bm25_index(chunks)

    question = "when did Louis XVI ascend the throne"

    results = hyde_retrieve(question, chunks, embeddings, bm25, top_k=5)

    for idx, chunk, score in results:
        print(f"Chunk #{idx} (score {score:.5f}):")
        print(chunk[:200])
        print()