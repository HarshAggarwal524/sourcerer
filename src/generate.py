from src.config import LLM_MODEL
from src.llm import generate_llm


def generate_answer(question, context_chunks, model=LLM_MODEL):
    """
    Sends the question + context to the configured LLM
    and returns the answer text.
    """

    if isinstance(context_chunks, list):
        context = "\n\n---\n\n".join(context_chunks)
    else:
        context = context_chunks

    prompt = (
        "Answer the question using only the following context. "
        "If the answer isn't in the context, say so clearly instead of guessing.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )

    try:
        return generate_llm(prompt, model)

    except Exception as e:
        print(f"[generate.py] LLM call failed: {e}")
        return None