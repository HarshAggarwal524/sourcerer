import os
from dotenv import load_dotenv
from groq import Groq
from src.config import LLM_MODEL

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def generate_answer(question, context_chunks, model=LLM_MODEL):
    """
    Sends the question + context to Groq's LLM and returns the answer text.
    context_chunks: either a single string or a list of strings.
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
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"[generate.py] LLM call failed: {e}")
        return None
