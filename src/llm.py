import os
from dotenv import load_dotenv

from src.config import LLM_PROVIDER, LLM_MODEL

load_dotenv()


def generate_llm(prompt, model=LLM_MODEL):
    """
    Generic LLM interface.

    Provider and model are controlled entirely through config.py.
    """

    if LLM_PROVIDER == "gemini":
        from google import genai

        client = genai.Client(
            api_key=os.getenv("GEMINI_API_KEY")
        )

        response = client.models.generate_content(
            model=model,
            contents=prompt,
        )

        return response.text

    elif LLM_PROVIDER == "groq":
        from groq import Groq

        client = Groq(
            api_key=os.getenv("GROQ_API_KEY")
        )

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
        )

        return response.choices[0].message.content

    else:
        raise ValueError(
            f"Unsupported LLM provider: {LLM_PROVIDER}"
        )
        
def stream_llm(prompt, model=LLM_MODEL):
    """
    Streams the LLM response token by token.
    Yields text chunks as they arrive.
    Only implemented for Gemini — Groq fallback yields full response at once.
    """
    if LLM_PROVIDER == "gemini":
        from google import genai

        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

        for chunk in client.models.generate_content_stream(
            model=model,
            contents=prompt,
        ):
            if chunk.text:
                yield chunk.text

    elif LLM_PROVIDER == "groq":
        # Groq streaming exists but for simplicity, yield full response at once
        result = generate_llm(prompt, model)
        if result:
            yield result
            
def rewrite_query(question, history, model=LLM_MODEL):
    """
    Rewrites a question using conversation history to resolve
    ambiguous references like "he", "it", "that", "them".
    Returns a standalone question with full context filled in.
    If no rewrite is needed, returns the original question unchanged.
    """
    if not history:
        return question

    history_text = "\n".join(
        f"Q: {q}\nA: {a}" for q, a in history
    )

    prompt = (
        "You are a query rewriter. Given a conversation history and a follow-up question, "
        "rewrite the follow-up question to be fully self-contained — "
        "resolving any pronouns or references using the history.\n"
        "If the question is already self-contained, return it unchanged.\n"
        "Return ONLY the rewritten question. No explanation, no preamble.\n\n"
        f"Conversation history:\n{history_text}\n\n"
        f"Follow-up question: {question}"
    )

    try:
        return generate_llm(prompt, model).strip()
    except Exception as e:
        print(f"[llm.py] Query rewrite failed: {e}")
        return question