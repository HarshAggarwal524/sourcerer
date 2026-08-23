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