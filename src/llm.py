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