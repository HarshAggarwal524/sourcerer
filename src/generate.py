import os
from dotenv import load_dotenv
from groq import Groq

from .parse import extract_text
from .chunking import chunk_text
from .embed import embed_chunks, embed_query
from .store import save_data, load_data
from .retrieve import retrieve

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_answer(question, context_chunk, model="groq/compound-mini"):
    """
    Sends the question + context to Groq's LLM and returns the answer text.
    """
    prompt = (
        "Answer the question using only the following context. "
        "If the answer isn't in the context, say so clearly instead of guessing.\n\n"
        f"Context: {context_chunk}\n\n"
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


if __name__ == "__main__":
    test_path = "/Users/harshaggarwal/Projects_4/sourcerer/sampel.pdf"  # replace with your real path

    data = load_data()
    if data is None:
        text = extract_text(test_path)
        chunks = chunk_text(text, chunk_size=300)
        embeddings = embed_chunks(chunks)
        save_data(chunks, embeddings)
    else:
        chunks = data["chunks"]
        embeddings = data["embeddings"]

    question = "What is phase 7 in the project?"
    query_vec = embed_query(question)

    results = retrieve(query_vec, embeddings, chunks, top_k=1)
    top_index, top_chunk, score = results[0]   # <-- now unpacking 3 values

    print(f"Retrieved chunk index {top_index} (score {score:.4f}):\n{top_chunk[:200]}\n")

    answer = generate_answer(question, top_chunk)
    print("Question asked:", question)
    print("Answer:")
    print(answer)