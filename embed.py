from sentence_transformers import SentenceTransformer
from parse import extract_text
from chunking import chunk_text

# Load once at module level so it's not reloaded on every call
model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_chunks(chunks):
    """
    Takes a list of text chunks, returns a numpy array of shape (N, 384).
    """
    if not chunks:
        return None
    return model.encode(chunks)

def embed_query(query):
    """
    Takes a single query string, returns its embedding as a numpy array of shape (384,).
    """
    return model.encode(query)


if __name__ == "__main__":
    test_path = "/Users/harshaggarwal/Projects_4/sourcerer/sampel.pdf"  # replace with your real path
    text = extract_text(test_path)

    if text:
        chunks = chunk_text(text, chunk_size=300)
        embeddings = embed_chunks(chunks)
        print(f"Number of chunks: {len(chunks)}")
        print(f"Embeddings shape: {embeddings.shape}")

        query_vec = embed_query("What is this document about?")
        print(f"Query embedding shape: {query_vec.shape}")
    else:
        print("No text extracted, nothing to embed.")