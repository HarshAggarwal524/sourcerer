import hashlib
import chromadb
from chromadb.config import Settings
from src.config import EMBEDDING_MODEL

# Chroma persists to this local directory automatically
CHROMA_PATH = "chroma_db"


def get_collection_name(pdf_path, model_name):
    """
    Generates a unique collection name based on PDF content + model name.
    Same logic as your old cache key — same PDF + same model = same collection.
    Different PDF or different model = different collection, no stale data.
    """
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    combined = pdf_bytes + model_name.encode("utf-8")
    hash_value = hashlib.sha256(combined).hexdigest()[:16]
    return f"sourcerer_{hash_value}"


def get_chroma_client():
    """
    Returns a persistent Chroma client that saves data to CHROMA_PATH on disk.
    Persistent = survives script restarts, same as your old pickle cache.
    """
    return chromadb.PersistentClient(path=CHROMA_PATH)


def ingest_document(pdf_path, chunks, embeddings, model_name):
    """
    Stores chunks + their embeddings into a Chroma collection.
    If a collection for this PDF+model already exists, skips ingestion.
    Returns the collection.
    """
    client = get_chroma_client()
    collection_name = get_collection_name(pdf_path, model_name)

    # check if already ingested
    existing = [c.name for c in client.list_collections()]
    if collection_name in existing:
        print(f"[stage6_store] Collection already exists: {collection_name}")
        return client.get_collection(collection_name)

    print(f"[stage6_store] Creating new collection: {collection_name}")
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}  # use cosine similarity for queries
    )

    # Chroma needs string IDs for each document
    ids = [str(i) for i in range(len(chunks))]

    collection.add(
        ids=ids,
        documents=chunks,                          # raw text
        embeddings=[e.tolist() for e in embeddings],  # numpy → list
        metadatas=[{"chunk_index": i} for i in range(len(chunks))]
    )

    print(f"[stage6_store] Stored {len(chunks)} chunks.")
    return collection


def load_collection(pdf_path, model_name):
    """
    Loads an existing Chroma collection for this PDF+model pair.
    Returns None if not found (document hasn't been ingested yet).
    """
    client = get_chroma_client()
    collection_name = get_collection_name(pdf_path, model_name)

    existing = [c.name for c in client.list_collections()]
    if collection_name not in existing:
        return None

    return client.get_collection(collection_name)


def chroma_retrieve(query_embedding, collection, top_k=3):
    """
    Queries Chroma directly for the top_k most similar chunks.
    Chroma handles cosine similarity search internally — no numpy needed.
    Returns (index, chunk_text, score) triples, same shape as your
    existing retrieve() function so downstream code stays unchanged.
    """
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    # Chroma returns distances (lower = more similar for cosine).
    # Convert to similarity scores (1 - distance) to match your existing
    # convention where higher = better.
    output = []
    for i in range(len(results["ids"][0])):
        idx = int(results["metadatas"][0][i]["chunk_index"])
        chunk = results["documents"][0][i]
        distance = results["distances"][0][i]
        score = 1 - distance  # cosine similarity
        output.append((idx, chunk, score))

    return output


if __name__ == "__main__":
    from main import MODEL_NAME
    from src.parse import extract_text
    from src.chunking import chunk_text
    from src.embed import embed_chunks, embed_query

    pdf_path = "data/testpdf.pdf"

    # ingest
    text = extract_text(pdf_path)
    chunks = chunk_text(text, chunk_size=300)
    embeddings = embed_chunks(chunks)

    collection = ingest_document(pdf_path, chunks, embeddings, EMBEDDING_MODEL)

    # query
    question = "What was the name of the cooperative community Robert Owen built?"
    query_vec = embed_query(question)
    results = chroma_retrieve(query_vec, collection, top_k=3)

    print(f"\nQuery: {question}\n")
    for idx, chunk, score in results:
        print(f"Chunk #{idx} (score {score:.4f}):")
        print(chunk[:200])
        print()