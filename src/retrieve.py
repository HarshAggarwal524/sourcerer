import numpy as np
from sentence_transformers import util
from .parse import extract_text
from .chunking import chunk_text
from .embed import embed_chunks, embed_query
from .store import save_data, load_data

def retrieve(query_embedding, chunk_embeddings, chunks, top_k=1):
    """
    Returns the top_k chunks most similar to the query embedding,
    along with their similarity scores AND their original index.
    """
    scores = util.cos_sim(query_embedding, chunk_embeddings)[0]
    scores = scores.numpy()

    top_indices = np.argsort(scores)[::-1][:top_k]

    # Now returning the index too, not just (chunk, score)
    results = [(int(i), chunks[i], float(scores[i])) for i in top_indices]
    return results