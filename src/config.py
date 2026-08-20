# ─── Central configuration ───────────────────────────────────────────────────
# Change model names here — they'll propagate everywhere automatically.

# LLM used for answer generation and grounding checks
LLM_PROVIDER = "gemini"
LLM_MODEL = "gemini-3.6-flash"

# Embedding model for chunks and queries
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Reranker model
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# Chunking
CHUNK_SIZE = 300
CHUNK_NEIGHBORS = 1   # neighbors on each side for context expansion

# Retrieval
RERANK_POOL = 12      # candidate pool size before reranking
TOP_K_RERANK = 3      # how many chunks to keep after reranking