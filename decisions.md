# Decisions Log

## Stage 0 — Basic Pipeline

### Chunking
- Chunk size: 300 words
- Why: balance between context size and retrieval precision
- Overlap: none yet

### Embedding model
- Model: all-MiniLM-L6-v2 (sentence-transformers)
- Why: free, local, fast, small (384-dim) — deliberately a modest baseline so later
  stages (BM25, reranking) have room to show measurable improvement

### LLM
- Provider: Groq (free tier)
- Model: llama-3.1-8b-instant
- Why: fast, free, no local compute needed

### Storage / caching
- Format: pickle
- Cache key: SHA-256 hash of (PDF file bytes + model name), truncated to 16 chars
- Why: avoids recomputing embeddings for the same PDF+model on every run;
  avoids filename collisions between different PDFs with the same name;
  automatically invalidates cache if the embedding model changes

### Known limitations (intentional, to revisit later)
- Pickle storage doesn't scale — one file per PDF, no cross-document search,
  no indexing. Revisit in Stage 6 with Chroma.
- Floating-point non-determinism across different hardware/library versions
  not addressed — negligible impact on retrieval ranking, out of scope for now.
- No file size / page limits enforced yet.
- Retrieval quality (Stage 0 baseline) is weak/inconsistent on vague questions —
  expected, to be addressed by Stage 1 (report card) and Stage 2 (BM25).

### Stage 0 baseline observations
- scores ranged ~0.05–0.43 on test questions, best chunk not always semantically correct
- Observed: MiniLM cosine similarity scores are compressed into a narrow band
  (~0.05 for unrelated, ~0.40-0.50 for relevant) rather than spanning 0-1.
  Absolute score is not reliable as a standalone confidence signal — relative
  ranking matters more. This motivates Stage 5's LLM-based grounding check
  over a simple similarity threshold.