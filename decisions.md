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

## Stage 1 — Report Card (Baseline)

- Test document: data/testpdf.pdf (33 chunks, ~9-10k words, two NCERT history chapters:
  Socialism in Europe/Russian Revolution, and The French Revolution)
- Test sets: four tiers, all manually reviewed for correct/unambiguous ground truth
  (except the experimental set, see below)

### Sanity check
- Verified scoring correctness by testing a random-chunk-selection baseline against
  the easy test set: scored Recall@3 ≈ 0.07-0.10, matching theoretical chance
  (3/33 ≈ 0.09). Confirms real pipeline's scores reflect genuine retrieval quality,
  not a scoring bug.

### Baseline results (plain vector search, Stage 0 pipeline)

| Test set | Questions | Description | Recall@3 | MRR |
|---|---|---|---|---|
| Easy | 30 | Generated directly from chunks, high vocabulary overlap | 0.8333 | 0.7761 |
| Hard | 29 | Paraphrased/synonym-based, vaguer natural phrasing | 0.6897 | 0.6357 |
| Hardest | 22 | Multi-hop/inferential — combines two facts per chunk | 0.7273 | 0.5671 |
| Ambiguous (experimental) | 29 | Deliberately vague/riddle-like — NOT used for stage comparisons, ground truth less reliable | 0.6207 | 0.5270 |

### Key observations
- Hard set shows the expected drop vs. Easy — plain vector search is measurably
  worse at handling paraphrased/synonym-heavy questions.
- Hardest set has HIGHER Recall@3 than Hard but LOWER MRR. Multi-hop questions
  retain enough distinctive anchor details (names, numbers) to usually surface
  the correct chunk somewhere in top 3, but rarely rank it #1, since no single
  clean paraphrase match exists. This is a different failure mode than Hard set's
  pure-paraphrasing problem (fewer ranking misses, more complete retrieval misses).
- This predicts Stage 3 (reranking) may specifically help MRR on the Hardest set,
  since reranking's job is fixing "found but ranked too low."
- Ambiguous experimental set scored lowest as expected, but the drop was less
  severe than anticipated — MiniLM embeddings appear reasonably robust to
  semantic vagueness in general.

### Why these baselines matter
These four sets (three official + one experimental) are the fixed yardstick every
subsequent stage (2: BM25+RRF, 3: reranking, 4: query rewriting, 5: grounding check)
will be measured against. Same document, same questions, held constant throughout —
isolates what each new technique actually contributes.

### Stage 2 debugging note
- Found BM25's keyword_search() was returning zero-score chunks to pad out
  top_k, which then leaked into RRF fusion as false signal. Fixed by
  filtering out any chunk with score == 0 before returning results.

## Stage 2 — BM25 + RRF Hybrid Search

### Results vs. Stage 1 baseline

| Test set | Recall@3 (before → after) | MRR (before → after) |
|---|---|---|
| Easy | 0.8333 → 0.9667 | 0.7761 → 0.8881 |
| Hard | 0.6897 → 0.7931 | 0.6357 → 0.7322 |
| Hardest | 0.7273 → 0.5909 | 0.5671 → 0.5735 |
| Ambiguous | 0.6207 → 0.5172 | 0.5270 → 0.5037 |

### Observation
BM25+RRF meaningfully improved Easy and Hard sets — questions with enough
distinctive keyword overlap for keyword matching to reinforce/confirm what
vector search found. It HURT Hardest and Ambiguous sets — these questions
lack a single clean keyword anchor to the correct chunk, so BM25 confidently
surfaces chunks with incidental word overlap that are actually wrong, and
RRF's fusion trusts that signal enough to displace the correct chunk.

This is a known real-world tradeoff of hybrid search: it helps most on
literal/specific queries and can actively hurt on abstract/inferential ones.
Worth considering in Stage 3 whether reranking (a more context-aware second
pass) can recover the Hardest/Ambiguous performance lost here.