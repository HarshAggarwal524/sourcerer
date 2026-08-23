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
- Scores ranged ~0.05–0.43 on test questions, best chunk not always semantically correct.
- MiniLM cosine similarity scores are compressed into a narrow band (~0.05 for
  unrelated, ~0.40-0.50 for relevant) rather than spanning 0-1. Absolute score
  is not reliable as a standalone confidence signal — relative ranking matters more.
  This motivates Stage 5's LLM-based grounding check over a simple similarity threshold.

---

## Stage 1 — Report Card (Baseline)

### Test document
- File: data/testpdf.pdf
- Size: 33 chunks, ~9-10k words
- Content: two NCERT history chapters — Socialism in Europe/Russian Revolution,
  and The French Revolution

### Test sets
Four tiers built, all manually reviewed for correct/unambiguous ground truth
(except the experimental set):

| Test set | Questions | Description |
|---|---|---|
| Easy | 30 | Generated directly from chunks, high vocabulary overlap |
| Hard | 29 | Paraphrased/synonym-based, vaguer natural phrasing |
| Hardest | 22 | Multi-hop/inferential — combines two facts per chunk |
| Ambiguous (experimental) | 29 | Deliberately vague/riddle-like — NOT used for stage comparisons, ground truth less reliable |

### Sanity check
- Verified scoring correctness by testing a random-chunk-selection baseline:
  scored Recall@3 ≈ 0.07-0.10, matching theoretical chance (3/33 ≈ 0.09).
  Confirms real pipeline's scores reflect genuine retrieval quality, not a scoring bug.

### Key observations
- Hard set shows expected drop vs. Easy — plain vector search is measurably worse
  at handling paraphrased/synonym-heavy questions.
- Hardest set has HIGHER Recall@3 than Hard but LOWER MRR — multi-hop questions
  retain enough distinctive anchor details to surface the correct chunk in top 3,
  but rarely rank it #1. Different failure mode than Hard set's pure-paraphrasing problem.
- Ambiguous experimental set scored lowest as expected, but the drop was less severe
  than anticipated — MiniLM embeddings are reasonably robust to semantic vagueness.
- These four sets are the fixed yardstick held constant throughout all subsequent
  stages — isolates what each new technique actually contributes.

### BM25 debugging note (discovered during Stage 2 build)
- Found BM25's keyword_search() was returning zero-score chunks to pad out top_k,
  which then leaked into RRF fusion as false signal. Fixed by filtering out any
  chunk with score == 0 before returning results.

---

## Stage 2 — BM25 + RRF Hybrid Search

### What was built
- Added BM25 keyword search index (rank_bm25) alongside existing vector search.
- Combined both ranked lists using Reciprocal Rank Fusion (RRF, k=60) —
  rank-based merging that avoids incompatible score scales between BM25 and cosine similarity.
- Tokenization: lowercase + strip punctuation before splitting — prevents "Bolshevik"
  vs "bolshevik" mismatches and punctuation-adjacent token fragmentation.

### Observation
BM25+RRF meaningfully improved Easy and Hard sets — questions with enough distinctive
keyword overlap for keyword matching to reinforce what vector search found.
It HURT Hardest and Ambiguous sets — these questions lack a clean keyword anchor,
so BM25 confidently surfaces chunks with incidental word overlap that are actually
wrong, and RRF trusts that signal enough to displace the correct chunk.

Known real-world tradeoff of hybrid search: helps most on literal/specific queries,
can actively hurt on abstract/inferential ones.

---

## Stage 3 — Reranking (bge-reranker-v2-m3)

### What was built
- Added cross-encoder reranker (BAAI/bge-reranker-v2-m3) as a second pass after
  hybrid retrieval.
- Retrieves a wider candidate pool (originally 30, later corrected to 12 — see
  methodology note below), then reranker re-scores each candidate by reading the
  question and chunk together as a pair.
- Run on Google Colab (GPU) due to local RAM constraints (8GB local vs ~9GB+ needed).

### Observation
Reranking improved EVERY test set on EVERY metric — unlike Stage 2's mixed results.
Largest gains on exactly the sets Stage 2 hurt (Hardest, Ambiguous), confirming the
hypothesis: a cross-encoder can recover correct chunks that BM25's keyword noise had
buried, because it directly reasons about relevance rather than relying on proxy signals.
Easy set hitting 1.0/1.0 likely reflects a ceiling effect from test set construction,
not evidence the retrieval problem is fully solved.

### Methodology note: candidate pool size
- Originally ran with candidate_pool=30 out of 33 total chunks — effectively feeding
  the reranker almost the entire document, making upstream retrieval irrelevant.
- Corrected to candidate_pool=12 in the final consolidated run (Stage 4 onwards) to
  give retrieval genuine selectivity. Stage 3 numbers in the results table reflect
  the corrected pool=12 run for fair comparison.

### Note for Stage 7
Current test PDF is cleanly extracted text from a digital source — not representative
of real-world messy documents (scanned pages, images, multi-column layouts, sidebars,
tables). Stage 7 should test the full pipeline against a genuinely messy PDF to
evaluate real-world extraction robustness, separate from retrieval-quality work done
in Stages 2-4.

---

## Stage 4 — HyDE (Hypothetical Document Embeddings)

### What was built
- Added a pre-retrieval query rewriting step: before searching, the LLM generates
  a short hypothetical passage that *sounds like* it could answer the question,
  styled as a document excerpt.
- Embedding that passage instead of the raw question bridges the style gap between
  questions (interrogative) and document chunks (declarative statements).
- Both original question and hypothetical passage are logged for debugging.
- Tested in two configurations:
  - Stage 4a: HyDE alone (replaces plain hybrid retrieval, no reranking)
  - Stage 4b: HyDE + Reranking (HyDE for candidate selection, then reranked)

### Methodology note: candidate pool fix
- First Stage 4b run used candidate_pool=30 (same bug as Stage 3) — produced
  results identical to Stage 3, proving reranker was seeing the full corpus
  regardless of retrieval method. Fixed to candidate_pool=12 for valid comparison.

### Key finding: HyDE does NOT help once reranking is applied
Comparing Stage 3 (rerank, pool=12) vs. Stage 4b (HyDE+rerank, pool=12):
HyDE made Recall@3 worse on Hard/Hardest/Ambiguous. Likely cause: HyDE's
rewritten query sometimes contains fabricated specifics (hallucinated dates/names
not in the source document) which change which candidates reach the reranker —
occasionally for the worse, since the reranker was already doing strong
context-aware relevance judgment directly against the real question.

HyDE is useful when reranking is NOT present (Stage 4a vs. Stage 2 shows
real Recall gains on Hard/Hardest/Ambiguous), but adds no value — and sometimes
mild harm — once a strong reranker is already in the pipeline.

This is a legitimate, useful negative result: the pipeline's best configuration
is Stage 3 (hybrid + reranking), not Stage 4b (HyDE + reranking).

### Document-dependence caveat
This PDF (clean, simple NCERT textbook prose) may not reveal HyDE's full benefit.
HyDE helps most when there is a large vocabulary/style gap between questions and
source text (e.g. dense academic papers, legal contracts, technical specs). A
supplementary test on a harder document is flagged as a future exploratory task
for Stage 11's "what I'd build next" section — not part of the main results.

---

## Final Pipeline Decision (post-Stage 4)

**Default pipeline:** Stage 3 — hybrid BM25+RRF → reranking (pool=12).
Best overall Recall@3 across all sets.

**HyDE caveat:** on Hardest and Ambiguous sets, Stage 4b (HyDE+rerank)
shows better MRR than Stage 3 alone (0.787 vs 0.777 on Hardest,
0.734 vs 0.697 on Ambiguous) — meaning when it finds the right chunk,
it ranks it higher. The tradeoff is slightly lower Recall@3. HyDE is
therefore document/query-type dependent: most useful on abstract,
inferential, or vague questions where ranking quality matters more than
coverage. Available as a configurable mode — not enabled by default.


## Grounding Rate (Stage 5)

| Test set | Stage 3 | Stage 4b (HyDE+Rerank) |
|---|---|---|
| Easy (30 Q) | 0.9667 | 0.9333 |
| Hard (29 Q) | 0.6897 | 0.6897 |
| Hardest (22 Q) | 0.3636 | 0.5455 |
| Ambiguous (29 Q) | 0.3793 | 0.5862 |

### Key finding
Stage 3 has better Recall@3 on most sets. Stage 4b has meaningfully better
grounding rate on Hardest and Ambiguous — HyDE's rewritten query tends to
retrieve chunks that more closely match answer-style prose, leading the LLM
to stay closer to the source text rather than inferring beyond it.

Neither pipeline dominates: Stage 3 is better at finding the right chunk;
Stage 4b produces more grounded answers on abstract/inferential questions.
Final pipeline choice depends on whether retrieval coverage or answer
faithfulness is prioritized for a given use case.

## Stage 6 — Getting the Foundations Right

### What was built
- Replaced pickle-based storage (`src/store.py`) with Chroma, a proper local
  vector database (`chromadb` library).
- Chroma persists to `chroma_db/` directory automatically — no manual
  save/load required, no binary pickle files.
- Collection keyed by SHA-256 hash of (PDF bytes + model name), truncated
  to 16 chars — same collision-avoidance logic as the old pickle cache key.
  Same PDF + same model = same collection. Different PDF or model = different
  collection, no stale data.
- Added `hybrid_retrieve_chroma()` to `src/fusion.py` — same BM25+RRF fusion
  logic as before, but uses Chroma for the vector search step instead of a
  numpy embeddings array. BM25 keyword search still runs on raw chunks fetched
  from Chroma via `collection.get()`.
- Formalized two distinct functions in `main.py`:
  - `process_document()` — parse → chunk → embed → ingest into Chroma
  - `get_collection()` — load existing collection or process if missing.
    This is the single entry point both `main.py` and `app.py` use.
- `app.py` updated to use `get_collection()` and `hybrid_retrieve_chroma()`
  instead of old `load_or_build()` and `retrieve()`.
- Added `chroma_db/` to `.gitignore` — generated data, not source code,
  same reasoning as `cache/`.

### Architecture (data flow)
PDF file
→ extract_text() [src/parse.py]
→ chunk_text() [src/chunking.py]
→ embed_chunks() [src/embed.py]
→ ingest_document() [src/stage6_store.py]
→ Chroma (chroma_db/) [persistent local vector database]
↓
question
→ embed_query() [src/embed.py]
→ chroma_retrieve() [src/stage6_store.py] ← vector search via Chroma

keyword_search() [src/keyword_search.py] ← BM25 keyword search
→ reciprocal_rank_fusion() [src/fusion.py] ← RRF merges both lists
→ rerank() [src/stage3_rerank.py] ← cross-encoder re-scores
→ generate_answer() [src/generate.py] ← Groq LLM generates answer
→ check_grounding() [src/stage5_trust.py] ← LLM judge verifies answer
→ answer + confidence tag

### Debugging note
- First Colab run produced a stale/corrupted Chroma collection that always
  returned chunk #22 regardless of query. Fixed by deleting `chroma_db/`
  and rebuilding fresh. Root cause: earlier broken ingestion run had left
  a partial collection. `ingest_document()` now checks for existing
  collections before creating, but doesn't validate their integrity —
  a known limitation at this scale.

### Known limitations
- `chroma_db/` is local only — not persisted across Colab sessions.
  Must rebuild on each fresh Colab run (fast, ~seconds for this document).
- BM25 index still rebuilt in memory on every run — not stored in Chroma.
  Acceptable for this document size; would need separate persistence at scale.
- Multi-document search not yet implemented — each PDF gets its own
  Chroma collection, no cross-document querying. Revisit if project
  expands to multi-document support.
- No collection integrity check on load — if a collection exists but was
  partially written (e.g. interrupted ingestion), it will be loaded as-is
  without error. Fix: store chunk count in collection metadata and verify
  on load.

### Verified end-to-end on Colab (GPU)
- Louis XVI → Chunk #19, reranker score 0.5062, HIGH CONFIDENCE ✅
- Reign of Terror → Chunk #27, reranker score 0.9262, HIGH CONFIDENCE ✅
- Jacobins → Chunk #26, reranker score 0.4299, HIGH CONFIDENCE ✅
- Capital of Australia → Chunk #5, reranker score 0.0013, LOW CONFIDENCE ✅

### Model update (Stage 7)
- llama-3.1-8b-instant deprecated by Groq
- Switched to groq/compound-mini — fast, direct answers, no thinking tokens
- Other working options on current Groq account: allam-2-7b, qwen/qwen3.6-27b,
  openai/gpt-oss-20b, openai/gpt-oss-120b


## Stage 7 — Handling Bigger, Messier Files

### Test documents
1. Clean digital PDF (data/testpdf.pdf) — 33 chunks, baseline document
2. Full NCERT Class 9 History textbook (class_9_history.pdf) — 217 chunks,
   real-world messy PDF with figures, tables, cartoons, multi-column layouts,
   index, decorative cover page

### Findings from messy PDF test
- Cover page (chunk #0): severely garbled — decorative title text repeated 5x
  due to multi-column/decorative layout. Not a crash, just bad chunk content.
  Harmless in practice since it never scores well on real queries.
- Figure captions bleeding into chunks (e.g. "Fig.9 –" mid-sentence) — pipeline
  handles gracefully, LLM ignores caption noise and extracts real content.
- 217 chunks processed correctly, answers were accurate with appropriate
  HIGH/LOW CONFIDENCE tags on test questions.
- Broad questions ("What was the French Revolution?") correctly returned
  LOW CONFIDENCE when no single chunk window fully covered the answer —
  Stage 5 grounding check working correctly on real-world content.

### Guardrails added to src/parse.py
- MAX_PAGES = 150 — rejects PDFs over 150 pages with clear error message
- MAX_FILE_SIZE_MB = 50 — rejects files over 50MB
- MIN_CHARS_PER_PAGE = 100 — detects likely scanned/image PDFs and rejects
  with a specific "scanned PDF not supported" message instead of silently
  producing empty chunks
- app.py already calls check_file_limits() before processing

### Known limitations (documented)
- Scanned/image-only PDFs not supported (no OCR) — Stage 7 detects and rejects
- Cover pages with decorative/repeated text produce garbage chunks — not fixed,
  noted as acceptable since these chunks never rank highly in retrieval
- Figure captions bleed into text chunks — not fixed, LLM handles gracefully
- Files over 150 pages or 50MB rejected — configurable in parse.py constantsz
