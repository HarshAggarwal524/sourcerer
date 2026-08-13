# Results Table

All scores measured on data/testpdf.pdf (33 chunks, two NCERT history chapters).
Rerank variants use candidate_pool=12. Ambiguous set is experimental — not used
for stage-to-stage comparisons (ground truth less reliable).

## Recall@3

| Test set | S1 Vector | S2 Hybrid | S3 +Rerank | S4a HyDE | S4b HyDE+Rerank |
|---|---|---|---|---|---|
| Easy (30 Q) | 0.8333 | 1.0000 | 1.0000 | 0.9667 | 1.0000 |
| Hard (29 Q) | 0.6897 | 0.7586 | 0.9310 | 0.7931 | 0.8966 |
| Hardest (22 Q) | 0.7273 | 0.7727 | 0.8636 | 0.6364 | 0.8182 |
| Ambiguous* (29 Q) | 0.6207 | 0.6552 | 0.8621 | 0.6552 | 0.8276 |

## MRR

| Test set | S1 Vector | S2 Hybrid | S3 +Rerank | S4a HyDE | S4b HyDE+Rerank |
|---|---|---|---|---|---|
| Easy (30 Q) | 0.7761 | 0.8881 | 1.0000 | 0.8364 | 1.0000 |
| Hard (29 Q) | 0.6357 | 0.7322 | 0.9224 | 0.7038 | 0.8707 |
| Hardest (22 Q) | 0.5671 | 0.5735 | 0.7765 | 0.6222 | 0.7868 |
| Ambiguous* (29 Q) | 0.5270 | 0.5037 | 0.6968 | 0.5329 | 0.7339 |

## Key Takeaways Per Stage

| Stage | What it added | Net effect |
|---|---|---|
| S1 → S2 (BM25+RRF) | Keyword search alongside vector search | Helped Easy/Hard, hurt Hardest/Ambiguous |
| S2 → S3 (Reranking) | Cross-encoder second pass | Uniform large gain on all sets — strongest single addition |
| S2 → S4a (HyDE alone) | Query rewriting before retrieval | Modest Recall gains on Hard/Hardest/Ambiguous, mixed MRR |
| S3 → S4b (HyDE+Rerank) | HyDE on top of reranking | Slightly worse on most sets — reranker already handles relevance well |

## Grounding Rate (Stage 5)

| Test set | Stage 3 | Stage 4b (HyDE+Rerank) |
|---|---|---|
| Easy (30 Q) | 0.9667 | 0.9333 |
| Hard (29 Q) | 0.6897 | 0.6897 |
| Hardest (22 Q) | 0.3636 | 0.5455 |
| Ambiguous (29 Q) | 0.3793 | 0.5862 |