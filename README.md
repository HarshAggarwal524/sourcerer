# PDF RAG Pipeline

A RAG-based PDF question-answering system using hybrid retrieval, reranking, HyDE, and LLM-based grounding.

## Pipeline

```text
PDF
 ↓
Chunking → Embeddings → ChromaDB
                         ↓
Question → Vector + BM25 → RRF → Reranker → LLM → Grounding Check
```

## Tech Stack

- **Embeddings:** `all-MiniLM-L6-v2`
- **Keyword Search:** BM25
- **Vector DB:** ChromaDB
- **Reranker:** `BAAI/bge-reranker-v2-m3`
- **LLM:** Groq (`groq/compound-mini`)
- **Framework:** Python

## Results

Evaluated on 33 chunks from two NCERT history chapters.

| Method | Easy | Hard | Hardest |
|---|---:|---:|---:|
| Vector | 83.3% | 69.0% | 72.7% |
| Hybrid | 100% | 75.9% | 77.3% |
| Hybrid + Rerank | **100%** | **93.1%** | **86.4%** |
| HyDE + Rerank | 100% | 89.7% | 81.8% |

### Key Finding

Cross-encoder reranking provided the most consistent improvement across the evaluation sets.

HyDE is available as an optional mode, but did not consistently improve retrieval when reranking was already enabled.

## Project Structure

```text
├── app.py
├── main.py
├── data/
├── src/
│   ├── parse.py
│   ├── chunking.py
│   ├── embed.py
│   ├── keyword_search.py
│   ├── fusion.py
│   ├── stage3_rerank.py
│   ├── stage5_trust.py
│   └── stage6_store.py
└── README.md
```

## Limitations

- Scanned PDFs/OCR are not supported.
- Maximum file size: 50 MB.
- Maximum pages: 150.
- Multi-document retrieval is not implemented yet.

## Future Work

- OCR support
- Multi-document search
- Better PDF layout handling
- Larger evaluation datasets
- Improved grounding and confidence calibration

Link to app: https://sourcerer-m4ki9x5c2uugy6hrjdvlny.streamlit.app/#answer
