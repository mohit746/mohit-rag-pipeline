# mohit-rag-pipeline

A 30-day sprint to build a production-grade RAG (Retrieval-Augmented Generation) system from scratch.

## Stack
- **LLM**: Groq (fast inference)
- **Embeddings**: HuggingFace `all-MiniLM-L6-v2` (free, local)
- **Vector store**: ChromaDB
- **Framework**: LangChain

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

## Days

| Day | Topic | Script |
|-----|-------|--------|
| 6 | Document loading + chunking strategies | `src/01_chunking_strategies.py` |

## Day 6 — Chunking Strategies

Drop 3 PDFs into `data/`, then run:

```bash
python src/01_chunking_strategies.py
```

Compares three strategies:
1. **Fixed size** — `CharacterTextSplitter(chunk_size=500)`
2. **Recursive** — `RecursiveCharacterTextSplitter(chunk_size=512)` ← production default
3. **Paragraph** — split on `\n\n`

Prints chunk count, avg/min/max size, and a sample chunk for each.
