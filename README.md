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


## DAY 9 - self written notes below

## Stack
- **LLM**: Groq (llama-3.3-70b-versatile)
- **Embeddings**: Google Gemini (`models/gemini-embedding-001`, 3072-dim)
- **Vector store**: Pinecone (migrated from ChromaDB on Day 9)
- **Framework**: LangChain


## Why Pinecone over ChromaDB in Production

-> Chroma DB sits in your local system, in pinecone datbase is on cloud, developer does not have to worry about the data on local.
-> Scalability is issue in ChromaDB, it is not the problem with pinecone.
-> ChromaDB will be down if your system is turned off, pinecone has 0 downtime.
-> ChromaDB has list type of integration where as in pinecone it have tuples/JSON set.
-> ChromaDB setup and connection has to be perfect and requires more code, in pinecode you just have to connect with the right URL with API keys.
-> ChromaDB uses internal disk to setup the database, Pinecone uses its own proprietary storage engine built for vector search. 

## pgvector vs Pinecone — When Each Wins

-> pgvector is used for already existing database due to postgres availability.
? pgvector is better for <5 M vector, where pinecone is better for >10M vector, sub- 10ms query latency
-> pgvector is best when user need JOIN queries with relational database.
-> pgvector is good when you need atomic transactions (insert document + vector creation)
-> Multi tenant SaaS with namespace then pinecone is better.
-> If there is no existing DB and new setup is created with greenfield project - pinecone is best option


## Day 10 — RAGAS Evaluation

Evaluated 14 Q&A pairs generated from Python Refresher 1.pdf using a manual RAGAS-style scorer (Groq llama-3.3-70b as judge, Gemini embeddings for relevancy).

| Metric | Score | Target | Status |
|--------|-------|--------|--------|
| Faithfulness | 0.49 | > 0.75 | ❌ Below target |
| Answer Relevancy | 0.84 | > 0.70 | ✅ Above target |

**Key finding:** Low faithfulness on questions about .split(), map(), list() wrapping — LLM answers from general knowledge when retrieved chunks are MCQ-format and don't contain full explanations.
