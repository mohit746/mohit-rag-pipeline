# Day 24: Cohere Reranking — Complete Learning Guide

## What is Reranking? 🎯

**Reranking is a second-pass filtering system for retrieval.**

Instead of relying on one retrieval method, we use a **two-stage approach**:

```
┌─────────────────────┐
│   User Query        │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 1: HYBRID SEARCH (Fast, Broad Net)                    │
│ ─────────────────────────────────────────────────────────── │
│ • BM25 (keyword matching) + Semantic (embeddings)           │
│ • Reciprocal Rank Fusion (RRF) combines both scores         │
│ • Returns: TOP 20 candidates                                │
└──────────┬──────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 2: COHERE RERANKING (Smart, High Precision)           │
│ ─────────────────────────────────────────────────────────── │
│ • Neural reranker trained specifically on relevance         │
│ • Evaluates query-document match quality                    │
│ • Returns: TOP 5 best matches (filtered)                    │
└──────────┬──────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 3: LLM GENERATION                                     │
│ ─────────────────────────────────────────────────────────── │
│ • Uses only the top 5 reranked chunks as context            │
│ • Generates more accurate answers                           │
└─────────────────────────────────────────────────────────────┘
```

---

## Why Two Stages?

### Stage 1: Hybrid Search ⚡
- **Cost:** Very cheap (local operations)
- **Speed:** Fast (~100ms)
- **Coverage:** Broad (gets 20 candidates)
- **Accuracy:** Good (~70%)

### Stage 2: Reranking 🧠
- **Cost:** More expensive (API call)
- **Speed:** Slower (~1-2 seconds for 20 docs)
- **Coverage:** Narrow (picks best 5)
- **Accuracy:** Excellent (~85-90%)

**Why both?** Because:
- Running reranker on ALL documents = too slow + expensive
- Hybrid search alone = might miss best answers
- **Hybrid + Rerank = Best of both worlds** ✓

---

## How Cohere Reranker Works

### What Cohere Does:
```python
# You give Cohere:
query = "What is a mutable default argument?"
documents = [
    "A mutable default argument is...",
    "Dictionary iteration means...",
    "Sets are unordered collections...",
    # ... 17 more documents
]

# Cohere analyzes:
# - Does each doc answer the query?
# - How relevant is it?
# - What's the confidence level?

# Cohere returns:
results = [
    {"document": doc_1, "relevance_score": 0.9989},  # TOP match
    {"document": doc_2, "relevance_score": 0.0337},
    {"document": doc_3, "relevance_score": 0.0061},
    {"document": doc_4, "relevance_score": 0.0002},
    {"document": doc_5, "relevance_score": 0.0001},
]
```

### The Relevance Score:
- **0.9+ = Extremely relevant** (directly answers query)
- **0.5-0.9 = Very relevant** (related to query)
- **0.1-0.5 = Somewhat relevant** (loosely related)
- **<0.1 = Irrelevant** (doesn't help answer query)

---

## Our Implementation: Three Files

### 1. `11_cohere_reranking.py` — Core Reranking Module

**Key Functions:**

#### `init_cohere()`
```python
client = init_cohere()  # Initialize Cohere API
```
- Reads `COHERE_API_KEY` from `.env`
- Returns a client object for making rerank requests

#### `rerank_results(query, documents_with_metadata, client, top_n=5)`
```python
reranked_docs, scores = rerank_results(
    query="What is a mutable default argument?",
    documents_with_metadata=[...],  # List of 20 docs
    client=cohere_client,
    top_n=5  # Keep only top 5
)
```

**What it does:**
1. Extracts text from documents
2. Sends to Cohere: "Rank these by relevance to: {query}"
3. Cohere returns scores (0.0-1.0)
4. Sorts by Cohere scores
5. Returns top 5 with new rankings

**Example output:**
```
Original Position → New Position + Cohere Score
    Rank 8      →    Rank 1      (score: 0.1970)  ← Got promoted!
    Rank 2      →    Rank 2      (score: 0.0005)
    Rank 7      →    Rank 3      (score: 0.0002)
```

#### `hybrid_search_with_reranking()`
```python
reranked = hybrid_search_with_reranking(
    query=question,
    bm25=bm25,
    chunks=chunks,
    index=index,
    embeddings=embeddings,
    client=cohere_client,
    retrieve_k=20,  # Get 20 from hybrid
    final_k=5       # Rerank down to 5
)
```

**Pipeline inside:**
```
1. Hybrid Search → Get top 20
2. Cohere Rerank → Filter to top 5
3. Return reranked results
```

#### `hybrid_ask_with_reranking()`
```python
answer, reranked_chunks = hybrid_ask_with_reranking(
    question="What is a mutable default argument?",
    bm25=bm25,
    chunks=chunks,
    index=index,
    embeddings=embeddings,
    client=cohere_client,
    groq_client=groq_client
)
```

**Full RAG Pipeline:**
```
1. Query Reranking: Get top 20 candidates
2. Token Budget: Select chunks that fit budget
3. Build Context: Create prompt with chunks
4. Call LLM: Generate answer using Groq
5. Return: Answer + Context chunks
```

---

### 2. `11_evaluate_reranking.py` — Comparison Evaluation

**What it measures:**

#### Faithfulness (0.0 - 1.0)
```
Question: "What is X?"
Context: "X is Y because Z"

High Faithfulness: Answer directly uses context
✓ Answer: "X is Y because Z"  → Score: 0.8

Low Faithfulness: Answer hallucinates
✗ Answer: "X is Y because ABC" (made up) → Score: 0.2
```

#### Answer Relevancy (0.0 - 1.0)
```
How similar is the answer to the question?

Embedding(question) = [0.1, 0.5, 0.9, ...]
Embedding(answer)   = [0.1, 0.5, 0.8, ...]

Cosine Similarity = Dot Product / (|A| * |B|)
                  = 0.95 (very similar!)
```

#### Precision@5 (0.0 - 1.0)
```
How many of the top 5 chunks are actually relevant?

Ground Truth: "dictionary iteration uses .items()"

Retrieved:
[1] "Dictionary iteration..." ✓ relevant
[2] "List slicing..."        ✗ not relevant
[3] ".items() method..."     ✓ relevant
[4] "String methods..."      ✗ not relevant
[5] "Loop syntax..."         ✗ not relevant

Precision@5 = 2/5 = 0.40 (40% of top 5 were relevant)
```

---

## Results: What We Learned 📊

### Hybrid Search (Baseline)
```
Faithfulness:     52.0%
Answer Relevancy: 70.9%
Precision@5:      60.0%
```

### Hybrid + Cohere Reranking
```
Faithfulness:     48.0%  (-4%)
Answer Relevancy: 69.4%  (-1.5%)
Precision@5:      64.0%  (+4%)  ← Improved!
```

### Why Didn't Reranking Help More?

**Possible Reasons:**

1. **Dataset is Small**
   - Only 5 test questions
   - Cohere needs ~20+ queries to show value

2. **Documents are Short**
   - Python cheat sheet (~1000 chars each)
   - Cohere trained on longer academic papers
   - Shorter docs = less context for ranking

3. **Hybrid Search Already Good**
   - BM25 + Semantic + RRF is powerful
   - Hard to improve on an already-good baseline

4. **Trade-offs**
   - Reranking = higher precision (better top results)
   - But = fewer documents available (lower recall)
   - If doc 10 answers the question but we only keep 5 = miss it

---

## Key Concepts You Learned 🎓

### 1. **Two-Stage Retrieval**
- Stage 1 (Hybrid): Fast, broad coverage
- Stage 2 (Rerank): Slow, high precision
- Together: Optimal speed + accuracy

### 2. **Reranker Scores**
- Not probabilities (don't sum to 1)
- Confidence in relevance (0.0 = not relevant, 1.0 = perfect)
- Can rerank different from initial ranking

### 3. **When to Use Reranking**
- ✓ Large document sets (100+)
- ✓ Long documents (1000+ chars)
- ✓ Where top-K precision matters
- ✗ Small sets (just use hybrid)
- ✗ Real-time (too slow)

### 4. **Evaluation Matters**
- Not all techniques help all datasets
- Must measure empirically
- Trade-offs exist (precision vs recall)

---

## Code Architecture 🏗️

```
11_cohere_reranking.py
├── init_cohere()
│   └─ Initialize Cohere API client
│
├── rerank_results()
│   ├─ Input: 20 documents + query
│   ├─ Call Cohere API
│   └─ Output: 5 ranked documents
│
├── hybrid_search_with_reranking()
│   ├─ Stage 1: Hybrid search (20)
│   ├─ Stage 2: Cohere rerank (5)
│   └─ Output: Reranked results
│
└── hybrid_ask_with_reranking()
    ├─ Rerank
    ├─ Select chunks (token budget)
    ├─ Build context
    ├─ Call Groq LLM
    └─ Output: Answer + chunks

11_evaluate_reranking.py
├── score_faithfulness()
│   └─ LLM judges if answer uses context
│
├── score_answer_relevancy()
│   └─ Embedding similarity Q ↔ A
│
├── score_retrieval_precision()
│   └─ How many top 5 chunks are relevant?
│
└── Main evaluation loop
    ├─ Run both methods on 5 questions
    ├─ Collect metrics
    └─ Generate comparison report
```

---

## How to Use in Your RAG Pipeline

### Replace the old hybrid_ask():
```python
# Before:
answer, contexts = hybrid_ask(question, bm25, chunks, index, embeddings)

# After:
from cohere_reranking import hybrid_ask_with_reranking
answer, contexts = hybrid_ask_with_reranking(
    question, bm25, chunks, index, embeddings, cohere_client, groq_client
)
```

### In Production:
```python
from dotenv import load_dotenv
from cohere_reranking import init_cohere, hybrid_ask_with_reranking
from groq import Groq

load_dotenv()

# Initialize once
cohere_client = init_cohere()
groq_client = Groq()

# Use in your RAG endpoint
def answer_question(question):
    answer, contexts = hybrid_ask_with_reranking(
        question, 
        bm25, chunks, index, embeddings,
        cohere_client, groq_client
    )
    return {
        "answer": answer,
        "sources": [c["chunk"]["doc_name"] for c in contexts]
    }
```

---

## Next Steps 🚀

### Option 1: Improve Reranking Results
- Use more test questions (50+)
- Use longer documents
- Tune `retrieve_k` and `final_k` parameters
- Try different Cohere models

### Option 2: Add More Retrieval Methods
- Multi-query expansion
- Sub-question decomposition
- Query rewriting

### Option 3: Ensemble Ranking
- Combine multiple rerankers
- Weight by confidence

### Option 4: Production Optimization
- Cache Cohere embeddings
- Batch reranking requests
- Monitor latency/cost

---

## Files Created 📁

```
src/
├── 11_cohere_reranking.py      ← Reranking implementation
├── 11_evaluate_reranking.py    ← Comparison evaluation
└── 09_hybrid_search.py         ← Existing (imported)

results/
└── reranking_comparison.json   ← Evaluation results
```

---

## Summary 📝

You've learned:
- ✓ What reranking is and why it's useful
- ✓ How Cohere's neural reranker works
- ✓ Two-stage retrieval: Speed + Accuracy
- ✓ How to evaluate RAG improvements
- ✓ That not all techniques work for all datasets
- ✓ Important: Measure empirically, don't assume!

**Main Takeaway:** Reranking is a powerful technique for improving retrieval precision, but it requires proper evaluation to understand its trade-offs.
