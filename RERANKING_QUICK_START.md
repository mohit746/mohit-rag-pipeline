# Reranking Quick Start 🚀

## One-Minute Overview

**Reranking = Better retrieval using a second-pass filter**

```python
# Old way (less accurate):
results = hybrid_search(query)  # Get top 10

# New way (more accurate):
candidates = hybrid_search(query, k=20)      # Get top 20
results = cohere_rerank(candidates, top_n=5) # Rerank to top 5
```

---

## Setup (Already Done!)

```bash
# 1. Install library
pip install cohere

# 2. Add API key to .env
COHERE_API_KEY="your_key_here"

# 3. Initialize client
from cohere_reranking import init_cohere
client = init_cohere()
```

---

## Quick Usage

### Basic Reranking
```python
from cohere_reranking import rerank_results

# You have 20 documents + a query
reranked_docs, scores = rerank_results(
    query="What is X?",
    documents_with_metadata=docs_list,
    client=cohere_client,
    top_n=5
)

# Result: Top 5 docs sorted by Cohere relevance score
for doc, score in zip(reranked_docs, scores):
    print(f"Score: {score} | {doc['text'][:50]}...")
```

### Full RAG Pipeline with Reranking
```python
from cohere_reranking import hybrid_ask_with_reranking
from groq import Groq

cohere_client = init_cohere()
groq_client = Groq()

answer, contexts = hybrid_ask_with_reranking(
    question="What is a mutable default argument?",
    bm25=bm25,
    chunks=chunks,
    index=index,
    embeddings=embeddings,
    client=cohere_client,
    groq_client=groq_client
)

print(answer)
```

---

## Understanding the Scores

### Cohere Relevance Score (0.0 - 1.0)

```
Score Meaning
──────────────────────────────────────────────────
0.90+  Perfect match - directly answers query
0.70+  Excellent - very relevant content
0.50+  Good - mostly relevant
0.30+  Fair - somewhat related
0.10+  Weak - loosely related
<0.10  Irrelevant - doesn't help
```

### Example
```python
Query: "What is list slicing?"

Documents Ranked by Cohere:
[1] "list[1:3] returns elements at index..." → 0.95 ✓ Perfect!
[2] "Lists are mutable sequences..." → 0.42 (about lists, not slicing)
[3] "String indexing works like..." → 0.15 (wrong type)
```

---

## Reranking vs Other Techniques

| Feature | Hybrid Search | Hybrid + Rerank |
|---------|---------------|-----------------|
| **Speed** | Very fast (local) | Slower (API call) |
| **Cost** | Free | ~$0.01 per query |
| **Precision** | 60-70% | 70-85% |
| **Recall** | High (keeps more docs) | Lower (filters to 5) |
| **Best for** | Quick retrieval | Accuracy matters |

---

## Common Patterns

### Pattern 1: Rerank for Precision
```python
# Get many candidates, filter to best
candidates = hybrid_search(query, k=50)
best = cohere_rerank(candidates, top_n=5)
```

### Pattern 2: Rerank Multiple Retrievers
```python
# Combine results from different methods, then rerank
bm25_results = bm25_search(query)
semantic_results = semantic_search(query)
all_results = combine(bm25_results, semantic_results)
best = cohere_rerank(all_results, top_n=5)
```

### Pattern 3: Rerank with Filtering
```python
# Only rerank docs that pass a threshold
candidates = hybrid_search(query, k=20)
filtered = [d for d in candidates if d['score'] > 0.5]
best = cohere_rerank(filtered, top_n=5)
```

---

## Evaluating Your Own Data

```python
from evaluate_reranking import (
    score_faithfulness,
    score_answer_relevancy,
    score_retrieval_precision
)

# Score answers
f = score_faithfulness(answer, contexts, groq_client)      # 0.0-1.0
r = score_answer_relevancy(question, answer, embeddings)   # 0.0-1.0
p = score_retrieval_precision(question, chunks, gt, embed)  # 0.0-1.0

print(f"Faithfulness: {f:.2f}")
print(f"Relevancy: {r:.2f}")
print(f"Precision@5: {p:.2f}")
```

---

## Troubleshooting

### ❌ API Key Error
```
ModuleNotFoundError: No module named 'cohere'
```
**Fix:** `pip install cohere`

### ❌ Authentication Error
```
AuthenticationError: Invalid API key
```
**Fix:** Check `COHERE_API_KEY` in `.env` file

### ❌ Scores Are All Low (< 0.1)
**Likely cause:** Documents don't match query
**Solution:** Use `retrieve_k=30` to get more candidates

### ❌ Reranking Doesn't Help
**Possible reasons:**
- Dataset too small (need 20+ test cases)
- Hybrid search already excellent
- Document length mismatch (Cohere trained on longer docs)

**Solution:** Test on larger dataset or different documents

---

## Performance Tips

### Speed Optimization
```python
# ✓ Good: Rerank top 20 only
candidates = hybrid_search(query, k=20)
best = cohere_rerank(candidates, top_n=5)

# ✗ Bad: Rerank everything
all_docs = get_all_documents()  # 10,000+ docs
best = cohere_rerank(all_docs)  # TOO SLOW!
```

### Cost Optimization
```python
# Only rerank when hybrid score is borderline
candidates = hybrid_search(query, k=20)
if max_hybrid_score > 0.9:
    return candidates[:5]  # Hybrid confident enough
else:
    return cohere_rerank(candidates, top_n=5)  # Use reranker
```

### Batch Reranking
```python
# Rerank multiple queries at once
queries = [q1, q2, q3, q4, q5]
all_candidates = [hybrid_search(q, k=20) for q in queries]

for query, candidates in zip(queries, all_candidates):
    best = cohere_rerank(candidates, top_n=5)
    # Use best results
```

---

## Files Reference

### Implementation
- `src/11_cohere_reranking.py` — Core functions
  - `init_cohere()` — Initialize client
  - `rerank_results()` — Rerank documents
  - `hybrid_search_with_reranking()` — Two-stage retrieval
  - `hybrid_ask_with_reranking()` — Full RAG pipeline

### Evaluation
- `src/11_evaluate_reranking.py` — Compare methods
  - Run: `python src/11_evaluate_reranking.py`
  - Output: `results/reranking_comparison.json`

### Documentation
- `RERANKING_GUIDE.md` — Detailed explanation
- `RERANKING_QUICK_START.md` — This file!

---

## What's Next?

### Level 1: Experiment
- [ ] Test on your own data
- [ ] Try different `retrieve_k` values
- [ ] Compare with baseline

### Level 2: Optimize
- [ ] Implement cost check (only rerank if needed)
- [ ] Batch multiple queries
- [ ] Monitor latency

### Level 3: Advanced
- [ ] Multi-reranker ensemble
- [ ] Query expansion + rerank
- [ ] Re-ranking with metadata filtering

---

## Remember

> "Reranking trades **speed** for **accuracy**. Use it when precision matters more than latency."

✓ Good use cases:
- Offline processing
- Small batch queries
- When accuracy is critical

✗ Bad use cases:
- Real-time APIs
- Huge document sets
- When speed is critical
