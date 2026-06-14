"""
COHERE RERANKING MODULE
═══════════════════════════════════════════════════════════════════════════════

What is Reranking?
─────────────────
Reranking is like a second-pass filtering:
1. Hybrid search gives us 20 "decent" candidates (fast, broad net)
2. Reranker takes those 20 and intelligently reorders them (slower, smarter)
3. We keep only the top 5 (precision over recall)

Why Cohere Reranker?
────────────────────
- Trained specifically to rank documents by relevance to a query
- Understands semantic meaning (not just keyword matching)
- Returns confidence scores (0.0 - 1.0) for each result
- Much cheaper than running full embedding model on every document

Pipeline visualization:
[Query] → [Hybrid Search] → [Top 20 candidates]
                ↓
        [Cohere Reranker]
                ↓
        [Top 5 best matches] → [LLM]
"""

import os
import cohere
from dotenv import load_dotenv

load_dotenv()


# ═══════════════════════════════════════════════════════════════════════════════
# 1. INITIALIZE COHERE CLIENT
# ═══════════════════════════════════════════════════════════════════════════════

def init_cohere():
    """
    Initialize Cohere API client

    Returns:
        cohere.ClientV2: Cohere client object

    Note: Uses COHERE_API_KEY from .env
    """
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        raise ValueError("COHERE_API_KEY not found in .env")

    client = cohere.ClientV2(api_key=api_key)
    print("✓ Cohere client initialized")
    return client


# ═══════════════════════════════════════════════════════════════════════════════
# 2. RERANK FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def rerank_results(query: str, documents_with_metadata: list, client, top_n: int = 5):
    """
    Rerank documents using Cohere's neural reranker model

    Args:
        query: The search query (what the user is asking)
        documents_with_metadata: List of dicts with:
            - "text": The document chunk text
            - "chunk": Original chunk object (metadata)
            - "rrf_score": Original hybrid search score
            - "rank": Original position
        client: Cohere ClientV2 object
        top_n: How many top results to return (default: 5)

    Returns:
        Tuple of (reranked_docs, scores):
            - reranked_docs: List of documents in NEW order (best first)
            - scores: List of Cohere confidence scores (0.0-1.0)

    How it works:
    ─────────────
    1. Extract just the text from documents
    2. Send to Cohere: "Rank these docs by relevance to: {query}"
    3. Cohere returns scores for each document
    4. Sort by these new scores
    5. Return top_n results
    """

    # Extract text from documents
    # (Cohere.rerank expects a list of strings)
    texts = [doc["text"] for doc in documents_with_metadata]

    print(f"\n  Reranking {len(texts)} documents with Cohere...")
    print(f"  Model: rerank-english-v3.0")

    # Call Cohere rerank API
    # ─────────────────────────
    # rerank-english-v3.0 is trained on:
    # - Query-document relevance
    # - Semantic understanding
    # - Works best for English text (good for Python docs!)
    response = client.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=texts,  # List of document strings
        top_n=top_n      # Return only top N (efficient!)
    )

    # Parse results
    # ─────────────
    # Cohere returns: results.results = list of RerankResult objects
    # Each has: .index (original position) and .relevance_score (0.0-1.0)

    reranked_docs = []
    scores = []

    for result in response.results:
        original_index = result.index
        cohere_score = result.relevance_score

        # Get the original document and ADD the Cohere score
        original_doc = documents_with_metadata[original_index]

        reranked_doc = {
            **original_doc,  # Keep all original fields
            "cohere_score": cohere_score,  # Add new Cohere score
            "rerank_position": len(reranked_docs) + 1  # New position (1, 2, 3...)
        }

        reranked_docs.append(reranked_doc)
        scores.append(round(cohere_score, 4))

    print(f"  ✓ Reranked to top {len(reranked_docs)}")

    return reranked_docs, scores


# ═══════════════════════════════════════════════════════════════════════════════
# 3. FULL HYBRID + RERANK PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def hybrid_search_with_reranking(
    query: str,
    bm25,
    chunks,
    index,
    embeddings,
    client,
    retrieve_k: int = 20,  # Get top 20 from hybrid
    final_k: int = 5,      # Rerank down to top 5
    hybrid_search_func=None  # Optional: pass hybrid_search function
):
    """
    Complete pipeline: Hybrid Search → Cohere Rerank

    Args:
        query: User's question
        bm25, chunks, index, embeddings: From 09_hybrid_search.py
        client: Cohere ClientV2
        retrieve_k: How many to get from hybrid search (larger net)
        final_k: How many to return after reranking (higher precision)
        hybrid_search_func: The hybrid_search function (will import if not provided)

    Returns:
        reranked_results: List of documents with both RRF and Cohere scores

    Why this 2-stage approach?
    ──────────────────────────
    1. Hybrid search is FAST → get 20 candidates quickly
    2. Reranker is SMART → intelligently pick best 5
    3. Together: Speed + Accuracy

    vs. Just using Reranker:
    - Would need to rerank ALL documents (slow)
    - Hybrid + Rerank is much faster
    """

    # Import hybrid search function if not provided
    if hybrid_search_func is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "hybrid_search", os.path.join(os.path.dirname(__file__), "09_hybrid_search.py")
        )
        hybrid_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hybrid_module)
        hybrid_search_func = hybrid_module.hybrid_search

    print(f"\n{'='*70}")
    print(f"HYBRID SEARCH + COHERE RERANKING")
    print(f"{'='*70}")
    print(f"Query: {query}")
    print(f"\nStage 1: Hybrid Search (BM25 + Semantic + RRF)")
    print(f"  → Retrieving top {retrieve_k} candidates...")

    # Stage 1: Get candidates from hybrid search
    hybrid_results = hybrid_search_func(
        query=query,
        bm25=bm25,
        chunks=chunks,
        index=index,
        embeddings=embeddings,
        top_k=retrieve_k  # Get 20 instead of 10
    )

    # Format for reranker
    # Cohere needs the text + we keep metadata
    docs_for_rerank = [
        {
            "text": result["chunk"]["text"],  # Just the text
            "chunk": result["chunk"],         # Keep original chunk
            "rrf_score": result["rrf_score"],  # Keep hybrid score
            "rank": result["rank"]            # Keep original position
        }
        for result in hybrid_results
    ]

    print(f"  ✓ Got {len(docs_for_rerank)} candidates\n")

    # Stage 2: Rerank with Cohere
    print(f"Stage 2: Cohere Reranking")
    reranked, cohere_scores = rerank_results(
        query=query,
        documents_with_metadata=docs_for_rerank,
        client=client,
        top_n=final_k
    )

    # Print comparison
    print(f"\nComparison - Original vs Reranked Position:")
    print(f"{'─'*70}")
    for i, doc in enumerate(reranked, 1):
        print(f"  [{i}] Original Rank: {doc['rank']:2d} (RRF: {doc['rrf_score']:.4f})")
        print(f"      Cohere Score: {doc['cohere_score']:.4f}")
        print(f"      Text: {doc['chunk']['text'][:60]}...")
        print()

    return reranked


# ═══════════════════════════════════════════════════════════════════════════════
# 4. INTEGRATION WITH RAG PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def hybrid_ask_with_reranking(
    question: str,
    bm25,
    chunks,
    index,
    embeddings,
    client,
    groq_client,
    hybrid_module=None
):
    """
    Full RAG pipeline with reranking:
    Query → Hybrid Search → Cohere Rerank → LLM

    This replaces the old hybrid_ask() with a smarter version
    """

    # Import hybrid module if not provided
    if hybrid_module is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "hybrid_search", os.path.join(os.path.dirname(__file__), "09_hybrid_search.py")
        )
        hybrid_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hybrid_module)

    # Use hybrid_search_with_reranking instead of plain hybrid_search
    reranked_results = hybrid_search_with_reranking(
        query=question,
        bm25=bm25,
        chunks=chunks,
        index=index,
        embeddings=embeddings,
        client=client,
        retrieve_k=20,
        final_k=5,
        hybrid_search_func=hybrid_module.hybrid_search
    )

    # Get token budget selector and context builder from hybrid module
    select_chunks_within_budget = hybrid_module.select_chunks_within_budget
    build_context = hybrid_module.build_context

    # Select chunks within token budget
    selected = select_chunks_within_budget(
        [{"chunk": r["chunk"], "rrf_score": r["rrf_score"], "rank": r["rerank_position"]}
         for r in reranked_results]
    )

    if not selected:
        return "I don't have that information in the provided documents.", []

    # Build context and get answer
    context = build_context(selected)

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": f"""You are a helpful assistant.
Answer the question using ONLY the context below.
For every fact you state, cite the source using: [Source: title, Page X]
If the answer is not in the context, say "I don't have that information in the provided documents."
Context: {context}"""
            },
            {"role": "user", "content": question}
        ]
    )

    answer = response.choices[0].message.content
    return answer, reranked_results


# ═══════════════════════════════════════════════════════════════════════════════
# TEST / DEMO
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import importlib.util
    from groq import Groq

    # Import hybrid search module
    spec = importlib.util.spec_from_file_location(
        "hybrid_search", os.path.join(os.path.dirname(__file__), "09_hybrid_search.py")
    )
    hybrid_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hybrid_module)

    # Initialize everything
    print("Initializing...")
    chunks = hybrid_module.load_and_chunk_documents()
    bm25, chunks = hybrid_module.build_bm25_index(chunks)
    index, embeddings = hybrid_module.init_pinecone()
    cohere_client = init_cohere()
    groq_client = Groq()

    # Test queries
    test_queries = [
        "What is a mutable default argument?",
        "How does dictionary iteration work?",
    ]

    print("\n" + "="*70)
    print("HYBRID SEARCH + COHERE RERANKING TEST")
    print("="*70)

    for query in test_queries:
        reranked = hybrid_search_with_reranking(
            query=query,
            bm25=bm25,
            chunks=chunks,
            index=index,
            embeddings=embeddings,
            client=cohere_client,
            hybrid_search_func=hybrid_module.hybrid_search
        )
        print()
