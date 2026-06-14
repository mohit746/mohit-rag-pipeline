"""
EVALUATE HYBRID SEARCH vs HYBRID + COHERE RERANKING
═══════════════════════════════════════════════════════════════════════════════

This script compares:
1. Baseline: Hybrid Search alone (BM25 + Semantic + RRF)
2. Improved: Hybrid Search + Cohere Reranking

Metrics we measure:
- Faithfulness: Does the answer stick to the context? (0.0 - 1.0)
- Relevancy: How similar is answer to the question? (0.0 - 1.0)
- Retrieval Precision@5: Are the top 5 chunks relevant?

Expected outcome:
- Cohere reranking should boost retrieval quality
- Better ranked docs → better answers → higher scores
"""

import os
import json
import numpy as np
import importlib.util
from dotenv import load_dotenv
from groq import Groq
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

# Import modules
spec_hybrid = importlib.util.spec_from_file_location(
    "hybrid_search", os.path.join(os.path.dirname(__file__), "09_hybrid_search.py")
)
hybrid_module = importlib.util.module_from_spec(spec_hybrid)
spec_hybrid.loader.exec_module(hybrid_module)

spec_rerank = importlib.util.spec_from_file_location(
    "reranking", os.path.join(os.path.dirname(__file__), "11_cohere_reranking.py")
)
rerank_module = importlib.util.module_from_spec(spec_rerank)
spec_rerank.loader.exec_module(rerank_module)


# Test questions (same as before)
EVAL_QUESTIONS = [
    {
        "question": "What is a mutable default argument in Python?",
        "ground_truth": "a mutable default argument is a default argument that is a mutable object, such as a list, and is evaluated only once when the function is defined, not each time the function is called"
    },
    {
        "question": "How does list slicing with assignment work in Python?",
        "ground_truth": "When you do list slicing with assignment, the elements at the specified indices in the list are replaced by the new values."
    },
    {
        "question": "What happens when you call my_set.add([4, 5]) in Python?",
        "ground_truth": "When you call my_set.add([4, 5]) in Python, it will raise a TypeError because lists are not hashable and cannot be added to a set."
    },
    {
        "question": "What is function call and return values in Python?",
        "ground_truth": "A function call in Python is when you invoke a function by its name followed by parentheses, optionally with arguments inside. The return value is the result that the function gives back after execution, which can be used in further expressions or stored in variables."
    },
    {
        "question": "What is dictionary iteration in Python?",
        "ground_truth": "Dictionary iteration in Python refers to the process of looping through the keys, values, or key-value pairs of a dictionary using methods like .keys(), .values(), or .items() in a for loop."
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# SCORING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def score_faithfulness(answer, contexts, groq_client):
    """
    Score how faithful the answer is to the provided context

    Logic:
    ──────
    We ask an LLM: "Given this context, is the answer true?"
    - 1.0 = Every fact comes from context (no hallucination)
    - 0.0 = Answer contains made-up facts

    Args:
        answer: The LLM's generated answer
        contexts: List of chunk dicts from retrieval
        groq_client: Groq client for scoring

    Returns:
        float: Faithfulness score (0.0 - 1.0)
    """
    context_text = "\n".join([c["chunk"]["text"] for c in contexts])

    prompt = f"""Rate how faithful this answer is to the given context on a scale of 0.0 to 1.0.
1.0 = every fact in the answer comes directly from the context.
0.0 = the answer contains hallucinations or unsupported claims.

Context: {context_text[:2000]}

Answer: {answer}

Reply with only a decimal number between 0.0 and 1.0."""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    try:
        return float(response.choices[0].message.content.strip())
    except:
        return 0.0


def score_answer_relevancy(question, answer, embeddings_model):
    """
    Score how relevant the answer is to the question

    Logic:
    ──────
    Use embeddings to measure semantic similarity:
    - Embed the question
    - Embed the answer
    - Compute cosine similarity (dot product of normalized vectors)
    - Result: how much do Q and A "talk about the same thing"?

    Args:
        question: The original question
        answer: The generated answer
        embeddings_model: GoogleGenerativeAIEmbeddings instance

    Returns:
        float: Relevancy score (0.0 - 1.0)
    """
    q_emb = np.array(embeddings_model.embed_query(question))
    a_emb = np.array(embeddings_model.embed_query(answer))

    # Cosine similarity formula:
    # sim = (A · B) / (||A|| * ||B||)
    similarity = float(np.dot(q_emb, a_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(a_emb)))
    return similarity


def score_retrieval_precision(question, retrieved_chunks, ground_truth, embeddings_model):
    """
    Score how many of the top 5 chunks are actually relevant

    Logic:
    ──────
    Compare retrieved chunks to ground truth answer using embeddings:
    - Embed ground truth
    - Embed each chunk
    - If chunk similarity > 0.6, it's considered "relevant"
    - Count: how many of top 5 are relevant?
    - Precision@5 = (relevant chunks) / 5

    Args:
        question: User's question
        retrieved_chunks: List of retrieved chunk dicts
        ground_truth: The expected answer
        embeddings_model: GoogleGenerativeAIEmbeddings instance

    Returns:
        float: Precision score (0.0 - 1.0)
    """
    gt_emb = np.array(embeddings_model.embed_query(ground_truth))

    relevant_count = 0
    threshold = 0.6  # What counts as "relevant"?

    for chunk in retrieved_chunks[:5]:  # Only look at top 5
        chunk_text = chunk["chunk"]["text"]
        chunk_emb = np.array(embeddings_model.embed_query(chunk_text))

        similarity = float(np.dot(gt_emb, chunk_emb) / (np.linalg.norm(gt_emb) * np.linalg.norm(chunk_emb)))

        if similarity > threshold:
            relevant_count += 1

    return relevant_count / 5.0


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("="*80)
    print("HYBRID SEARCH vs HYBRID + COHERE RERANKING")
    print("="*80)

    # Initialize
    print("\nInitializing...")
    chunks = hybrid_module.load_and_chunk_documents()
    bm25, chunks = hybrid_module.build_bm25_index(chunks)
    index, embeddings = hybrid_module.init_pinecone()
    cohere_client = rerank_module.init_cohere()
    groq_client = Groq()

    # Storage for results
    results_hybrid = []
    results_reranked = []

    print(f"\nEvaluating {len(EVAL_QUESTIONS)} questions...\n")

    # Evaluate each question
    for i, item in enumerate(EVAL_QUESTIONS, 1):
        question = item["question"]
        ground_truth = item["ground_truth"]

        print(f"\n{'─'*80}")
        print(f"[{i}/{len(EVAL_QUESTIONS)}] {question[:60]}...")
        print(f"{'─'*80}")

        # ───────────────────────────────────────────────────────────────────────
        # METHOD 1: HYBRID SEARCH (baseline)
        # ───────────────────────────────────────────────────────────────────────
        print("\n  [Hybrid Search]", end=" ")
        answer_hybrid, contexts_hybrid = hybrid_module.hybrid_ask(
            question, bm25, chunks, index, embeddings
        )

        # Score
        f_score_h = score_faithfulness(answer_hybrid, contexts_hybrid, groq_client)
        r_score_h = score_answer_relevancy(question, answer_hybrid, embeddings)
        p_score_h = score_retrieval_precision(question, contexts_hybrid, ground_truth, embeddings)

        results_hybrid.append({
            "question": question,
            "answer": answer_hybrid,
            "faithfulness": f_score_h,
            "relevancy": r_score_h,
            "precision@5": p_score_h
        })

        print(f"F:{f_score_h:.3f} | R:{r_score_h:.3f} | P@5:{p_score_h:.3f}")

        # ───────────────────────────────────────────────────────────────────────
        # METHOD 2: HYBRID + COHERE RERANKING (improved)
        # ───────────────────────────────────────────────────────────────────────
        print("  [Hybrid + Rerank] ", end=" ")
        answer_reranked, contexts_reranked = rerank_module.hybrid_ask_with_reranking(
            question, bm25, chunks, index, embeddings, cohere_client, groq_client, hybrid_module
        )

        # Format contexts for scoring (convert to expected format)
        contexts_for_scoring = [
            {"chunk": r["chunk"]} for r in contexts_reranked
        ]

        # Score
        f_score_r = score_faithfulness(answer_reranked, contexts_for_scoring, groq_client)
        r_score_r = score_answer_relevancy(question, answer_reranked, embeddings)
        p_score_r = score_retrieval_precision(question, contexts_for_scoring, ground_truth, embeddings)

        results_reranked.append({
            "question": question,
            "answer": answer_reranked,
            "faithfulness": f_score_r,
            "relevancy": r_score_r,
            "precision@5": p_score_r
        })

        print(f"F:{f_score_r:.3f} | R:{r_score_r:.3f} | P@5:{p_score_r:.3f}")

        # Show improvement
        improvement_f = f_score_r - f_score_h
        improvement_r = r_score_r - r_score_h
        improvement_p = p_score_r - p_score_h

        print(f"\n  Improvement: F{improvement_f:+.3f} | R{improvement_r:+.3f} | P@5{improvement_p:+.3f}")

    # ═══════════════════════════════════════════════════════════════════════════
    # SUMMARY STATISTICS
    # ═══════════════════════════════════════════════════════════════════════════

    def average_scores(results_list):
        """Calculate average faithfulness, relevancy, precision"""
        faithfulness = np.mean([r["faithfulness"] for r in results_list])
        relevancy = np.mean([r["relevancy"] for r in results_list])
        precision = np.mean([r["precision@5"] for r in results_list])
        return faithfulness, relevancy, precision

    f_hybrid, r_hybrid, p_hybrid = average_scores(results_hybrid)
    f_reranked, r_reranked, p_reranked = average_scores(results_reranked)

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\n{'Metric':<20} {'Hybrid':<15} {'Hybrid+Rerank':<15} {'Improvement':<15}")
    print(f"{'-'*65}")
    print(f"{'Faithfulness':<20} {f_hybrid:.4f}{'':<9} {f_reranked:.4f}{'':<9} {f_reranked-f_hybrid:+.4f}")
    print(f"{'Answer Relevancy':<20} {r_hybrid:.4f}{'':<9} {r_reranked:.4f}{'':<9} {r_reranked-r_hybrid:+.4f}")
    print(f"{'Precision@5':<20} {p_hybrid:.4f}{'':<9} {p_reranked:.4f}{'':<9} {p_reranked-p_hybrid:+.4f}")
    print(f"{'-'*65}")
    print(f"{'Avg Score':<20} {(f_hybrid+r_hybrid+p_hybrid)/3:.4f}{'':<9} {(f_reranked+r_reranked+p_reranked)/3:.4f}{'':<9} {((f_reranked+r_reranked+p_reranked)/3-(f_hybrid+r_hybrid+p_hybrid)/3):+.4f}")
    print("="*80)

    # Save results
    results_file = os.path.join(os.path.dirname(__file__), "..", "results", "reranking_comparison.json")
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    with open(results_file, "w") as f:
        json.dump({
            "hybrid_search": {
                "results": results_hybrid,
                "avg_faithfulness": float(f_hybrid),
                "avg_relevancy": float(r_hybrid),
                "avg_precision@5": float(p_hybrid)
            },
            "hybrid_with_reranking": {
                "results": results_reranked,
                "avg_faithfulness": float(f_reranked),
                "avg_relevancy": float(r_reranked),
                "avg_precision@5": float(p_reranked)
            },
            "improvement": {
                "faithfulness": float(f_reranked - f_hybrid),
                "relevancy": float(r_reranked - r_hybrid),
                "precision@5": float(p_reranked - p_hybrid)
            }
        }, f, indent=2)

    print(f"\n✓ Results saved to {results_file}")
