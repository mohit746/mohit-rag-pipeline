import streamlit as st
import os
import sys
from dotenv import load_dotenv

load_dotenv()

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

st.set_page_config(page_title="Healthcare RAG Pipeline", page_icon="🏥", layout="wide")
st.title("🏥 Healthcare RAG Pipeline")
st.caption("Hybrid Search (BM25 + Semantic) + Cohere Reranking | github.com/mohit746/mohit-rag-pipeline")

# Sidebar - method selector
st.sidebar.header("Retrieval Method")
method = st.sidebar.radio("Choose:", [
    "Baseline RAG",
    "Hybrid Search (BM25 + Semantic)",
    "Hybrid + Cohere Rerank ⭐"
])

st.sidebar.markdown("---")
st.sidebar.markdown("**Pipeline**")
st.sidebar.markdown("ChromaDB → Pinecone → BM25 + RRF → Cohere Rerank")

# Initialize session state for models
@st.cache_resource
def load_models():
    """Load all models once"""
    import importlib.util

    # Import hybrid search
    spec = importlib.util.spec_from_file_location(
        "hybrid_search", os.path.join(os.path.dirname(__file__), "src/09_hybrid_search.py")
    )
    hybrid_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hybrid_module)

    # Import reranking
    spec2 = importlib.util.spec_from_file_location(
        "reranking", os.path.join(os.path.dirname(__file__), "src/11_cohere_reranking.py")
    )
    reranking_module = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(reranking_module)

    # Import baseline RAG
    spec3 = importlib.util.spec_from_file_location(
        "rag", os.path.join(os.path.dirname(__file__), "src/04_rag_pipeline.py")
    )
    rag_module = importlib.util.module_from_spec(spec3)
    spec3.loader.exec_module(rag_module)

    from groq import Groq

    print("Loading models...")
    chunks = hybrid_module.load_and_chunk_documents()
    bm25, chunks = hybrid_module.build_bm25_index(chunks)
    index, embeddings = hybrid_module.init_pinecone()
    cohere_client = reranking_module.init_cohere()
    groq_client = Groq()

    return {
        "chunks": chunks,
        "bm25": bm25,
        "index": index,
        "embeddings": embeddings,
        "cohere_client": cohere_client,
        "groq_client": groq_client,
        "hybrid_search": hybrid_module.hybrid_search,
        "hybrid_search_with_reranking": reranking_module.hybrid_search_with_reranking,
        "ask": rag_module.ask
    }

# Main query
query = st.text_input("Ask a healthcare question:",
                       placeholder="e.g. What are early symptoms of diabetes?")

if st.button("Search", type="primary") and query:
    try:
        with st.spinner("Loading models..."):
            models = load_models()

        with st.spinner("Retrieving..."):
            if method == "Baseline RAG":
                answer = models["ask"](query)
                st.markdown("### Answer")
                st.write(answer)

            elif method == "Hybrid Search (BM25 + Semantic)":
                results = models["hybrid_search"](
                    query=query,
                    bm25=models["bm25"],
                    chunks=models["chunks"],
                    index=models["index"],
                    embeddings=models["embeddings"],
                    top_k=5
                )
                st.markdown("### Top 5 Results")
                for i, result in enumerate(results, 1):
                    with st.expander(f"Result {i} — RRF Score: {result['rrf_score']:.3f}"):
                        st.write(result["chunk"]["text"])

            else:  # Hybrid + Rerank
                results = models["hybrid_search_with_reranking"](
                    query=query,
                    bm25=models["bm25"],
                    chunks=models["chunks"],
                    index=models["index"],
                    embeddings=models["embeddings"],
                    client=models["cohere_client"],
                    retrieve_k=20,
                    final_k=5,
                    hybrid_search_func=models["hybrid_search"]
                )
                st.markdown("### Top 5 After Reranking")
                for i, doc in enumerate(results, 1):
                    score = doc.get("cohere_score", 0.0)
                    with st.expander(f"Result {i} — Cohere Score: {score:.3f}"):
                        st.write(doc["chunk"]["text"])

    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("Make sure all required APIs are configured in .env (Pinecone, Cohere, Groq)")

# Comparison table at bottom
st.markdown("---")
st.markdown("### Method Comparison")
st.table({
    "Method": ["Baseline RAG", "Hybrid Search", "Hybrid + Rerank"],
    "Precision@5": ["~0.60", "~0.72", "~0.85"],
    "Latency": ["Fast", "Medium", "Medium + rerank"],
    "Best For": ["Simple queries", "Keyword + concept", "High-stakes retrieval"]
})
