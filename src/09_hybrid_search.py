import os
import numpy as np
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from groq import Groq
import glob

load_dotenv()

# ═══════════════════════════════════════════════════════════════════════════════
# 1. LOAD AND CHUNK DOCUMENTS
# ═══════════════════════════════════════════════════════════════════════════════

def load_and_chunk_documents():
    """Load PDFs and chunk them"""
    DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
    
    # Load PDFs
    pdf_paths = glob.glob(os.path.join(DATA_DIR, "*.pdf"))
    print(f"Found {len(pdf_paths)} PDFs")
    
    all_docs = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        docs = loader.load()
        all_docs.extend(docs)
        print(f"  ✓ {os.path.basename(path)} → {len(docs)} pages")
    
    # Chunk documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(all_docs)
    
    print(f"\nTotal chunks: {len(chunks)}")
    
    # Convert to format we need: [{"text": "...", "doc_name": "...", "page": 1}, ...]
    formatted_chunks = []
    for i, chunk in enumerate(chunks):
        doc_name = os.path.basename(chunk.metadata.get("source", "unknown"))
        page_number = chunk.metadata.get("page", 0)
        
        formatted_chunks.append({
            "id": f"chunk_{i}",
            "text": chunk.page_content,
            "doc_name": doc_name,
            "page_number": page_number
        })
    
    return formatted_chunks


# ═══════════════════════════════════════════════════════════════════════════════
# 2. BUILD BM25 INDEX
# ═══════════════════════════════════════════════════════════════════════════════

def build_bm25_index(chunks):
    """
    Build BM25 index from chunks
    Returns: (BM25Okapi object, list of chunks)
    """
    # Tokenize: split text into words
    tokenized_chunks = []
    for chunk in chunks:
        # Simple tokenization: lowercase + split by whitespace
        tokens = chunk["text"].lower().split()
        tokenized_chunks.append(tokens)
    
    # Build BM25 index
    bm25 = BM25Okapi(tokenized_chunks)
    
    print(f"✓ BM25 index built for {len(chunks)} chunks")
    return bm25, chunks


def bm25_search(query: str, bm25, chunks, top_k: int = 5):
    """
    Search using BM25 (keyword matching)
    Returns: list of (chunk_dict, score) tuples
    """
    # Tokenize query same way as chunks
    tokens = query.lower().split()
    
    # Get BM25 scores for all chunks
    scores = bm25.get_scores(tokens)
    
    # Get top_k results
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        if scores[idx] > 0:  # Only include if there's a match
            results.append({
                "chunk": chunks[idx],
                "score": float(scores[idx]),
                "rank": len(results) + 1
            })
    
    return results

# ═══════════════════════════════════════════════════════════════════════════════
# 3. SEMANTIC SEARCH (PINECONE)
# ═══════════════════════════════════════════════════════════════════════════════

def init_pinecone():
    """Initialize Pinecone and embeddings"""
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index("mohit-rag")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    return index, embeddings


def semantic_search(query: str, index, embeddings, top_k: int = 5):
    """
    Search using Pinecone (semantic/embedding-based)
    Returns: list of (chunk_id, score) tuples
    """
    # Get query embedding
    query_embedding = embeddings.embed_query(query)
    
    # Search in Pinecone
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    
    # Format results
    semantic_results = []
    for i, match in enumerate(results.matches):
        semantic_results.append({
            "chunk_id": match.id,
            "score": float(match.score),
            "rank": i + 1,
            "metadata": match.metadata
        })
    
    return semantic_results


# ═══════════════════════════════════════════════════════════════════════════════
# 4. RECIPROCAL RANK FUSION (RRF)
# ═══════════════════════════════════════════════════════════════════════════════

def reciprocal_rank_fusion(bm25_results, semantic_results, k=60):
    """
    Fuse BM25 and semantic results using RRF
    score = 1/(k + rank)
    
    Returns: ranked list of fused results
    """
    # Build score dict: chunk_id -> rrf_score
    rrf_scores = {}
    
    # Add BM25 scores
    for result in bm25_results:
        chunk_id = result["chunk"]["id"]
        rank = result["rank"]
        rrf_score = 1 / (k + rank)
        if chunk_id not in rrf_scores:
            rrf_scores[chunk_id] = 0
        rrf_scores[chunk_id] += rrf_score
    
    # Add semantic scores
    for result in semantic_results:
        chunk_id = result["chunk_id"]
        # Map semantic chunk_id back to our chunk
        rank = result["rank"]
        rrf_score = 1 / (k + rank)
        if chunk_id not in rrf_scores:
            rrf_scores[chunk_id] = 0
        rrf_scores[chunk_id] += rrf_score
    
    # Sort by combined score
    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    return fused


def hybrid_search(query: str, bm25, chunks, index, embeddings, top_k: int = 5):
    """
    Hybrid search: BM25 + Semantic + RRF fusion
    Returns: top_k results
    """
    # Get both result sets
    bm25_results = bm25_search(query, bm25, chunks, top_k=10)  # Get more to fuse
    semantic_results = semantic_search(query, index, embeddings, top_k=10)
    
    # Fuse with RRF
    fused_scores = reciprocal_rank_fusion(bm25_results, semantic_results, k=60)
    
    # Map back to chunks and return top_k
    hybrid_results = []
    for chunk_id, rrf_score in fused_scores[:top_k]:
        # Find chunk details
        chunk = next((c for c in chunks if c["id"] == chunk_id), None)
        if chunk:
            hybrid_results.append({
                "chunk": chunk,
                "rrf_score": rrf_score,
                "rank": len(hybrid_results) + 1
            })
    
    return hybrid_results


# ═══════════════════════════════════════════════════════════════════════════════
# 5. FULL HYBRID RAG PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def select_chunks_within_budget(matches, max_tokens=3000):
    """Select chunks while respecting token budget"""
    selected = []
    token_count = 0
    for match in matches:
        chunk_tokens = len(match["chunk"]["text"].split()) * 1.3
        if token_count + chunk_tokens > max_tokens:
            break
        selected.append(match)
        token_count += chunk_tokens
    return selected


def build_context(matches):
    """Build context string from selected chunks"""
    parts = []
    for match in matches:
        chunk = match["chunk"]
        text = chunk["text"]
        title = chunk["doc_name"]
        page = chunk["page_number"]
        parts.append(f"{text}\n[Source: {title}, Page {page}]")
    return "\n\n---\n\n".join(parts)


def hybrid_ask(question: str, bm25, chunks, index, embeddings):
    """
    Full hybrid RAG pipeline: search + LLM generation
    Returns: (answer, contexts)
    """
    client = Groq()
    
    # Hybrid search
    hybrid_results = hybrid_search(question, bm25, chunks, index, embeddings, top_k=10)
    
    # Select chunks within token budget
    selected = select_chunks_within_budget(hybrid_results)
    
    if not selected:
        return "I don't have that information in the provided documents.", []
    
    # Build context
    context = build_context(selected)
    
    # Call LLM
    response = client.chat.completions.create(
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
    return answer, selected


# if __name__ == "__main__":
#     chunks = load_and_chunk_documents()
#     bm25, chunks = build_bm25_index(chunks)
#     index, embeddings = init_pinecone()
    
#     test_queries = [
#         "mutable default argument",
#         "list slicing assignment",
#         "dictionary iteration"
#     ]
    
#     print("\n" + "="*60)
#     print("COMPARING BM25 vs SEMANTIC SEARCH")
#     print("="*60)
    
#     for query in test_queries:
#         print(f"\n{'─'*60}")
#         print(f"Query: {query}")
#         print(f"{'─'*60}")
        
#         # BM25 results
#         print("\n  BM25 (Keyword):")
#         bm25_results = bm25_search(query, bm25, chunks, top_k=3)
#         for r in bm25_results:
#             print(f"    [{r['rank']}] Score: {r['score']:.2f} | {r['chunk']['doc_name']}")
        
#         # Semantic results
#         print("\n  Semantic (Pinecone):")
#         semantic_results = semantic_search(query, index, embeddings, top_k=3)
#         for r in semantic_results:
#             print(f"    [{r['rank']}] Score: {r['score']:.2f} | Chunk: {r['chunk_id']}")


# if __name__ == "__main__":
#     chunks = load_and_chunk_documents()
#     bm25, chunks = build_bm25_index(chunks)
#     index, embeddings = init_pinecone()
    
#     test_queries = [
#         "mutable default argument",
#         "list slicing assignment",
#         "dictionary iteration"
#     ]
    
#     print("\n" + "="*60)
#     print("HYBRID SEARCH TEST (BM25 + Semantic + RRF)")
#     print("="*60)
    
#     for query in test_queries:
#         print(f"\n{'─'*60}")
#         print(f"Query: {query}")
#         print(f"{'─'*60}")
        
#         hybrid_results = hybrid_search(query, bm25, chunks, index, embeddings, top_k=5)
#         for r in hybrid_results:
#             print(f"  [{r['rank']}] RRF Score: {r['rrf_score']:.4f} | {r['chunk']['doc_name']} (Page {r['chunk']['page_number']})")
#             print(f"      Text: {r['chunk']['text'][:80]}...")


if __name__ == "__main__":
    chunks = load_and_chunk_documents()
    bm25, chunks = build_bm25_index(chunks)
    index, embeddings = init_pinecone()
    
    # Test hybrid RAG
    test_questions = [
        "What is a mutable default argument in Python?",
        "How does list slicing with assignment work?",
        "What is dictionary iteration?"
    ]
    
    print("\n" + "="*60)
    print("HYBRID RAG PIPELINE TEST")
    print("="*60)
    
    for question in test_questions:
        print(f"\nQ: {question}")
        print(f"{'─'*60}")
        answer, contexts = hybrid_ask(question, bm25, chunks, index, embeddings)
        print(f"A: {answer}")


