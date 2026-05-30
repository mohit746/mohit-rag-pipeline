import os
import numpy as np
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from groq import Groq

# This is the final RAG pipeline that ties everything together:
# Starting with env setup -> loading PDFs -> chunking -> embedding -> storing in ChromaDB -> retrieval -> LLM response generation with source citations
load_dotenv()
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
client = Groq()

MAX_CONTEXT_TOKENS = 3000
semantic_cache = []
CACHE_THRESHOLD = 0.95

retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.5})


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def cache_lookup(query_embedding):
    for cached_emb, cached_response in semantic_cache:
        if cosine_similarity(query_embedding, cached_emb) > CACHE_THRESHOLD:
            return cached_response
    return None

# Helper function to select chunks while respecting the token budget
def select_chunks_within_budget(chunks):
    selected = []
    token_count = 0
    for doc, score in chunks:
        chunk_tokens = len(doc.page_content.split()) * 1.3
        if token_count + chunk_tokens > MAX_CONTEXT_TOKENS:
            break
        selected.append((doc, score))
        token_count += chunk_tokens
    print(f"  Using {len(selected)} of {len(chunks)} chunks (~{int(token_count)} tokens)")
    return selected


# To build the context string for the LLM, we concatenate the selected chunks and add source citations
def build_context(chunks):
    parts = []
    for doc, score in chunks:
        title = doc.metadata.get('title', 'Unknown')
        page = doc.metadata.get('page_label', '?')
        parts.append(f"{doc.page_content}\n[Source: {title}, Page {page}]")
    return "\n\n---\n\n".join(parts)

# Full RAG pipeline function
def ask(question: str) -> str:
    query_emb = np.array(embeddings.embed_query(question))
    cached_response = cache_lookup(query_emb)
    if cached_response:
        print("  ⚡ CACHE HIT — skipping retrieval + LLM")
        return cached_response

    # retrieve direct method:
    # mmr_docs = db.max_marginal_relevance_search(question, k=5, fetch_k=20)

    # as_retriever method:
    mmr_docs = retriever.invoke(question)

    raw_results = db.similarity_search_with_score(question, k=5)

    
    # early exit guard: if no relevant documents, return early without calling LLM
    best_score = min(raw_results, key=lambda x: x[1])[1]
    if best_score > 0.75:
        print(f"  No relevant documents found (best score: {best_score:.4f}).")
        return "I don't have that information in the provided documents."
    
    # raw_results = [(doc, 0.0) for doc in mmr_docs]
    mmr_results = [(doc, 0.0) for doc in mmr_docs]
    # guard
    # results = select_chunks_within_budget(raw_results)
    results = select_chunks_within_budget(mmr_results)

    source_info = {}
    for doc, score in results:
        title = doc.metadata.get('title', 'Unknown')
        page_label = doc.metadata.get('page_label', '?')
        if title not in source_info:
            source_info[title] = set()
        source_info[title].add(page_label)
    
    print(f" source_info: {source_info}")
 
    
    # build context
    context = build_context(results)
    
    # call LLM
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": f"""You are a helpful assistant. 
            Answer the question using ONLY the context below. 
            For every fact you state, cite the source using: [Source: title, Page X] 
            If the answer is not in the context, say "I don't have that information in the provided documents."
            Context:{context}"""},
            {"role": "user", "content": question}
        ]
    )
    source_lines = [f"- {title} (Pages: {', '.join(str(p) for p in sorted(pages, key=lambda p: int(p) if str(p).isdigit() else p))})" for title, pages in source_info.items()]
    final_response = response.choices[0].message.content + "\n\n📎 Sources:\n" + "\n".join(source_lines)
    semantic_cache.append((query_emb, final_response))
    return final_response

    # return response.choices[0].message.content + "\n\n" + "Sources:\n" + "\n".join([f"- {title} (Page {page_label})" for title, page_label in source_info.items()])


# Test the RAG pipeline with some questions
if __name__ == "__main__":
    questions = [
        # "Who is the richest person on Earth?"
        "What is a mutable default argument in Python?",
        "What is a mutable default argument in Python?",
        "What is a mutable default argument in Python and why is it dangerous?",
        # "How does dictionary comprehension work?",
        "What happens when you do list slicing with assignment?",
    ]
    
    for q in questions:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        print(f"{'─'*60}")
        answer = ask(q)
        print(answer)
    
    # Below code is to verify the dimensions of embeddings - you can run it once and then comment it out
    # embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    # test = embeddings.embed_query("hello")
    # print("Dimensionality of the hello word: ", len(test))  # This is your dimension number
