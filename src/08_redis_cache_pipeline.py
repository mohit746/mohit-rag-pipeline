import os
import numpy as np
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from groq import Groq
from upstash_redis import Redis
import json


# This is the final RAG pipeline that ties everything together:
load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
INDEX_NAME = "mohit-rag"
index = pc.Index(INDEX_NAME)

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
client = Groq()

MAX_CONTEXT_TOKENS = 3000
CACHE_THRESHOLD = 0.95

redis = Redis(url=os.getenv("UPSTASH_REDIS_REST_URL"), 
              token=os.getenv("UPSTASH_REDIS_REST_TOKEN"))

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def cache_lookup(query_embedding):
    keys = redis.keys("cache:*")
    for key in keys:
        entry = json.loads(redis.get(key))
        cached_emb = np.array(entry["embedding"])
        if cosine_similarity(query_embedding, cached_emb) > CACHE_THRESHOLD:
            print(f"  ⚡ CACHE HIT — key: {key}")
            return entry["answer"]
    return None


# Helper function to select chunks while respecting the token budget
def select_chunks_within_budget(matches):
    selected = []
    token_count = 0
    for match in matches:
        chunk_tokens = len(match.metadata.get("text", "").split()) * 1.3
        if token_count + chunk_tokens > MAX_CONTEXT_TOKENS:
            break
        selected.append(match)
        token_count += chunk_tokens
    print(f"  Using {len(selected)} of {len(matches)} chunks (~{int(token_count)} tokens)")
    return selected

def build_context(matches):
    parts = []
    for match in matches:
        text = match.metadata.get("text", "")
        title = match.metadata.get("doc_name", "Unknown")
        page = match.metadata.get("page_number", "?")
        parts.append(f"{text}\n[Source: {title}, Page {page}]")
    return "\n\n---\n\n".join(parts)


# Full RAG pipeline function
def ask(question: str, use_cache: bool = True) -> str:
    query_emb = np.array(embeddings.embed_query(question))
    if use_cache == True:
        cached_response = cache_lookup(query_emb)
        if cached_response:
            print("  ⚡ CACHE HIT — skipping retrieval + LLM")
            return cached_response, []

    # retrieval step
    raw_results = index.query(vector=query_emb.tolist(), top_k=5, include_metadata=True)

    
    # early exit guard: if no relevant documents, return early without calling LLM
    best_score = max(m.score for m in raw_results.matches)
    if best_score < 0.3:
        print(f"  No relevant documents found (best score: {best_score:.4f}).")
        return "I don't have that information in the provided documents.", []

    source_info = {}
    for match in raw_results.matches:
        title = match.metadata.get('doc_name', 'Unknown')
        page_label = match.metadata.get('page_number', '?')
        if title not in source_info:
            source_info[title] = set()
        source_info[title].add(page_label)
    
    print(f" source_info: {source_info}")
 
    
    # build context
    results = select_chunks_within_budget(raw_results.matches)
    text_chunks = [m.metadata.get("text", "") for m in results]
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
    cache_key = f"cache:{redis.dbsize()}"
    redis.set(cache_key, json.dumps({
        "embedding": query_emb.tolist(),
        "answer": final_response
    }))
    return final_response, text_chunks
    # return final_response
    # return response.choices[0].message.content + "\n\n" + "Sources:\n" + "\n".join([f"- {title} (Page {page_label})" for title, page_label in source_info.items()])


# Test the RAG pipeline with some questions
if __name__ == "__main__":
    questions = [
        # "Who is the richest person on Earth?"
        "What is a mutable default argument in Python?",
        "What are mutable default arguments in Python programming?",
        # "What is a mutable default argument in Python and why is it dangerous?",
        # "How does dictionary comprehension work?",
        "What happens when you do list slicing with assignment?",
    ]
    
    for q in questions:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        print(f"{'─'*60}")
        answer, text_chunks = ask(q, use_cache=True)
        print(answer)
    
    # Below code is to verify the dimensions of embeddings - you can run it once and then comment it out
    # embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    test = embeddings.embed_query("hello")
    print("Dimensionality of the hello word: ", len(test))  # This is your dimension number
