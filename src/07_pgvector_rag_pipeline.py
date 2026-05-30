import os
import glob
import numpy as np
import psycopg2
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from groq import Groq

load_dotenv()

conn = psycopg2.connect(os.getenv("NEON_DATABASE_URL"))
conn.autocommit = True

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
client = Groq()

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MAX_CONTEXT_TOKENS = 3000
semantic_cache = []
CACHE_THRESHOLD = 0.95


def setup_table():
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute("DROP TABLE IF EXISTS docs")
        cur.execute("""
            CREATE TABLE docs (
                id SERIAL PRIMARY KEY,
                content TEXT NOT NULL,
                doc_name TEXT NOT NULL,
                page_number TEXT,
                embedding vector(3072)
            )
        """)
    print("Table ready.")


def ingest_documents():
    pdf_paths = glob.glob(os.path.join(DATA_DIR, "*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDFs found in {DATA_DIR}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)

    total = 0
    for path in pdf_paths:
        doc_name = os.path.basename(path)
        loader = PyPDFLoader(path)
        docs = loader.load()
        chunks = splitter.split_documents(docs)
        print(f"  {doc_name}: {len(docs)} pages → {len(chunks)} chunks")

        with conn.cursor() as cur:
            for chunk in chunks:
                emb = embeddings.embed_query(chunk.page_content)
                page = chunk.metadata.get("page_label", chunk.metadata.get("page", "?"))
                cur.execute(
                    "INSERT INTO docs (content, doc_name, page_number, embedding) VALUES (%s, %s, %s, %s::vector)",
                    (chunk.page_content, doc_name, str(page), emb)
                )
        total += len(chunks)

    print(f"Ingested {total} chunks total.")


# ── Retrieval helpers ─────────────────────────────────────────────────────────

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def cache_lookup(query_embedding):
    for cached_emb, cached_response in semantic_cache:
        if cosine_similarity(query_embedding, cached_emb) > CACHE_THRESHOLD:
            return cached_response
    return None


def similarity_search(query_emb, top_k=5):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT content, doc_name, page_number,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM docs
            ORDER BY similarity DESC
            LIMIT %s
        """, (query_emb.tolist(), top_k))
        return cur.fetchall()


def select_chunks_within_budget(rows):
    selected = []
    token_count = 0
    for row in rows:
        chunk_tokens = len(row[0].split()) * 1.3
        if token_count + chunk_tokens > MAX_CONTEXT_TOKENS:
            break
        selected.append(row)
        token_count += chunk_tokens
    print(f"  Using {len(selected)} of {len(rows)} chunks (~{int(token_count)} tokens)")
    return selected


def build_context(rows):
    parts = []
    for content, doc_name, page_number, _ in rows:
        parts.append(f"{content}\n[Source: {doc_name}, Page {page_number}]")
    return "\n\n---\n\n".join(parts)


# ── Full RAG pipeline ─────────────────────────────────────────────────────────

def ask(question: str) -> str:
    query_emb = np.array(embeddings.embed_query(question))

    cached = cache_lookup(query_emb)
    if cached:
        print("  ⚡ CACHE HIT — skipping retrieval + LLM")
        return cached

    rows = similarity_search(query_emb, top_k=5)

    best_score = max(r[3] for r in rows) if rows else 0
    if best_score < 0.3:
        print(f"  No relevant documents found (best score: {best_score:.4f}).")
        return "I don't have that information in the provided documents."

    source_info = {}
    for _, doc_name, page_number, _ in rows:
        source_info.setdefault(doc_name, set()).add(page_number)
    print(f"  source_info: {source_info}")

    selected = select_chunks_within_budget(rows)
    context = build_context(selected)

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

    source_lines = [
        f"- {title} (Pages: {', '.join(sorted(pages, key=lambda p: int(p) if str(p).isdigit() else p))})"
        for title, pages in source_info.items()
    ]
    final_response = response.choices[0].message.content + "\n\n📎 Sources:\n" + "\n".join(source_lines)
    semantic_cache.append((query_emb, final_response))
    return final_response


if __name__ == "__main__":
    setup_table()
    ingest_documents()

    questions = [
        "What is a mutable default argument in Python?",
        "What is a mutable default argument in Python?",
        "What is a mutable default argument in Python and why is it dangerous?",
        "What happens when you do list slicing with assignment?",
    ]

    for q in questions:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        print(f"{'─'*60}")
        answer = ask(q)
        print(answer)
