import os
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
    # retrieve
    raw_results = db.similarity_search_with_score(question, k=5)
    
    # guard
    results = select_chunks_within_budget(raw_results)
    
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
    return response.choices[0].message.content


# Test the RAG pipeline with some questions
if __name__ == "__main__":
    questions = [
        "What is a mutable default argument in Python and why is it dangerous?",
        "How does dictionary comprehension work?",
        "What happens when you do list slicing with assignment?",
    ]
    
    for q in questions:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        print(f"{'─'*60}")
        answer = ask(q)
        print(answer)
