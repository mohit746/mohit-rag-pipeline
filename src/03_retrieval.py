import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)


def retrieve(query: str, top_k: int = 5):
    results = db.similarity_search_with_score(query, k=top_k)
    return results

if __name__ == "__main__":
    questions = [
        "What is a list in Python?",
        "How do dictionaries work?",
        "What is the difference between a list and a tuple?",
        "How do you handle exceptions in Python?",
        "What are default arguments in functions?",
    ]
    
    for q in questions:
        print(f"\n{'='*60}")
        print(f"Query: {q}")
        print(f"{'─'*60}")
        results = retrieve(q)
        for i, (doc, score) in enumerate(results):
            print(doc.metadata)
            print(f"  Result {i+1} | score: {score:.4f}")
            print(f"  {doc.page_content[:200].strip()}")
            print()
