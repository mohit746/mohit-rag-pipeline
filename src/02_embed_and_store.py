import os
import glob
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter
)
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from 01_chunking_strategies import ...


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# ── Helpers ──────────────────────────────────────────────────────────────────
load_dotenv()

def load_pdfs(data_dir: str):
    pdf_paths = glob.glob(os.path.join(data_dir, "*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDFs found in {data_dir}. Drop your files there and re-run.")

    print(f"\n{'='*60}")
    print(f"LOADING {len(pdf_paths)} PDF(s)")
    print(f"{'='*60}")

    all_docs = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        docs = loader.load()
        all_docs.extend(docs)
        print(f"  ✓ {os.path.basename(path)}  →  {len(docs)} pages")

    total_chars = sum(len(d.page_content) for d in all_docs)
    print(f"\n  Total pages : {len(all_docs)}")
    print(f"  Total chars : {total_chars:,}")
    return all_docs

def strategy_recursive(docs):
    # Production default: tries \n\n → \n → " " → "" before hard-cutting
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
        length_function=len,
    )
    return splitter.split_documents(docs)

def create_google_embeddings():

    print("Environment loaded. Initializing model...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    # Load and chunk documents
    docs = load_pdfs(DATA_DIR)
    print(f"\nLoaded {len(docs)} pages. Now chunking...")
    chunks = strategy_recursive(docs)

    print(f"Number of chunks created: {len(chunks)}")
    # This is my vector database creation entry - with embedding, chunks and output db file
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print("Documents embedded and stored in ChromaDB successfully.")
    # Quick test to confirm it works
    query = "What are the basic topics covered in the documents?"
    matching_docs = db.similarity_search(query, k=5)

    print(f"No. of matching results: {len(matching_docs)}")  
    print(f"Top match found: {matching_docs[0].page_content}")


    return embeddings

if __name__ == "__main__":
    create_google_embeddings()