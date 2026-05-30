import os
from datetime import datetime, timezone
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import importlib.util

load_dotenv()

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
INDEX_NAME = "mohit-rag"

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(INDEX_NAME)


def _load_module(filename):
    path = os.path.join(os.path.dirname(__file__), filename)
    spec = importlib.util.spec_from_file_location("embed_store", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

embed_store = _load_module("02_embed_and_store.py")
load_pdfs = embed_store.load_pdfs
strategy_recursive = embed_store.strategy_recursive

def pull_from_chroma(batch_size=100):
    collection = db._collection.get(include=["embeddings", "documents", "metadatas"])
    total = len(collection["embeddings"])
    print(f"Total embeddings in ChromaDB: {total}")
    return collection


def migrate_to_pinecone(batch_size=100):
    collection = pull_from_chroma(batch_size)
    total = len(collection["embeddings"])
    print(f"Starting migration of {total} embeddings to Pinecone...")
    vectors = []
    for i, (embedding, metadata) in enumerate(zip(collection["embeddings"], collection["metadatas"])):
        doc_id = f"doc_{i}"
        # inside the loop, append this:
        vectors.append({
            "id": doc_id,
            "values": embedding.tolist(),
            "metadata": {
                **metadata,
                "doc_name": metadata.get("title", "unknown") + ".pdf",
                "text": collection["documents"][i],
                "page_number": int(metadata.get("page_label", 0)) if str(metadata.get("page_label", "0")).isdigit() else 0,
                "date": int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp()),
                "indexed_at": datetime.now().isoformat()
            }
        })
        if len(vectors) == batch_size:
            index.upsert(vectors=vectors)
            print(f"  Upserted batch up to doc_{i}")
            vectors = []
    # after loop:
    if vectors:
        index.upsert(vectors=vectors)
        print(f"  Flushed {len(vectors)} remaining vectors")

    print("Migration completed successfully.")

def reindex_document(doc_path: str, namespace: str = ""):
    # Step 1: delete all vectors where doc_name matches doc_path
    # Step 2: load that PDF, chunk it, embed it
    # Step 3: upsert fresh vectors with new indexed_at
    doc_name = os.path.basename(doc_path)  # "Python Refresher 1.pdf" — consistent at delete + upsert
    index.delete(filter={"doc_name": doc_name}, namespace=namespace)
    loader = PyPDFLoader(doc_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
    chunks = splitter.split_documents(docs)
    new_vectors = []
    for i, chunk in enumerate(chunks):
        emb = embeddings.embed_query(chunk.page_content)
        new_vectors.append({
            "id": f"{doc_name}_chunk_{i}",
            "values": emb,
            "metadata": {
                "doc_name": doc_name,
                "text": chunk.page_content,
                "page_number": chunk.metadata.get("page_label", 0),
                "indexed_at": datetime.now().isoformat()
            }
        })
    index.upsert(vectors=new_vectors, namespace=namespace)
    print(f"Re-indexed {len(new_vectors)} chunks from {doc_path} into Pinecone.")

def test_metadata_filter():
    query_emb = embeddings.embed_query("What is a list in Python?")
    results = index.query(
        vector=query_emb,
        filter={"date": {"$gte": int(datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp())}},
        top_k=3,
        include_metadata=True
        )
    for r in results["matches"]:
        print(r["id"], r["score"], r["metadata"].get("title"))


def test_namespaces():
    # Create two namespaces
    index.upsert(vectors=[{"id": "ns1_doc1", "values": [0.1]*3072, "metadata": {"client": "client_a"}}], namespace="client_a")
    index.upsert(vectors=[{"id": "ns2_doc1", "values": [0.1]*3072, "metadata": {"client": "client_b"}}], namespace="client_b")

    # Query each namespace
    for ns in ["client_a", "client_b"]:
        results = index.query(vector=[0.1]*3072, top_k=1, namespace=ns)
        print(f"Results from {ns}: {[r['id'] for r in results['matches']]}")


if __name__ == "__main__":
    migrate_to_pinecone()
    test_metadata_filter()
    test_namespaces()
    pdf_path = os.path.join(os.path.dirname(__file__), "..", "data", "Python Refresher 1.pdf")
    reindex_document(pdf_path, namespace="")
