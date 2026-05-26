"""
Day 6 — RAG Pipeline: Document Loading + Chunking Strategies

Drop your PDFs into the /data folder, then run:
    python src/01_chunking_strategies.py
"""

import os
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter
)


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


# ── Helpers ──────────────────────────────────────────────────────────────────

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


def chunk_stats(chunks: list, label: str):
    sizes = [len(c.page_content) for c in chunks]
    avg = sum(sizes) / len(sizes) if sizes else 0
    print(f"\n{'─'*60}")
    print(f"Strategy : {label}")
    print(f"  Chunks       : {len(chunks)}")
    print(f"  Avg size     : {avg:.0f} chars")
    print(f"  Min / Max    : {min(sizes)} / {max(sizes)} chars")
    print(f"\n  Sample chunk ↓\n")
    sample = chunks[len(chunks) // 2].page_content.strip()
    print("  " + sample[:400].replace("\n", "\n  "))
    print()


# ── Chunking strategies ───────────────────────────────────────────────────────

def strategy_fixed_size(docs):
    splitter = CharacterTextSplitter(
        separator=" ",
        chunk_size=500,
        chunk_overlap=50,
    )
    return splitter.split_documents(docs)


def strategy_recursive(docs):
    # Production default: tries \n\n → \n → " " → "" before hard-cutting
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
        length_function=len,
    )
    return splitter.split_documents(docs)


def strategy_paragraph(docs):
    splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=1500,       # paragraphs can be long
        chunk_overlap=100,
        is_separator_regex=False,
    )
    return splitter.split_documents(docs)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    docs = load_pdfs(DATA_DIR)

    strategies = [
        ("Fixed size (CharacterTextSplitter, 500 chars)", strategy_fixed_size),
        ("Recursive (RecursiveCharacterTextSplitter, 512 chars) ← production pick", strategy_recursive),
        ("Paragraph (split on \\n\\n)", strategy_paragraph),
    ]

    results = {}
    for label, fn in strategies:
        chunks = fn(docs)
        chunk_stats(chunks, label)
        results[label] = chunks

    print(f"\n{'='*60}")
    print("WHY RecursiveCharacterTextSplitter WINS")
    print(f"{'='*60}")
    print("""
  Fixed-size splits on a single character — it will cut mid-sentence.
  Paragraph splits preserve structure but chunks vary wildly in size.

  Recursive tries a priority list of separators:
    \\n\\n  →  \\n  →  " "  →  ""
  It only falls back to the next separator when the current chunk
  would exceed chunk_size. Result: chunks that respect natural
  boundaries (paragraphs → sentences → words) and stay within budget.

  Production settings: chunk_size=512, chunk_overlap=64 (≈12%).
  Overlap prevents context loss when an answer spans a chunk boundary.
""")

    return results


if __name__ == "__main__":
    main()
