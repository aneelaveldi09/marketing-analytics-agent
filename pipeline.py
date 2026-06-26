"""Build the FAISS index from Wikipedia articles. Run this once before launching the app."""
import sys
sys.path.insert(0, ".")

from src.data_loader import load_corpus
from src.embeddings import embed_texts
from src.vector_store import build_index


def main(use_cache: bool = True):
    print("Step 1: Fetching marketing analytics articles from Wikipedia...")
    chunks = load_corpus(use_cache=use_cache)
    if not chunks:
        print("No chunks fetched. Check your internet connection and try again.")
        return
    print(f"\nStep 2: Embedding {len(chunks)} chunks with sentence-transformers...")
    embeddings = embed_texts([c["text"] for c in chunks])
    print("\nStep 3: Building FAISS index...")
    build_index(chunks, embeddings)
    print("\nDone. Run: streamlit run app.py")


if __name__ == "__main__":
    force = "--force" in sys.argv
    main(use_cache=not force)
