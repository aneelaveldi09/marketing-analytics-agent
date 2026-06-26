import json
import os
import faiss
import numpy as np
from src.config import INDEX_DIR

_cache: tuple | None = None


def build_index(chunks: list[dict], embeddings: np.ndarray) -> None:
    os.makedirs(INDEX_DIR, exist_ok=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype("float32"))
    faiss.write_index(index, f"{INDEX_DIR}/index.faiss")
    with open(f"{INDEX_DIR}/chunks.json", "w") as f:
        json.dump(chunks, f)
    print(f"Index saved: {index.ntotal} vectors, dim={dim}")


def _load() -> tuple:
    global _cache
    if _cache is None:
        index = faiss.read_index(f"{INDEX_DIR}/index.faiss")
        with open(f"{INDEX_DIR}/chunks.json") as f:
            chunks = json.load(f)
        _cache = (index, chunks)
    return _cache


def search(query_embedding: np.ndarray, k: int = 5) -> tuple[list[dict], list[float]]:
    index, chunks = _load()
    scores, indices = index.search(
        query_embedding.reshape(1, -1).astype("float32"), k
    )
    results = [chunks[i] for i in indices[0] if i < len(chunks)]
    return results, scores[0].tolist()


def index_exists() -> bool:
    return os.path.exists(f"{INDEX_DIR}/index.faiss")


def index_size() -> int:
    if not index_exists():
        return 0
    index, _ = _load()
    return index.ntotal
