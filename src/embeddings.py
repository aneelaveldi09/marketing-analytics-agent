import numpy as np
from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_MODEL

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    return get_model().encode(texts, normalize_embeddings=True, show_progress_bar=True)


def embed_query(text: str) -> np.ndarray:
    return get_model().encode([text], normalize_embeddings=True)[0]
