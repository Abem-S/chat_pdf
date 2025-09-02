# app/reranker.py
from sentence_transformers import CrossEncoder

RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

def get_reranker():
    """
    Lazily loads the CrossEncoder model on CPU.
    Returns a CrossEncoder instance.
    """
    return CrossEncoder(RERANKER_MODEL_NAME, device="cpu")

def rerank(query, docs, top_k=3):
    """
    Re-rank documents based on semantic similarity to the query.
    Returns top_k documents.
    """
    if not docs:
        return []

    reranker = get_reranker()  # <- load model here, not at import time
    pairs = [[query, doc.page_content] for doc in docs]
    scores = reranker.predict(pairs)

    reranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    top_docs = [doc for doc, score in reranked[:top_k]]
    return top_docs
