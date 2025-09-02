# app/reranker.py
from sentence_transformers import CrossEncoder

# Load once globally for CPU
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
reranker = CrossEncoder(RERANKER_MODEL_NAME, device="cpu")  # Force CPU

def rerank(query, docs, top_k=3):
    """
    Re-rank the list of documents based on semantic relevance to the query.
    Returns top_k reranked documents.
    """
    if not docs:
        return []

    pairs = [[query, doc.page_content] for doc in docs]
    scores = reranker.predict(pairs)

    # Sort documents by score descending and take top_k
    reranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    top_docs = [doc for doc, score in reranked[:top_k]]
    return top_docs
