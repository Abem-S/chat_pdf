from sentence_transformers import CrossEncoder
import torch

def get_reranker():
    """
    Lazily loads the CrossEncoder model on CPU to avoid meta tensor issues.
    """
    device = "cpu"  # Force CPU
    model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker = CrossEncoder(model_name, device=device)
    return reranker

def rerank(query, docs, top_k=3):
    """
    Re-rank the list of documents based on semantic relevance to the query.
    Returns top_k reranked documents.
    """
    if not docs:
        return []

    reranker = get_reranker()
    pairs = [[query, doc.page_content] for doc in docs]
    scores = reranker.predict(pairs)

    reranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    top_docs = [doc for doc, score in reranked[:top_k]]
    return top_docs
