from sentence_transformers import CrossEncoder
import torch

RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

def get_reranker():
    """
    Lazily load CrossEncoder directly on CPU.
    Avoids meta tensor issues.
    """
    device = torch.device("cpu")
    reranker = CrossEncoder(RERANKER_MODEL_NAME, device=device)
    return reranker

def rerank_documents(query, docs, top_k=3):
    """
    Re-rank retrieved documents by semantic relevance.
    """
    if not docs:
        return []
    reranker = get_reranker()
    pairs = [[query, doc.page_content] for doc in docs]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in reranked[:top_k]]
