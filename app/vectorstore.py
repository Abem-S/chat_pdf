# app/vectorstore.py
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def get_embeddings():
    """
    Lazily load HuggingFaceEmbeddings on CPU to avoid meta tensor errors.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},  # force CPU for PyTorch
        encode_kwargs={"device": "cpu"}  # also force CPU during batch encoding
    )

def store_chunks(chunks):
    """
    Stores chunks in an in-memory FAISS vectorstore with CPU-safe embeddings.
    """
    if not chunks:
        return None

    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

def get_bm25_retriever(chunks):
    """
    Returns BM25Retriever for keyword-based search.
    """
    if not chunks:
        return None
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = 4
    return bm25

def get_vectorstore(chunks=None):
    """
    Returns a FAISS vectorstore if chunks are provided.
    """
    if not chunks:
        return None
    embeddings = get_embeddings()
    return FAISS.from_documents(chunks, embeddings)
