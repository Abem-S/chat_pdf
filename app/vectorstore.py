# app/vectorstore.py
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def get_embeddings():
    """
    Lazily load HuggingFaceEmbeddings on CPU in a way compatible with LangChain.
    """
    # Remove extra unsupported args
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"device": "cpu"},
    )

def store_chunks(chunks):
    """
    Build FAISS vectorstore from chunks with CPU-safe embeddings.
    """
    if not chunks:
        return None
    embeddings = get_embeddings()
    return FAISS.from_documents(chunks, embeddings)

def get_bm25_retriever(chunks):
    if not chunks:
        return None
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = 4
    return bm25

def get_vectorstore(chunks=None):
    if not chunks:
        return None
    embeddings = get_embeddings()
    return FAISS.from_documents(chunks, embeddings)
