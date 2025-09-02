# app/vectorstore.py

from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever

def get_embeddings():
    """
    Returns a HuggingFaceEmbeddings instance forced to CPU and safe for Streamlit Cloud.
    """
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},  # force CPU for PyTorch
        encode_kwargs={"device": "cpu"}  # also force CPU for batch encoding
    )

def store_chunks(chunks):
    """
    Stores the given chunks in an in-memory FAISS vectorstore with HuggingFace embeddings.
    Returns the vectorstore instance.
    """
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

def get_bm25_retriever(chunks):
    """
    Initializes and returns a BM25Retriever from text chunks.
    Useful for keyword-based retrieval in hybrid search.
    """
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = 4
    return bm25

def get_vectorstore(chunks=None):
    """
    Loads a FAISS vectorstore from chunks if provided.
    Returns None if no chunks are given.
    """
    if chunks:
        embeddings = get_embeddings()
        return FAISS.from_documents(chunks, embeddings)
    return None
