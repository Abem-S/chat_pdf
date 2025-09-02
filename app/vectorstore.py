from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever

def store_chunks(chunks):
    """
    Stores the given chunks in an in-memory ChromaDB with HuggingFace embeddings.
    Returns the vectorstore instance.
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # In-memory vectorstore (no persistence)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=None  # No persistence
    )

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
    Returns a Chroma vectorstore.
    For in-memory usage, you need to provide the chunks again if needed.
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if chunks:
        return Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=None)
    
    # Empty in-memory vectorstore if no chunks provided
    return Chroma(persist_directory=None, embedding_function=embeddings)
