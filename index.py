import os
import streamlit as st
from app.loaders import load_and_chunk_pdf
from app.vectorstore import store_chunks, get_bm25_retriever
from app.chain import build_llm_chain, retrieve_hybrid_docs, rerank_documents
from app.pdf_handler import upload_pdfs

# -----------------------------
# Streamlit page configuration
# -----------------------------
st.set_page_config(page_title="📄 Chat with PDF", layout="wide")
st.title("DocsAI: Chat with your PDF")

UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# -----------------------------
# Initialize session_state
# -----------------------------
if "chunks" not in st.session_state:
    st.session_state["chunks"] = []

if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None

if "bm25" not in st.session_state:
    st.session_state["bm25"] = None

# -----------------------------
# STEP 1: Upload PDF
# -----------------------------
pdf_file, submitted = upload_pdfs()

# -----------------------------
# STEP 2: Load + Index PDF
# -----------------------------
if pdf_file and submitted:
    file_path = os.path.join(UPLOAD_DIR, pdf_file.name)
    with open(file_path, "wb") as f:
        f.write(pdf_file.read())

    st.sidebar.success(f"Uploaded: {pdf_file.name}")

    # Load, split, and embed PDF into chunks
    with st.spinner("🔍 Indexing PDF..."):
        new_chunks = load_and_chunk_pdf(file_path)
        st.success("✅ PDF indexed successfully!")

        # Add new chunks to session_state
        st.session_state["chunks"].extend(new_chunks)

        # Build in-memory vectorstore with all chunks
        st.session_state["vectorstore"] = store_chunks(st.session_state["chunks"])
        st.session_state["bm25"] = get_bm25_retriever(st.session_state["chunks"])

# -----------------------------
# STEP 3: Ask a question
# -----------------------------
st.header("💬 Ask a question")
query = st.text_input("Enter your question")

if query:
    if not st.session_state["vectorstore"]:
        st.warning("⚠️ Please upload a PDF first.")
        st.stop()

    # STEP 4: Retrieve documents (Hybrid search)
    retrieved_docs = retrieve_hybrid_docs(query, st.session_state["vectorstore"])

    # STEP 5: Apply reranker
    reranked_docs = rerank_documents(query, retrieved_docs)

    # STEP 6: Build the chain
    chain = build_llm_chain()

    # Stream response into Streamlit
    st.subheader("🤖 Answer:")
    response_container = st.empty()
    full_response = ""

    # Pass both query and reranked docs into the chain
    response_container.markdown(chain.invoke({"question": query, "docs": reranked_docs}))

    # STEP 7: Show retrieved chunks in the sidebar
    st.sidebar.subheader("🔍 Retrieved Chunks")

    if reranked_docs:
        for i, doc in enumerate(reranked_docs):
            st.sidebar.markdown(f"**Chunk {i+1}**")
            st.sidebar.caption(doc.page_content[:400])
    else:
        st.sidebar.info("No chunks retrieved yet.")
