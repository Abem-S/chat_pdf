import os
import streamlit as st
from app.loaders import load_and_chunk_pdf
from app.vectorstore import store_chunks, get_bm25_retriever
from app.chain import build_llm_chain, retrieve_hybrid_docs
from app.pdf_handler import upload_pdfs
from app.reranker import rerank  # <- use the new CPU-safe reranker

# Streamlit page config
st.set_page_config(page_title="📄 Chat with PDF", layout="wide")
st.title("DocsAI: Chat with your PDF")

# Upload directory
UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize session state for chunks and vectorstore
if "chunks" not in st.session_state:
    st.session_state["chunks"] = []
if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None
if "bm25" not in st.session_state:
    st.session_state["bm25"] = None

# STEP 1: Upload PDF
pdf_file, submitted = upload_pdfs()

# STEP 2: Load + Index PDF if user submitted
if pdf_file and submitted:
    file_path = os.path.join(UPLOAD_DIR, pdf_file.name)
    with open(file_path, "wb") as f:
        f.write(pdf_file.read())

    st.sidebar.success(f"Uploaded: {pdf_file.name}")

    with st.spinner("🔍 Indexing PDF..."):
        new_chunks = load_and_chunk_pdf(file_path)
        st.session_state["chunks"].extend(new_chunks)

        # Build in-memory FAISS vectorstore
        st.session_state["vectorstore"] = store_chunks(st.session_state["chunks"])
        st.session_state["bm25"] = get_bm25_retriever(st.session_state["chunks"])
        st.success("✅ PDF indexed successfully!")

# STEP 3: User asks a question
st.header("💬 Ask a question")
query = st.text_input("Enter your question")

if query:
    if st.session_state["vectorstore"]:
        # STEP 4: Retrieve documents (Hybrid search)
        retrieved_docs = retrieve_hybrid_docs(query, st.session_state["vectorstore"])

        # STEP 5: Apply CPU-safe reranker
        reranked_docs = rerank(query, retrieved_docs)

        # STEP 6: Build the chain
        chain = build_llm_chain()

        # Stream response into Streamlit
        st.subheader("🤖 Answer:")
        response_container = st.empty()
        response_container.markdown(chain.invoke({"question": query, "docs": reranked_docs}))

        # STEP 7: Show retrieved chunks in the sidebar
        st.sidebar.subheader("🔍 Retrieved Chunks")
        if reranked_docs:
            for i, doc in enumerate(reranked_docs):
                st.sidebar.markdown(f"**Chunk {i+1}**")
                st.sidebar.caption(doc.page_content[:400])
        else:
            st.sidebar.info("No chunks retrieved yet.")
    else:
        st.warning("⚠️ Please upload a PDF first.")
