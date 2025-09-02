# 📄 DocsAI: Chat with Your PDF

[**Live Demo**](https://2pdfchat.streamlit.app/)

DocsAI is a Streamlit web application that allows you to upload PDF documents and interact with them using natural language queries. It leverages **FAISS** for vector storage, **HuggingFace embeddings** for semantic understanding, and optionally **BM25** for hybrid search.

---

## 🚀 Features

- Upload PDF files and automatically split them into searchable chunks.
- Store embeddings in a FAISS vectorstore for fast semantic search.
- Ask questions about your PDFs and get relevant answers.
- Optional hybrid search using BM25 for keyword matching.
- View retrieved chunks for transparency in the sidebar.

---

## 🛠 Tech Stack

- **Frontend:** Streamlit
- **Vector Store:** FAISS
- **Embeddings:** HuggingFace `sentence-transformers/all-MiniLM-L6-v2`
- **PDF Parsing:** PyPDF2
- **Retrieval:** LangChain
- **Reranking (optional):** BM25Retriever

---

## 📁 Project Structure
```
chat_pdf/
├── app/
│   ├── chain.py           # Builds the LLM chain for answering questions
│   ├── config.py          # Configuration settings, e.g., API keys
│   ├── loaders.py         # Loads PDFs and splits into chunks
│   ├── pdf_handler.py     # Handles PDF uploads
│   ├── reranker.py        # Optional BM25 reranker
│   └── vectorstore.py     # Handles FAISS vectorstore
├── uploaded_files/        # Stores user-uploaded PDFs
├── chroma_store/          # (Optional) If you were using Chroma earlier; can be removed with FAISS
├── index.py               # Main Streamlit app
├── requirements.txt       # Python dependencies
├── README.md              # Project README
```

## 🛠 Setup Instructions

### 1. Create a Virtual Environment

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up GROQ API Key

To use this RAG chatbot, you'll need a GROQ API key.

#### Get Your API Key

1. Visit [console.groq.com/keys](https://console.groq.com/keys)
2. Sign up or log in to your GROQ account
3. Create a new API key
4. Copy the generated API key

#### Set Environment Variable

**Windows (Command Prompt)**

```cmd
set GROQ_API_KEY=your_api_key_here
```

**Windows (PowerShell)**

```powershell
$env:GROQ_API_KEY="your_api_key_here"
```

**macOS / Linux**

```bash
export GROQ_API_KEY="your_api_key_here"
```

**Alternative: Create a .env file**

You can also create a `.env` file in the project root directory:

```
GROQ_API_KEY=your_api_key_here


📝 Usage

1. Run the Streamlit app:
```bash
streamlit run index.py
```
2. Upload a PDF using the sidebar.
3. Wait for the PDF to be processed and indexed.
4. Enter your question in the text input box.
5. View the answer and relevant chunks in the sidebar.

🔧 Configuration

Embedding model: sentence-transformers/all-MiniLM-L6-v2 (configured in vectorstore.py)

FAISS index persistence: uploaded_files/faiss_index (or in-memory for temporary use)

Optional hybrid search via BM25 is configurable in reranker.py.

📌 Notes

Chroma is no longer required; FAISS is used for vector storage.

Ensure uploaded PDFs are not password-protected.

Large PDFs may take longer to index.

🌐 Live Demo

Try the app live here: https://2pdfchat.streamlit.app/
