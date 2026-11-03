# ðŸ“„ DocQuery AI â€” Intelligent Document Q&A System

> A **RAG-powered** (Retrieval-Augmented Generation) web application that lets users upload PDF documents and ask natural language questions, receiving AI-generated answers grounded in document content with **source citations**.

<a href="#"><img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white" alt="Python"></a> <a href="#"><img src="https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white" alt="FastAPI"></a> <a href="#"><img src="https://img.shields.io/badge/LangChain-0.3-green?logo=chainlink&logoColor=white" alt="LangChain"></a> <a href="#"><img src="https://img.shields.io/badge/Groq-Llama_3.3_70B-orange" alt="Groq"></a> <a href="#"><img src="https://img.shields.io/badge/FAISS-Vector_Search-purple" alt="FAISS"></a>

---

## âœ¨ Features

- **ðŸ“¤ PDF Upload & Processing** â€” Upload any PDF document (up to 20MB), auto-extracted and chunked for semantic search.
- **ðŸ§  RAG-Powered Answers** â€” Retrieves the most relevant document chunks using FAISS vector similarity search, then generates grounded answers with Groq's Llama 3.3 70B.
- **ðŸ“‘ Source Citations** â€” Every answer comes with page-level source references so you can verify the information.
- **ðŸ’¬ Multi-Turn Conversations** â€” Maintains conversation context with a sliding window memory (last 6 exchanges).
- **ðŸ›¡ï¸ Hallucination Guardrails** â€” The LLM is prompted to answer ONLY from the document context; if the information isn't there, it tells you.
- **âš¡ Sub-Second Retrieval** â€” FAISS enables near-instant semantic search across 100+ page documents.
- **ðŸŽ¨ Modern UI** â€” Clean, responsive dark theme with real-time processing feedback.

---

## ðŸ—ï¸ Architecture

```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”     â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚   Frontend   â”‚     â”‚              FastAPI Backend                 â”‚
â”‚  (HTML/CSS/  â”‚â”€â”€â”€â”€â–¶â”‚                                              â”‚
â”‚   JavaScript)â”‚     â”‚  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”  â”‚
â”‚              â”‚â—€â”€â”€â”€â”€â”‚  â”‚ PyMuPDF â”‚â”€â–¶â”‚ Text      â”‚â”€â–¶â”‚ HuggingFaceâ”‚  â”‚
â”‚  â€¢ Upload    â”‚     â”‚  â”‚ Extract â”‚  â”‚ Splitter  â”‚  â”‚ Embeddings â”‚  â”‚
â”‚  â€¢ Chat UI   â”‚     â”‚  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜  â””â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”˜  â”‚
â”‚  â€¢ Sources   â”‚     â”‚                                     â”‚        â”‚
â”‚              â”‚     â”‚  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”  â”Œâ”€â”€â”€â”€â”€â–¼â”€â”€â”€â”€â”€â”  â”‚
â”‚              â”‚     â”‚  â”‚ Groq    â”‚â—€â”€â”‚ LangChain â”‚â—€â”€â”‚   FAISS   â”‚  â”‚
â”‚              â”‚     â”‚  â”‚ LLM     â”‚  â”‚ Retrieval â”‚  â”‚ Vector DB â”‚  â”‚
â”‚              â”‚     â”‚  â”‚ (Llama) â”‚  â”‚ Chain     â”‚  â”‚           â”‚  â”‚
â”‚              â”‚     â”‚  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜  â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜     â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

---

## ðŸš€ Quick Start

### Prerequisites
- Python 3.10 or higher
- [Groq API Key](https://console.groq.com/keys) (free tier available - 30 requests/minute)

### Installation

```bash
# Clone the repository
git clone https://github.com/ParamendraVikramSingh/docquery-ai.git
cd docquery-ai

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your Groq API key
```

**Important:** Get your free Groq API key from [https://console.groq.com/keys](https://console.groq.com/keys)

### Run the Application

```bash
python main.py
```

The application will start on [http://localhost:8000](http://localhost:8000)

---

## ðŸ“¡ API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serve frontend UI |
| `GET` | `/api/health` | Health check and system status |
| `POST` | `/api/set-key` | Save Groq API key to `.env` |
| `POST` | `/api/upload` | Upload and process a PDF document |
| `GET` | `/api/status/{job_id}` | Poll background processing job status |
| `DELETE` | `/api/cancel/{job_id}` | Cancel an in-progress upload job |
| `POST` | `/api/query` | Ask a question about the uploaded document |
| `GET` | `/api/session/{id}` | Get session information |
| `DELETE` | `/api/session/{id}` | Clear a session and its data |

---

## ðŸ› ï¸ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Backend** | FastAPI | Async REST API framework |
| **LLM** | Groq (Llama 3.3 70B) | Answer generation via LPU inference |
| **Embeddings** | HuggingFace (all-MiniLM-L6-v2) | Semantic vector embeddings (384-dim) |
| **Vector Store** | FAISS | Similarity search and retrieval |
| **Orchestration** | LangChain | RAG chain, memory, prompt management |
| **PDF Parsing** | PyMuPDF (fitz) | Text extraction with page tracking |
| **Frontend** | HTML, CSS, JavaScript | Responsive chat interface |

---

## ðŸ“ Project Structure

```
docquery-ai/
â”œâ”€â”€ main.py              # FastAPI server + RAG pipeline
â”œâ”€â”€ requirements.txt     # Python dependencies
â”œâ”€â”€ .env                 # Environment variables (gitignored)
â”œâ”€â”€ .env.example         # Template for .env
â”œâ”€â”€ .gitignore
â”œâ”€â”€ LICENSE              # MIT License
â”œâ”€â”€ README.md
â”œâ”€â”€ static/
â”‚   â”œâ”€â”€ index.html       # Frontend entry point
â”‚   â”œâ”€â”€ style.css        # Dark theme styling
â”‚   â””â”€â”€ script.js        # Chat & upload logic
â”œâ”€â”€ uploads/             # Temporary PDF storage (gitignored)
â”‚   â””â”€â”€ .gitkeep
â””â”€â”€ faiss_stores/        # Persisted FAISS indices (gitignored)
    â””â”€â”€ .gitkeep
```

---

## ðŸ”‘ Key Implementation Details

### RAG Pipeline
1. **Document Ingestion**: PyMuPDF extracts text page-by-page, preserving page numbers for source tracking.
2. **Semantic Chunking**: `RecursiveCharacterTextSplitter` creates overlapping 800-char chunks with 200-char overlap to maintain context continuity.
3. **Vector Embeddings**: `all-MiniLM-L6-v2` generates 384-dimensional embeddings for each chunk.
4. **FAISS Indexing**: Embeddings are stored in a FAISS index for sub-millisecond similarity search.
5. **Retrieval**: Top-4 most semantically similar chunks are retrieved for each query.
6. **Generation**: Groq's Llama 3.3 70B generates a grounded answer using only the retrieved context.

### Hallucination Control
- The system prompt explicitly instructs the LLM to answer ONLY from the provided context.
- If information isn't found, the LLM responds with a clear "not found" message.
- Source citations allow users to verify every answer.

### Performance Optimizations
- **Lazy-loaded embeddings**: Model loads only when needed
- **Thread-safe singleton**: Prevents multiple model loads
- **Async processing**: CPU-heavy tasks run in background threads
- **Session management**: In-memory storage for development (Redis recommended for production)

---

## ðŸ”’ Security Considerations

- File size limits (20MB max)
- PDF-only uploads
- Input validation on all endpoints
- Session cleanup on shutdown
- Proper error handling and logging

---

## ðŸš§ Production Recommendations

For production deployment, consider:

1. **Replace in-memory sessions** with Redis or database storage
2. **Add authentication** and user management
3. **Implement rate limiting** to prevent abuse
4. **Use a production ASGI server** (e.g., Gunicorn with Uvicorn workers)
5. **Add monitoring** and logging aggregation
6. **Implement file scanning** for malware detection
7. **Use cloud storage** for uploaded PDFs (S3, GCS)
8. **Add HTTPS** with proper certificates
9. **Set up CORS** properly for your domain

---

## ðŸ› Troubleshooting

**Issue: "GROQ_API_KEY is not set" error**
- Make sure you've created a `.env` file from `.env.example`
- Add your Groq API key to the `.env` file
- Restart the application

**Issue: Empty answers or "not found" responses**
- Ensure your PDF has extractable text (not scanned images)
- Try rephrasing your question
- Check that the document covers the topic you're asking about

**Issue: Slow processing**
- First upload loads the embedding model (takes ~10-20 seconds)
- Large PDFs take longer to process
- Subsequent uploads are faster

---

## ðŸ“ License

This project is open source and available under the MIT License.

---

## ðŸ‘¤ Author

**Paramendra Vikram Singh**  
[GitHub](https://github.com/ParamendraVikramSingh)

---

## ðŸ¤ Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/ParamendraVikramSingh/docquery-ai/issues).

---

## â­ Show Your Support

Give a â­ï¸ if this project helped you!

