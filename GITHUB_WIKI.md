# 🌟 InsightsPop - GitHub Wiki

Welcome to the technical documentation for **InsightsPop**, a production-ready Retrieval-Augmented Generation (RAG) system. This wiki provides an in-depth look at the architecture, design rationale, and technical specifications of the project.

---

## 📑 Table of Contents
1. [Technical Specifications](#1-technical-specifications)
2. [Design Decisions](#2-design-decisions)
3. [System Architecture](#3-system-architecture)
4. [API Reference](#4-api-reference)
5. [Local Setup & Installation](#5-local-setup--installation)
6. [Deployment Guide (Render)](#6-deployment-guide-render)

---

## 1. Technical Specifications

InsightsPop is built on a specific set of parameters optimized for research document analysis:

- **LLM Engine:** `llama-3.3-70b-versatile` (via Groq Cloud) for high-throughput inference.
- **Inference Parameters:** Temperature set to `0.3` to prioritize factual accuracy over creativity.
- **Embedding Model:** `sentence-transformers/all-mpnet-base-v2` (768-dimensional vectors).
- **Chunking Strategy:** 
  - **Chunk Size:** 1,000 characters.
  - **Chunk Overlap:** 200 characters (20% overlap to maintain semantic context across boundaries).
- **Retrieval Logic:**
  - **Top-K:** 10 most relevant segments retrieved per query.
  - **Similarity Metric:** Dot product (optimized for MPNet embeddings).
  - **Hallucination Mitigation:** Responses are grounded in retrieved context. If the retrieval confidence score is below `0.4`, the system falls back to DuckDuckGo Live Search to prevent knowledge cutoff hallucinations.

---

## 2. Design Decisions

### Why FastAPI?
We chose **FastAPI** for the backend due to its asynchronous nature (ASGI), allowing the system to handle concurrent PDF processing and LLM calls without blocking. Its native Pydantic integration ensures strict type safety for API requests and responses.

### Why Pinecone (Serverless)?
To achieve low-latency retrieval without the overhead of managing a local vector database, we utilized **Pinecone's Serverless** architecture. This allows for scalable vector storage with metadata filtering (e.g., filtering by `user_email`) to ensure data isolation.

### Why Supabase for File Storage?
Since Render and other PaaS providers use ephemeral file systems, local uploads are lost on every deployment or restart. **Supabase Storage** was integrated to provide persistent, cloud-based blob storage for uploaded PDFs, ensuring documents remain accessible across sessions.

### Thread-Safe PDF Processing
PDF embedding is a CPU-intensive task. We implemented a thread-safe workflow using `run_in_executor` to offload embedding generation to background threads, ensuring the main event loop remains responsive for other API users.

---

## 3. System Architecture

![InsightsPop System Architecture](architecture.png)

InsightsPop follows a decoupled full-stack architecture:

- **Frontend:** Next.js 15 (React 19) with Tailwind CSS v4.
- **Orchestration:** LangChain for managing the RAG pipeline and retrieval chains.
- **Persistence Layer:** 
  - **SQLite:** Stores user session data, chat history, and file metadata.
  - **Supabase:** Stores physical PDF files.
  - **Pinecone:** Stores high-dimensional vector embeddings.

---

## 4. API Reference

### `GET /`
**Description:** Health check and system status.
- **Response (200):** `{"status": "ok", "message": "InsightsPop API is running"}`

### `POST /ask`
**Description:** Query the RAG system based on uploaded documents.
- **Headers:** 
  - `x-user-email`: Required for data isolation.
- **Request (Multipart Form):**
  - `question` (string): The user's query.
  - `session_id` (string, optional): For chat history persistence.
- **Response (200):**
```json
{
  "response": "Based on the uploaded document 'research.pdf', the main findings are...",
  "sources": ["research.pdf"],
  "history_saved": true
}
```
- **Error (401):** `{"detail": "User email header missing"}`
- **Error (500):** `{"error": "Description of the internal failure"}`

### `POST /upload`
**Description:** Upload and process PDF documents.
- **Request (Multipart Form):**
  - `files`: List of PDF files.
  - `user_email`: The email of the uploading user.
- **Workflow:** Saves to local temp storage -> Chunks & Embeds -> Upserts to Pinecone -> Uploads to Supabase -> Deletes temp file.

---

## 5. Local Setup & Installation

### Prerequisites
- Node.js 18+ & Python 3.10+
- API Keys: Groq, Pinecone, Supabase.

### Step 1: Clone & Install
```bash
git clone https://github.com/your-username/InsightsPopGSoC-RAG.git
cd InsightsPopGSoC-RAG
# Backend
cd server && pip install -r requirements.txt
# Frontend
cd ../client && npm install
```

### Step 2: Environment
Create a `.env` in the `server/` directory:
```env
PINECONE_API_KEY=your_key
GROQ_API_KEY=your_key
SUPABASE_URL=your_url
SUPABASE_KEY=your_key
```

---

## 6. Deployment Guide (Render)

### Configuration
1. **Runtime:** Python 3.13
2. **Build Command:** `pip install -r requirements.txt`
3. **Start Command:** `uvicorn server.main:app --host 0.0.0.0 --port $PORT`

### Production Stability Features
- **Deferred Validation:** API keys are validated only when the first request is made, preventing "startup crashes" on Render due to environment variable propagation delays.
- **Resource Management:** Automatic cleanup of temporary files post-embedding to prevent disk exhaustion.
