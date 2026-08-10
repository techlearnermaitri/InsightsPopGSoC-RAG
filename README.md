# ResearchInsight-AI

ResearchInsight-AI is an intelligent, end-to-end research assistant that transforms static PDF documents into an interactive, visual knowledge base. It combines advanced Retrieval-Augmented Generation (RAG) with dynamic Knowledge Graph visualization to help researchers discover hidden connections between papers.

## 🚀 Key Features

- **Smart PDF Ingestion**: Automatically extracts and "chunks" text from research papers, storing them in a persistent vector database for lightning-fast retrieval.

- **Contextual Chat**: An AI-powered chat interface that answers questions based strictly on the uploaded documents, eliminating "hallucinations" and providing factual, grounded responses.

- **Knowledge Graph Mapping**: Visually connects different research papers by automatically identifying shared concepts, keywords, and research themes.

- **API Resilience**: A "Senior-Level" backend architecture featuring API Key Rotation, ensuring the app remains functional even after hitting free-tier usage limits.

- **Persistent Memory**: Uses local storage to "remember" uploaded papers, so your research library is preserved even after a system restart.

## 🛠️ The Tech Stack

| Layer | Technology Used |
|-------|----------------|
| Frontend | React.js, Vite, React-Force-Graph (for 2D/3D visualization) |
| Backend | FastAPI (Python), Uvicorn |
| AI/LLM | Google Gemini 1.5 Flash, Google Generative AI Embeddings |
| Database | ChromaDB (Vector Storage & Persistence) |
| Orchestration | LangChain (for RAG pipeline and document processing) |

## 📋 Prerequisites

- Python 3.10+
- Node.js 18+ and npm (for frontend)
- Google API Key(s) for Gemini

## 🔧 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/InsightsPopGSoC-RAG.git
cd InsightsPopGSoC-RAG
```

### 2. Backend Setup

```bash
# Create virtual environment (recommended)
python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env  # Create .env file
# Edit .env and add your Google API key(s):
# GOOGLE_API_KEY=your_api_key_here
# GOOGLE_API_KEYS=key1,key2,key3  # For key rotation (optional)
```

### 3. Frontend Setup

```bash
cd templates/research-frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

### 4. Run the Backend

```bash
# From project root
uvicorn app:app --reload --port 8000
```

The API will be available at `http://localhost:8000`

## 📖 Usage

1. **Upload PDFs**: Use the upload interface to add research papers to your knowledge base.

2. **Ask Questions**: Use the chat interface to ask questions about your uploaded papers. The system will retrieve relevant context and provide grounded answers.

3. **Explore Connections**: View the knowledge graph to see how different papers are connected through shared concepts and keywords.

## 🔑 API Key Rotation

The system supports multiple API keys for resilience. Set multiple keys in your `.env` file:

```
GOOGLE_API_KEYS=key1,key2,key3
```

The system will automatically rotate to the next key if rate limits are hit.

## 📁 Project Structure

```
InsightsPopGSoC-RAG/
├── app.py                      # FastAPI main application
├── requirements.txt            # Python dependencies
├── rag_backend/                # RAG pipeline modules
│   ├── pdf_loader.py          # PDF processing
│   ├── chunker.py             # Text chunking
│   ├── graph_service.py       # Knowledge graph generation
│   ├── embeddings/            # Embedding utilities
│   └── vector_store/          # Vector database storage
├── utils/                      # Utility functions
│   └── pdf_extractor.py       # PDF text extraction
├── templates/                  # Frontend
│   └── research-frontend/     # React application
├── data/                       # Data storage
│   ├── uploads/               # Uploaded PDFs
│   └── chunks/                # Processed chunks
└── README.md                   # This file
```

## 🧪 API Endpoints

- `POST /upload` - Upload a PDF file
- `POST /chat?query=<your_query>` - Chat with the research assistant
- `GET /graph-data` - Get knowledge graph data
- `GET /health` - Health check endpoint

## 🐛 Troubleshooting

- **Vector database errors**: Ensure the `./rag_backend/vector_store` directory exists and is writable.
- **API key errors**: Verify your Google API key is set correctly in the `.env` file.
- **CORS errors**: Make sure the frontend URL is included in the CORS allowed origins in `app.py`.


