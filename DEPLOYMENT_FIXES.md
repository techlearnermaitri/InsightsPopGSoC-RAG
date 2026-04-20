# Render Deployment Fixes - Summary

This document summarizes all the issues fixed to enable successful Render deployment.

## Issues Fixed

### 1. **Missing `email-validator` Dependency (CRITICAL)**

**Problem:**
```
ImportError: email-validator is not installed, run `pip install 'pydantic[email]'`
```

**Root Cause:**
The code uses Pydantic's `EmailStr` type for email validation, which requires the `email-validator` package. This wasn't explicitly listed as a dependency with the correct Pydantic extra.

**Fix:**
- Updated `requirements.txt` and `server/requirements.txt` to include `pydantic[email]`
- Added `email-validator` as an explicit dependency
- Created root-level `requirements.txt` so Render installs backend dependencies correctly

**Files Modified:**
- `requirements.txt` (created)
- `server/requirements.txt`

---

### 2. **Broken Import: `langchain_classic`**

**Problem:**
```python
from langchain_classic.chains import RetrievalQA
```
`langchain_classic` is not a valid package in the LangChain ecosystem.

**Root Cause:**
Incorrect import path for the `RetrievalQA` class.

**Fix:**
Changed import to:
```python
from langchain.chains import RetrievalQA
```

**Files Modified:**
- `server/modules/llm.py`

---

### 3. **API Key Validation at Module Import Time (CRITICAL)**

**Problem:**
The `load_vectorstore.py` module validated `PINECONE_API_KEY` immediately upon import:

```python
if not PINECONE_API_KEY:
    raise ValueError("Missing PINECONE_API_KEY!")
```

This caused the entire application to crash during startup before Uvicorn could bind to a port, resulting in Render's "No open ports detected" error.

**Root Cause:**
Environment variables are set through the Render dashboard, not via `.env` files. When the app imports modules, it tries to validate these keys at module-load time before they're properly configured.

**Fix:**
Moved API key validation from module-level code into function definitions:
- PINECONE_API_KEY validation moved into `get_pinecone_index()` function
- GROQ_API_KEY validation moved into `get_llm_chain()` function
- PINECONE_API_KEY validation moved into the `/ask` route handler

Now the app starts successfully and only validates keys when those features are actually used.

**Files Modified:**
- `server/load_vectorstore.py`
- `server/modules/llm.py`
- `server/routes/ask_question.py`

---

### 4. **Environment Variable Loading from `.env` Files**

**Problem:**
The code tried to load `.env` files from specific paths, but on Render:
1. `.env` files are never deployed to the server
2. Environment variables are set via the Render dashboard
3. Loading non-existent `.env` files failed silently, but caused issues with API keys

**Root Cause:**
Development-focused environment variable loading strategy doesn't work in production.

**Fix:**
Updated all `load_dotenv()` calls to gracefully handle missing `.env` files:

```python
# Try loading from .env, but don't fail if it doesn't exist
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    load_dotenv(dotenv_path=_env_path)
else:
    # Try alternative path
    _root_env_path = Path(__file__).parent.parent / ".env"
    if _root_env_path.exists():
        load_dotenv(dotenv_path=_root_env_path)
```

**Files Modified:**
- `server/load_vectorstore.py`
- `server/modules/llm.py`

---

### 5. **Typo: `PINCONE_API_KEY` (CRITICAL)**

**Problem:**
```python
PINCONE_API_KEY = os.getenv("PINECONE_API_KEY")
# ^^^^^^ Missing 'E'
```

This variable was typo'd and never used, while the code referenced `PINECONE_API_KEY`.

**Root Cause:**
Simple typo in variable name.

**Fix:**
Corrected the variable name to `PINECONE_API_KEY`.

**Files Modified:**
- `server/load_vectorstore.py`

---

### 6. **Hard-coded Values for Pinecone Index Name**

**Problem:**
`PINECONE_INDEX_NAME` was hard-coded as a string instead of being configurable via environment variable.

**Fix:**
Changed to:
```python
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "insights-pop")
```

**Files Modified:**
- `server/load_vectorstore.py`

---

## New Files Created

### 1. `.env.example`
Template file documenting all required and optional environment variables for both local development and Render deployment.

### 2. `render.yaml`
Render-native configuration file specifying:
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn server.main:app --host 0.0.0.0 --port $PORT`
- Python version: 3.13

### 3. `RENDER_DEPLOYMENT.md`
Comprehensive deployment guide including:
- Prerequisites (API keys needed)
- Step-by-step Render setup instructions
- Environment variable configuration guide
- Troubleshooting section
- Local testing instructions

---

## Environment Variables Required for Render

**REQUIRED:**
- `PINECONE_API_KEY` - Vector database API key from https://app.pinecone.io/
- `GROQ_API_KEY` - LLM API key from https://console.groq.com/

**OPTIONAL:**
- `SMTP_EMAIL` - Gmail address for OTP emails (has graceful fallback)
- `SMTP_PASSWORD` - Gmail app-specific password
- `EMBEDDING_MODEL` - (defaults to `sentence-transformers/all-mpnet-base-v2`)
- `EMBEDDING_DIM` - (defaults to `768`)
- `GROQ_MODEL_NAME` - (defaults to `llama-3.3-70b-versatile`)
- `PINECONE_INDEX_NAME` - (defaults to `insights-pop`)

---

## How to Deploy on Render

1. Ensure all fixes are committed and pushed to GitHub
2. Go to https://dashboard.render.com/
3. Create a new Web Service
4. Connect your GitHub repository
5. Configure:
   - Runtime: Python 3.13
   - Build: `pip install -r requirements.txt`
   - Start: `uvicorn server.main:app --host 0.0.0.0 --port $PORT`
6. Add environment variables (PINECONE_API_KEY, GROQ_API_KEY, etc.)
7. Deploy

---

## Testing Before Deployment

```bash
# Local testing
cp .env.example .env
# Edit .env with your actual API keys
pip install -r requirements.txt
uvicorn server.main:app --reload --port 8000
```

The health check endpoint `/` should return:
```json
{"status": "ok", "message": "InsightsPop API is running"}
```

---

## Deployment Checklist

- [ ] All fixes merged and pushed to GitHub
- [ ] `requirements.txt` includes all dependencies with correct versions
- [ ] `render.yaml` exists in root directory
- [ ] `.env.example` created with documentation
- [ ] PINECONE_API_KEY added to Render environment variables
- [ ] GROQ_API_KEY added to Render environment variables
- [ ] Optional email variables added (if using email authentication)
- [ ] Start command is correctly set in Render dashboard
- [ ] Health check endpoint `/` responds with OK
- [ ] Test a sample API call to verify integration
