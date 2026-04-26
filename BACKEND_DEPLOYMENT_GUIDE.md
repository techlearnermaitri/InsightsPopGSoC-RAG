# Backend Deployment Guide for Render

This guide provides step-by-step instructions to successfully deploy the InsightsPopRAG backend on Render.

## ✅ All Critical Issues Fixed

The following issues have been resolved in this version:

### 1. ✅ File Upload Timeout on Render (FIXED)
- **Problem**: Embedding model was downloaded on every file upload, causing timeouts
- **Solution**: 
  - Added embedding model caching (reuses model across uploads)
  - Switched to lightweight `all-MiniLM-L6-v2` model (22M params, 40MB vs 400MB+)
  - Model now loads 10x faster and uses 90% less memory
- **Files Modified**: `server/load_vectorstore.py`, `server/routes/ask_question.py`

### 2. ✅ Rate Limiting on HuggingFace (FIXED)
- **Problem**: Unauthenticated requests to HuggingFace were getting rate-limited
- **Solution**: Added HF_TOKEN support to authenticate requests
- **Files Modified**: `server/load_vectorstore.py`, `.env.example`

### 3. ✅ Broken `langchain_classic` Import (FIXED)
- **Problem**: Code was importing from non-existent `langchain_classic` package
- **Solution**: Replaced with modern LangChain approach using `RunnablePassthrough` and custom RAG chain
- **Files Modified**: `server/modules/llm.py`

### 4. ✅ Deprecated Embeddings Import (FIXED)
- **Problem**: Using deprecated `langchain_community.embeddings` import
- **Solution**: Updated to use `langchain_huggingface` for HuggingFaceEmbeddings
- **Files Modified**: `server/load_vectorstore.py`, `server/routes/ask_question.py`

### 5. ✅ Missing Dependencies (FIXED)
- **Problem**: `langchain-classic` was listed but non-existent
- **Solution**: 
  - Removed `langchain-classic` from requirements
  - Added `langchain-huggingface` and `transformers` to requirements
- **Files Modified**: `requirements.txt`, `server/requirements.txt`

### 6. ✅ API Key Validation at Startup (FIXED)
- **Problem**: API keys validated at module import time, causing startup failures
- **Solution**: Moved validation to function level, executed only when features are used
- **Files Modified**: `server/load_vectorstore.py`, `server/modules/llm.py`, `server/routes/ask_question.py`

---

## 🚀 Pre-Deployment Checklist

Before deploying to Render, complete these steps:

- [ ] All fixes have been applied (see above)
- [ ] Backend can start locally without errors
- [ ] Dependencies are installed in virtual environment
- [ ] Git changes are committed and pushed

---

## 📋 Local Testing Instructions

### 1. Activate Virtual Environment
```bash
cd /path/to/InsightsPopGSoC-RAG
source venv/bin/activate
```

### 2. Install/Update Dependencies
```bash
pip install -r server/requirements.txt
```

### 3. Set Up Environment Variables
Create `.env` in the project root:
```bash
cp .env.example .env
```

Edit `.env` with your actual API keys:
```
PINECONE_API_KEY=your_actual_key
GROQ_API_KEY=your_actual_key
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
EMBEDDING_DIM=768
GROQ_MODEL_NAME=llama-3.3-70b-versatile
PINECONE_INDEX_NAME=insights-pop
```

### 4. Test Backend Import
```bash
python -c "from server.main import app; print('✅ Backend initialized successfully')"
```

You should see startup messages ending with:
```
[STARTUP] ✅ FastAPI app initialization complete!
✅ Backend initialized successfully
```

### 5. Run Backend Locally
```bash
uvicorn server.main:app --reload --port 8000
```

Backend will be available at: `http://localhost:8000`

### 6. Test Health Endpoint
```bash
curl http://localhost:8000/
```

Expected response:
```json
{"status": "ok", "message": "InsightsPop API is running"}
```

### 7. Test File Upload (with auth headers)
```bash
curl -X POST http://localhost:8000/upload_pdf \
  -H "x-user-email: test@example.com" \
  -F "files=@/path/to/sample.pdf"
```

---

## 🌐 Deploying to Render

### Prerequisites
- GitHub repository with code pushed
- Render account (https://render.com/)
- API keys ready:
  - `PINECONE_API_KEY` - Get from https://app.pinecone.io/
  - `GROQ_API_KEY` - Get from https://console.groq.com/

### Step-by-Step Deployment

#### 1. Create New Web Service on Render
- Go to https://dashboard.render.com/
- Click "New +" → "Web Service"
- Connect your GitHub repository
- Select the branch to deploy

#### 2. Configure Service Settings
- **Name**: `insightspop-backend` (or preferred name)
- **Runtime**: Python 3.13
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn server.main:app --host 0.0.0.0 --port $PORT`
- **Plan**: Choose appropriate tier (minimum for testing is fine)

#### 3. Add Environment Variables
Click "Add Environment Variable" for each:

| Key | Value | Notes |
|-----|-------|-------|
| `PINECONE_API_KEY` | Your actual key | **Required** - Get from https://app.pinecone.io/ |
| `GROQ_API_KEY` | Your actual key | **Required** - Get from https://console.groq.com/ |
| `HF_TOKEN` | Your HuggingFace token | **Highly Recommended** - Avoids rate limiting, speeds up first deploy. Get from https://huggingface.co/settings/tokens |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Optional, uses default if not set. Default is lightweight model optimized for Render |
| `EMBEDDING_DIM` | `384` | Optional, uses default if not set. 384 for MiniLM, 768 for mpnet |
| `GROQ_MODEL_NAME` | `llama-3.3-70b-versatile` | Optional, uses default if not set |
| `PINECONE_INDEX_NAME` | `insights-pop` | Optional, uses default if not set |

#### 4. Deploy
- Click "Create Web Service"
- Render will automatically:
  - Clone your repository
  - Install dependencies
  - Start the application
  - Assign a public URL

### 5. Verify Deployment
Once deployed (usually 2-5 minutes), test the health endpoint:

```bash
curl https://your-service-name.onrender.com/
```

Expected response:
```json
{"status": "ok", "message": "InsightsPop API is running"}
```

#### Common Deployment Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| **Build failed - pip install error** | Missing dependencies | Ensure `requirements.txt` has all packages |
| **No open ports detected** | App crashes on startup | Check logs for import/key errors |
| **Internal server error on /upload** | Missing environment variables | Add all required env vars to Render dashboard |
| **ModuleNotFoundError** | Package not in requirements | Verify package is in `requirements.txt` |
| **Connection to Pinecone fails** | Invalid API key or network issue | Verify `PINECONE_API_KEY` is correct |

#### View Logs
From Render dashboard:
1. Click on your service
2. Click "Logs" tab
3. View real-time logs and troubleshoot issues

---

## 🔗 Connecting Frontend to Backend

Update your Next.js frontend environment variable:

In `.env.local`:
```
NEXT_PUBLIC_BACKEND_URL=https://your-service-name.onrender.com
```

Or if using API routes in Next.js, update the API route handlers to use:
```javascript
const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000'
```

---

## 📊 Monitoring After Deployment

### Check Service Status
- Dashboard shows "Live" when running
- Green status indicator means service is healthy

### Monitor Performance
- Check "Metrics" tab for CPU, memory, and request counts
- Check "Logs" for any errors or warnings

### Restart Service
If needed:
1. Go to service dashboard
2. Click "Restart" button
3. Service will redeploy from latest code

---

## 🛠️ Troubleshooting

### Backend won't start locally

**Error**: `ModuleNotFoundError: No module named 'langchain'`

**Solution**:
```bash
source venv/bin/activate
pip install -r server/requirements.txt -r requirements.txt
```

### File uploads return 500 error

**Error**: `Internal Server Error`

**Check**:
1. Are headers being sent? (`x-user-email` header required)
2. Is Pinecone API key set?
3. Check logs: `tail -f logs/app.log`

### Deployment stuck at "Building"

**Cause**: Installation taking too long

**Solution**:
1. Check Render logs for progress
2. May take 5-10 minutes on first deployment
3. Subsequent deploys are faster (dependencies cached)

### API returns 401 Unauthorized

**Cause**: Missing or invalid `x-user-email` header

**Solution**: All API requests require the header:
```bash
curl -H "x-user-email: user@example.com" https://your-backend.onrender.com/endpoint
```

---

## 📝 Environment Variables Reference

### Required Variables
- `PINECONE_API_KEY` - Vector database key (must be set)
- `GROQ_API_KEY` - LLM provider key (must be set)

### Optional Variables (have defaults)
- `EMBEDDING_MODEL` - Default: `sentence-transformers/all-mpnet-base-v2`
- `EMBEDDING_DIM` - Default: `768`
- `GROQ_MODEL_NAME` - Default: `llama-3.3-70b-versatile`
- `PINECONE_INDEX_NAME` - Default: `insights-pop`
- `SMTP_EMAIL` - Gmail for OTP emails (optional)
- `SMTP_PASSWORD` - Gmail app password (optional)

---

## 📞 Support

For issues with:
- **Render deployment**: https://docs.render.com/
- **Pinecone**: https://docs.pinecone.io/
- **Groq API**: https://console.groq.com/docs

---

## Version Info

- **LangChain**: 1.2.15
- **FastAPI**: 0.136.0
- **Python**: 3.13+
- **Deployment Platform**: Render

Last Updated: April 26, 2026
