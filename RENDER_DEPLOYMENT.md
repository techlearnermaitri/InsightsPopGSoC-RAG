# Render Deployment Guide for InsightsPop RAG Backend

This guide walks you through deploying the InsightsPop RAG backend to Render.

## Prerequisites

Before deploying, make sure you have:

1. **Pinecone Account** - For vector database
   - Sign up at https://app.pinecone.io/
   - Create an API key in your account settings
   - Create or note your index name (default: `insights-pop`)

2. **Groq Account** - For LLM API
   - Sign up at https://console.groq.com/
   - Create an API key in your account settings
   - Note the API key

3. **Render Account** - For hosting
   - Sign up at https://render.com/
   - Connect your GitHub account

## Step-by-Step Deployment

### 1. Push code to GitHub

```bash
git add .
git commit -m "Prepare for Render deployment"
git push origin main
```

### 2. Create a new Web Service on Render

1. Go to https://dashboard.render.com/
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Choose the repository containing this code
5. Fill in the settings:
   - **Name**: `insightspop-backend` (or your preferred name)
   - **Runtime**: Python 3.13
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn server.main:app --host 0.0.0.0 --port $PORT`
   - **Instance Type**: Free (or Starter for production)

### 3. Add Environment Variables

In the Render dashboard for your service, go to **Environment** and add the following:

#### Required Variables:
- `PINECONE_API_KEY` - Your Pinecone API key
- `GROQ_API_KEY` - Your Groq API key

#### Optional Variables:
- `SMTP_EMAIL` - Gmail address (for email verification)
- `SMTP_PASSWORD` - Gmail app password (not your regular password)
- `EMBEDDING_MODEL` - (defaults to `sentence-transformers/all-mpnet-base-v2`)
- `EMBEDDING_DIM` - (defaults to `768`)
- `GROQ_MODEL_NAME` - (defaults to `llama-3.3-70b-versatile`)
- `PINECONE_INDEX_NAME` - (defaults to `insights-pop`)

### 4. Deploy

Click the **Deploy** button. Render will:
1. Pull your code from GitHub
2. Install dependencies from `requirements.txt`
3. Start your FastAPI application on port assigned by Render

### 5. Verify Deployment

After deployment completes:

1. Go to your service URL (e.g., `https://insightspop-backend.onrender.com/`)
2. Check the health endpoint: `GET /` should return `{"status": "ok", "message": "InsightsPop API is running"}`
3. Check the logs in the Render dashboard if there are any issues

## Troubleshooting

### Error: "email-validator is not installed"
**Solution**: This should be fixed by the updated `requirements.txt` with `pydantic[email]`. If you still see this error, make sure you've pushed the latest requirements.txt and redeployed.

### Error: "No open ports detected"
**Causes and Solutions**:
1. App crashed during startup (check logs for errors)
2. Missing API keys - make sure `PINECONE_API_KEY` and `GROQ_API_KEY` are set in Environment variables
3. Wrong start command - verify it's set to `uvicorn server.main:app --host 0.0.0.0 --port $PORT`

### Error: "Missing PINECONE_API_KEY" or "Missing GROQ_API_KEY"
**Solution**: Add these environment variables in the Render dashboard under your service's Environment section.

### Database Issues
The SQLite database file is stored locally on the Render instance. Note that:
- Render instances are ephemeral - the database may be lost if the instance is restarted
- For production, consider migrating to a persistent database (PostgreSQL on Render, MongoDB Atlas, etc.)

## CORS Configuration

The backend is configured to accept requests from any origin (`allow_origins=["*"]`). For production, update this in `server/main.py` to only allow your frontend domain.

## Health Check Endpoint

Render will periodically check your service's health at:
- `GET /` 

The app is configured to return `{"status": "ok", "message": "InsightsPop API is running"}` when healthy.

## Environment Variable Setup

If you prefer not to manually set each variable in the Render dashboard, you can create a `render.yaml` file in your repository (which is already provided). This file specifies:

```yaml
services:
  - type: web
    name: insightspop-backend
    runtime: python
    pythonVersion: 3.13
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn server.main:app --host 0.0.0.0 --port $PORT
```

However, **you still need to add environment variables through the Render dashboard** - the YAML only specifies build/start commands.

## Local Testing Before Deployment

Before deploying to Render, test locally:

```bash
# Create .env file from example
cp .env.example .env

# Fill in your API keys
nano .env

# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn server.main:app --reload --port 8000
```

## Additional Resources

- [Render Python Deployment Docs](https://render.com/docs/deploy-python)
- [Render Environment Variables](https://render.com/docs/environment-variables)
- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)

## Support

If you encounter deployment issues:
1. Check the Render service logs for specific error messages
2. Verify all required environment variables are set
3. Ensure your GitHub repository is up-to-date with the latest code
4. Try manually redeploying from the Render dashboard
