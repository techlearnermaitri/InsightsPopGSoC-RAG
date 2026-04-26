from fastapi import APIRouter, Form, Header, Depends, HTTPException
from fastapi.responses import JSONResponse
from server.modules.llm import get_llm_chain
from server.modules.query_handlers import query_chain
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pinecone import Pinecone
from pydantic import Field
from typing import List, Optional
from server.logger import logger
import os
import re
import json
from server.database import get_db

router = APIRouter()

@router.post("/ask")
async def ask_question(
    question: str = Form(...), 
    session_id: Optional[str] = Form(None),
    x_user_email: Optional[str] = Header(None), 
    db = Depends(get_db)
):
    if not x_user_email:
        raise HTTPException(status_code=401, detail="User email header missing")

    try:
        logger.info(f"User query:{question}")

        pinecone_filter = {"user_email": x_user_email}
        
        # Check for @alias mentions
        mention_match = re.search(r'@([\w\.\-]+)', question)
        if mention_match:
            alias = mention_match.group(1)
            cursor = db.cursor()
            cursor.execute("SELECT filename FROM user_files WHERE user_email = ? AND custom_name = ?", (x_user_email, alias))
            row = cursor.fetchone()
            if row:
                real_filename = row["filename"]
                # Match path formatting from PyPDFLoader
                pinecone_filter["source"] = f"uploaded_docs/{real_filename}"

        # Import get_cached_embedding_model to use cached model instead of downloading every time
        from server.load_vectorstore import get_cached_embedding_model
        
        # embed model + pinecone setup
        pinecone_api_key = os.environ.get("PINECONE_API_KEY")
        if not pinecone_api_key:
            raise ValueError(
                "Missing PINECONE_API_KEY! Please set this environment variable. "
                "On Render, add it to your Environment variables. "
                "Locally, add it to a .env file in the project root."
            )
        
        pinecone_index_name = os.environ.get("PINECONE_INDEX_NAME", "insights-pop")
        pc = Pinecone(api_key=pinecone_api_key)
        index = pc.Index(pinecone_index_name)
        
        # Use cached embedding model to avoid re-downloading
        embed_model = get_cached_embedding_model()
        embedded_query = embed_model.embed_query(question)
        
        # Apply filters
        res = index.query(vector=embedded_query, top_k=10, include_metadata=True, filter=pinecone_filter)

        docs = []
        highest_score = 0
        for match in res.get("matches", []):
            score = match.get("score", 0)
            if score > highest_score:
                highest_score = score
            docs.append(
                Document(
                    page_content=match["metadata"].get("text", ""),
                    metadata=match["metadata"]
                )
            )

        # Fallback to live internet search if context confidence is low or empty
        if highest_score < 0.4:
            try:
                from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun
                search = DuckDuckGoSearchRun()
                search_results = search.run(question)
                if search_results:
                    docs.append(
                        Document(
                            page_content=f"LIVE INTERNET SEARCH RESULTS: {search_results}",
                            metadata={"source": "Live Internet Web Search (DuckDuckGo)"}
                        )
                    )
            except Exception as e:
                logger.error(f"Duckduckgo search failed: {e}")

        class SimpleRetriever(BaseRetriever):
            tags: Optional[List[str]] = Field(default_factory=list)
            metadata: Optional[dict] = Field(default_factory=dict)

            def __init__(self, documents: List[Document]):
                super().__init__()
                self._docs = documents

            def _get_relevant_documents(self, query: str) -> List[Document]:
                return self._docs

            async def _aget_relevant_documents(self, query: str) -> List[Document]:
                return self._docs

        retriever = SimpleRetriever(docs)
        chain = get_llm_chain(retriever)
        result = query_chain(chain, question)
        
        # Deduplicate sources
        result["sources"] = list(set(result.get("sources", [])))

        # Log chat history to SQLite
        if session_id and session_id != 'null':
            cursor = db.cursor()
            cursor.execute(
                "INSERT INTO chat_messages (session_id, role, content) VALUES (?, ?, ?)",
                (session_id, 'user', question)
            )
            cursor.execute(
                "INSERT INTO chat_messages (session_id, role, content, sources) VALUES (?, ?, ?, ?)",
                (session_id, 'ai', result['response'], json.dumps(result['sources']))
            )
            db.commit()

        logger.info("query is successful")
        return result

    except Exception as e:
        logger.exception("error processing question")
        return JSONResponse(status_code=500, content={"error": str(e)})