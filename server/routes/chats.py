from fastapi import APIRouter, Depends, HTTPException, Header
import sqlite3
import uuid
import json
from server.database import get_db

router = APIRouter()

@router.post("/sessions")
async def create_session(x_user_email: str = Header(...), db=Depends(get_db)):
    """Create a new chat session"""
    session_id = str(uuid.uuid4())
    cursor = db.cursor()
    cursor.execute(
        "INSERT INTO chat_sessions (id, user_email, title) VALUES (?, ?, ?)",
        (session_id, x_user_email, "New Chat")
    )
    db.commit()
    return {"session_id": session_id, "title": "New Chat"}

@router.get("/sessions")
async def list_sessions(x_user_email: str = Header(...), db=Depends(get_db)):
    """List all chat sessions for a user"""
    cursor = db.cursor()
    cursor.execute(
        "SELECT id, title, created_at FROM chat_sessions WHERE user_email = ? ORDER BY created_at DESC",
        (x_user_email,)
    )
    sessions = cursor.fetchall()
    return [{"id": s["id"], "title": s["title"], "created_at": s["created_at"]} for s in sessions]

@router.get("/sessions/{session_id}")
async def get_session_messages(session_id: str, x_user_email: str = Header(...), db=Depends(get_db)):
    """Get all messages for a specific session"""
    cursor = db.cursor()
    
    # Verify ownership
    cursor.execute("SELECT user_email FROM chat_sessions WHERE id = ?", (session_id,))
    session = cursor.fetchone()
    if not session or session["user_email"] != x_user_email:
        raise HTTPException(status_code=403, detail="Session not found or unauthorized")

    cursor.execute(
        "SELECT role, content, sources FROM chat_messages WHERE session_id = ? ORDER BY created_at ASC",
        (session_id,)
    )
    messages = cursor.fetchall()
    
    results = []
    for msg in messages:
        sources_list = []
        try:
            if msg["sources"]:
                sources_list = json.loads(msg["sources"])
        except:
            pass
            
        results.append({
            "role": msg["role"],
            "content": msg["content"],
            "sources": sources_list
        })
        
    return results
