from fastapi import APIRouter, Depends, HTTPException, Header
from typing import List, Optional
from pydantic import BaseModel
from server.database import get_db

router = APIRouter()

class RenameRequest(BaseModel):
    old_name: str
    new_name: str

@router.get("/files")
async def get_user_files(x_user_email: Optional[str] = Header(None), db = Depends(get_db)):
    if not x_user_email:
        raise HTTPException(status_code=401, detail="User email header missing")
        
    cursor = db.cursor()
    cursor.execute("SELECT custom_name, filename, uploaded_at FROM user_files WHERE user_email = ?", (x_user_email,))
    files = cursor.fetchall()
    return [{"custom_name": f["custom_name"], "filename": f["filename"], "uploaded_at": f["uploaded_at"]} for f in files]

@router.put("/files/rename")
async def rename_file(request: RenameRequest, x_user_email: Optional[str] = Header(None), db = Depends(get_db)):
    if not x_user_email:
        raise HTTPException(status_code=401, detail="User email header missing")
        
    cursor = db.cursor()
    cursor.execute("UPDATE user_files SET custom_name = ? WHERE user_email = ? AND custom_name = ?", 
                   (request.new_name, x_user_email, request.old_name))
    db.commit()
    
    if cursor.rowcount == 0:
        raise HTTPException(status_code=404, detail="File not found")
        
    return {"message": "File renamed successfully"}
