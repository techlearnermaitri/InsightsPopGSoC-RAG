from fastapi import APIRouter , UploadFile , File, Header, HTTPException
from typing import List, Optional
from server.load_vectorstore import load_vector_store
from fastapi.responses import JSONResponse
from server.logger import logger

router=APIRouter()

@router.post("/upload_pdf")
async def upload_pdf(files:List[UploadFile] = File (...), x_user_email: Optional[str] = Header(None)):
    if not x_user_email:
        raise HTTPException(status_code=401, detail="User email header missing")

    try:
        logger.info("Received uploaded file")
        # Added the underscore right here!
        load_vector_store(files, x_user_email)
        logger.info("document added to vectorstore") # Fixed a tiny typo here too :)
        return {"message": "files processed and vector store updated"}

    except Exception as e:
        logger.exception("error during PDF Upload") # Fixed typo here too!
        return JSONResponse(status_code=500, content={"error": str(e)})