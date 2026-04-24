import asyncio
from fastapi import APIRouter, UploadFile, File, Header, HTTPException
from typing import List, Optional
from server.load_vectorstore import load_vector_store_from_data
from fastapi.responses import JSONResponse
from server.logger import logger

router = APIRouter()

@router.post("/upload_pdf")
async def upload_pdf(files: List[UploadFile] = File(...), x_user_email: Optional[str] = Header(None)):
    if not x_user_email:
        raise HTTPException(status_code=401, detail="User email header missing")

    try:
        logger.info(f"Received {len(files)} file(s) for upload")

        # Read all file bytes in the async context BEFORE handing off to a thread.
        # UploadFile objects are not thread-safe; read them here first.
        file_data = []
        for file in files:
            content = await file.read()
            file_data.append({"filename": file.filename, "content": content})
            logger.info(f"  Read {file.filename} ({len(content)} bytes)")

        # Offload the CPU/IO-heavy work (HuggingFace embedding + Pinecone upsert)
        # to a thread pool so we don't block the async event loop.
        await asyncio.get_running_loop().run_in_executor(None, load_vector_store_from_data, file_data, x_user_email)

        logger.info("Documents added to vectorstore successfully")
        return {"message": "files processed and vector store updated"}

    except Exception as e:
        logger.exception("Error during PDF upload")
        return JSONResponse(status_code=500, content={"error": str(e)})