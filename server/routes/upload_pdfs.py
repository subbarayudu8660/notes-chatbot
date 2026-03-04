from fastapi import APIRouter, File, UploadFile
from typing import List
from modules.load_vectorstore import load_vectorstore
from fastapi.responses import JSONResponse
from logger import logger


router = APIRouter()

@router.post("/upload_pdfs/")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    try:
        logger.info("Received uploaded files")
        load_vectorstore(files)
        logger.info("Files processed and vectorstore updated successfully")
        return {"message":"Files uploaded and processed successfully"}
        return 
    except Exception as e:
        logger.exception("Error uploading Pdf")
        return JSONResponse(content={"error":str(e)},status_code = 500)