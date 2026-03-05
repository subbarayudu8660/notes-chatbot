from fastapi import APIRouter,Form
from fastapi.responses import JSONResponse
from modules.llm import get_llm_chain
from modules.queryhandler import query_chain
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
from pydantic import Field
from typing import List,Optional
from server.logger import logger
import os

router = APIRouter()
@router.post("/ask/")
async def ask_question(question:str = Form(...)):
    try:
        logger.info(f"User query: {question}")

        pc = Pinecone(api_key = os.environ("PINECONE_API_KEY"))
        index = pc.Index(os.environ("PINECONE_INDEX_NAME"))
        embeddings_model = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")
        question_vector = embeddings_model.embed_query(question)
        res = index.query(vector = question_vector,top_k = 3,include_metadata = True)

        documents = [
            Document(
                page_content = match["metadata"].get("text",""),
                metadata = match["metadata"]
            )for match in res["matches"]
        ]

        class SimpleRetriever(BaseRetriever):
            tags = Optional[List[str]] = Field(default_factory=list)
            metadata = Optional[dict] = Field(default_factory=dict)
            def __init__(self,documents:List[Document]):
                super().__init__()
                self._documents = documents

            def get_relevant_documents(self,query:str) -> List[Document]:
                return self._documents
            
            retriever = SimpleRetriever(documents)
            chain = get_llm_chain(retriever)
            result = query_chain(chain,question)

            logger.info("Query is succesfully")
            return result


    except Exception as e:
        logger.exception("Error processing question")
        return JSONResponse(status_code=500,content = {"error":str(e)})