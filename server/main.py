from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from middlewares.exception_handlers import catch_exception_middleware
from routes.upload_pdfs import router as upload_router
from routes.ask_question import router as ask_router
from server.modules.llm import get_llm_chain
from server.modules.load_vectorstore import load_vectorstore

app = FastAPI(title = "Medical assistant api",description="API for AI medical Assistant chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=["*"],
    allow_methods=["*"],
    allow_header=["*"]
)

app.middleware("http")(catch_exception_middleware)


app.include_router()

