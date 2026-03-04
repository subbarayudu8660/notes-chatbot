from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from middlewares.exception_handlers import catch_exception_middleware


app = FastAPI(title = "Medical assistant api",description="API for AI medical Assistant chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=["*"],
    allow_methods=["*"],
    allow_header=["*"]
)

app.middleware("http")(catch_exception_middleware)

