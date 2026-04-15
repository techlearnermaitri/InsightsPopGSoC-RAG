from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from server.middlewares.exception_handlers import catch_exceptions_middleware

# Routers
from server.routes.upload_pdfs import router as upload_router
from server.routes.ask_question import router as ask_router
from server.routes.files import router as files_router
from server.routes.auth import router as auth_router
from server.routes.chats import router as chats_router

app = FastAPI(title="InsightsPopRAG", description="A Retrieval-Augmented Generation (RAG) system for insights extraction and analysis.")

# cors setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (for development/public APIs only)
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
    allow_credentials=True
)

# middle ware excpetion handling
app.middleware("http")(catch_exceptions_middleware)

# routers
app.include_router(upload_router)
app.include_router(ask_router)
app.include_router(files_router)
app.include_router(auth_router)
app.include_router(chats_router, prefix="/api/chats")
