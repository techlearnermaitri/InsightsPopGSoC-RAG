from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from server.middlewares.exception_handlers import catch_exceptions_middleware
from server.routes.upload_pdfs import router as upload_router
from server.routes.ask_question import router as ask_router



#This middleware is essential for enabling communication between a
#frontend web application and a backend API when they are hosted on 
# different domains, ports, or protocols
from fastapi.middleware.cors import CORSMiddleware


app=FastAPI(title="InsightsPopRAG", description="A Retrieval-Augmented Generation (RAG) system for insights extraction and analysis.")



# cors setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (for development/public APIs only)
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
    allow_credentials=True # Allow cookies and authentication credentials (Note: conflicts with allow_origins=["*"] for browser security)
)

#middle ware excpetion handling

app.middleware("http")(catch_exceptions_middleware)

#routers

#1.ypload pdfs
app.include_router(upload_router)

#2. asking queries

app.include_router(ask_router)

