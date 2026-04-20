from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from server.middlewares.exception_handlers import catch_exceptions_middleware

print("[STARTUP] Starting FastAPI app initialization...")

# Routers
try:
    print("[STARTUP] Importing upload router...")
    from server.routes.upload_pdfs import router as upload_router
    print("[STARTUP] ✓ Upload router imported")
except Exception as e:
    print(f"[ERROR] Failed to import upload router: {e}")
    import traceback
    traceback.print_exc()
    raise

try:
    print("[STARTUP] Importing ask router...")
    from server.routes.ask_question import router as ask_router
    print("[STARTUP] ✓ Ask router imported")
except Exception as e:
    print(f"[ERROR] Failed to import ask router: {e}")
    import traceback
    traceback.print_exc()
    raise

try:
    print("[STARTUP] Importing files router...")
    from server.routes.files import router as files_router
    print("[STARTUP] ✓ Files router imported")
except Exception as e:
    print(f"[ERROR] Failed to import files router: {e}")
    import traceback
    traceback.print_exc()
    raise

try:
    print("[STARTUP] Importing auth router...")
    from server.routes.auth import router as auth_router
    print("[STARTUP] ✓ Auth router imported")
except Exception as e:
    print(f"[ERROR] Failed to import auth router: {e}")
    import traceback
    traceback.print_exc()
    raise

try:
    print("[STARTUP] Importing chats router...")
    from server.routes.chats import router as chats_router
    print("[STARTUP] ✓ Chats router imported")
except Exception as e:
    print(f"[ERROR] Failed to import chats router: {e}")
    import traceback
    traceback.print_exc()
    raise

print("[STARTUP] Creating FastAPI app...")
app = FastAPI(title="InsightsPopRAG", description="A Retrieval-Augmented Generation (RAG) system for insights extraction and analysis.")

print("[STARTUP] Creating FastAPI app...")
app = FastAPI(title="InsightsPopRAG", description="A Retrieval-Augmented Generation (RAG) system for insights extraction and analysis.")

# cors setup
print("[STARTUP] Setting up CORS middleware...")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (for development/public APIs only)
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
    allow_credentials=True
)

# middle ware excpetion handling
print("[STARTUP] Setting up exception handler middleware...")
app.middleware("http")(catch_exceptions_middleware)

@app.get("/")
async def health_check():
    return {"status": "ok", "message": "InsightsPop API is running"}

# routers
print("[STARTUP] Including routers...")
try:
    app.include_router(upload_router)
    print("[STARTUP] ✓ Upload router included")
    app.include_router(ask_router)
    print("[STARTUP] ✓ Ask router included")
    app.include_router(files_router)
    print("[STARTUP] ✓ Files router included")
    app.include_router(auth_router)
    print("[STARTUP] ✓ Auth router included")
    app.include_router(chats_router, prefix="/chats")
    print("[STARTUP] ✓ Chats router included")
    print("[STARTUP] ✅ FastAPI app initialization complete!")
except Exception as e:
    print(f"[ERROR] Failed to include routers: {e}")
    import traceback
    traceback.print_exc()
    raise
