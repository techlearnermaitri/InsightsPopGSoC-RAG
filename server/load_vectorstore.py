import os
import time
import tempfile
from pathlib import Path
from dotenv import load_dotenv
from tqdm.auto import tqdm
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables — use explicit path so it works on hot-reload subprocesses
# Try loading from .env, but don't fail if it doesn't exist (Render uses env vars directly)
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    load_dotenv(dotenv_path=_env_path)
else:
    # Fallback: load from project root .env if it exists
    _root_env_path = Path(__file__).parent.parent / ".env"
    if _root_env_path.exists():
        load_dotenv(dotenv_path=_root_env_path)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

PINECONE_ENV = "us-east-1"
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "insights-pop")

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-mpnet-base-v2",
)
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))

# Absolute path so it resolves correctly regardless of CWD (Docker, Render, local).
UPLOAD_DIR = str(Path(__file__).parent.parent / "uploaded_docs")
try:
    os.makedirs(UPLOAD_DIR, exist_ok=True)
except Exception as e:
    import sys
    logger.warning(f"[UPLOAD] Failed to create upload directory: {e}")
    # Directory will be created when first needed

# Initialize pinecone instance only when needed
def get_pinecone_index():
    if not PINECONE_API_KEY:
        raise ValueError(
            "Missing PINECONE_API_KEY! Please set this environment variable. "
            "On Render, add it to your Environment variables. "
            "Locally, add it to a .env file in the project root or server/ directory."
        )
    
    pc = Pinecone(api_key=PINECONE_API_KEY)
    spec = ServerlessSpec(cloud="aws", region=PINECONE_ENV)

    existing_indexes = [i["name"] for i in pc.list_indexes()]

    if PINECONE_INDEX_NAME not in existing_indexes:
        logger.info(f"[PINECONE] Creating index: {PINECONE_INDEX_NAME} with dimension {EMBEDDING_DIM}")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="dotproduct",
            spec=spec
        )
        while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
            logger.info("[PINECONE] Waiting for index to be ready...")  
            time.sleep(2)
        logger.info("[PINECONE] ✓ Index ready")

    return pc.Index(PINECONE_INDEX_NAME)

from server.database import DB_FILE
import sqlite3


def _process_file_paths(file_paths, user_email, embed_model):
    """Shared logic: split, embed, and upsert a list of saved file paths."""
    for file_path in file_paths:
        try:
            logger.info(f"[PROCESSING] Loading PDF: {Path(file_path).name}")
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            logger.info(f"[PROCESSING] ✓ Loaded {len(documents)} pages")

            logger.info("[PROCESSING] Splitting documents into chunks...")
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(documents)
            logger.info(f"[PROCESSING] ✓ Created {len(chunks)} chunks")

            texts = [chunk.page_content for chunk in chunks]
            metadata = []
            for chunk in chunks:
                meta = chunk.metadata.copy()
                meta["text"] = chunk.page_content
                meta["user_email"] = user_email
                metadata.append(meta)

            ids = [f"{Path(file_path).stem}_{i}" for i in range(len(chunks))]

            logger.info(f"[EMBEDDINGS] Embedding {len(chunks)} chunks (this may take a minute)...")
            embeddings = embed_model.embed_documents(texts)
            logger.info(f"[EMBEDDINGS] ✓ Successfully embedded {len(embeddings)} chunks")

            logger.info("[PINECONE] Upserting to Pinecone...")
            index = get_pinecone_index()
            with tqdm(total=len(embeddings), desc="Upserting to Pinecone") as progress:
                vectors_to_upsert = list(zip(ids, embeddings, metadata))
                index.upsert(vectors=vectors_to_upsert)
                progress.update(len(embeddings))
            logger.info(f"[PINECONE] ✓ Successfully upserted to Pinecone")

            logger.info(f"[PROCESSING] ✓ Finished processing {Path(file_path).name}")
        except Exception as e:
            logger.error(f"[PROCESSING] Error processing file {Path(file_path).name}: {e}")
            raise


# ── Supabase Storage client (optional — only used when env vars are set) ─────
# If SUPABASE_URL / SUPABASE_KEY are absent (local dev), uploads are skipped.
def _get_supabase():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        return None
    try:
        from supabase import create_client
        return create_client(url, key)
    except ImportError:
        print("[WARNING] supabase package not installed — skipping cloud storage")
        return None


# ── Thread-safe version (called from async route via run_in_executor) ──────────
# Accepts pre-read bytes so UploadFile objects are never accessed inside a thread.
def load_vector_store_from_data(file_data: list, user_email: str):
    """
    file_data: list of {"filename": str, "content": bytes}
    Saves the files, logs them to SQLite, then embeds & upserts to Pinecone.
    """
    try:
        os.makedirs(UPLOAD_DIR, exist_ok=True)

        logger.info(f"[UPLOAD] Starting upload process for {len(file_data)} file(s)")
        
        # Use cached model to avoid re-downloading every time
        embed_model = get_cached_embedding_model()
        supabase = _get_supabase()
        file_paths = []

        # check_same_thread=False is required because this function runs inside
        # a ThreadPoolExecutor (via run_in_executor) — not the main thread.
        conn = sqlite3.connect(DB_FILE, check_same_thread=False)
        cursor = conn.cursor()

        for fd in file_data:
            # 1. Write to a local temp file — PyPDFLoader requires a file path on disk.
            #    On Render this is ephemeral, which is fine: we only need it for processing.
            save_path = Path(UPLOAD_DIR) / fd["filename"]
            os.makedirs(UPLOAD_DIR, exist_ok=True)
            logger.info(f"[UPLOAD] Writing {fd['filename']} ({len(fd['content'])} bytes)...")
            with open(save_path, "wb") as f:
                f.write(fd["content"])
            file_paths.append(str(save_path))
            logger.info(f"[UPLOAD] ✓ Saved {fd['filename']}")

            cursor.execute(
                "INSERT INTO user_files (user_email, custom_name, filename) VALUES (?, ?, ?)",
                (user_email, fd["filename"], fd["filename"]),
            )

        conn.commit()
        conn.close()
        logger.info("[UPLOAD] ✓ Files saved to database")

        # 2. Process: split → embed → upsert to Pinecone (this is the persistent store).
        logger.info("[UPLOAD] Starting embedding and upserting...")
        _process_file_paths(file_paths, user_email, embed_model)
        logger.info("[UPLOAD] ✓ All files processed successfully")
    except Exception as e:
        logger.error(f"[UPLOAD] Upload failed: {e}")
        raise

    # 3. Upload to Supabase Storage for file persistence across Render restarts.
    #    Runs AFTER processing so a Supabase failure doesn't block the RAG pipeline.
    if supabase:
        for fd in file_data:
            try:
                logger.info(f"[SUPABASE] Uploading {fd['filename']} to cloud storage...")
                supabase.storage.from_("pdfs").upload(
                    path=fd["filename"],
                    file=fd["content"],
                    file_options={"content-type": "application/pdf"},
                )
                logger.info(f"[SUPABASE] ✓ Uploaded {fd['filename']}")
            except Exception as e:
                logger.warning(f"[SUPABASE] Upload failed for {fd['filename']}: {e}")
    else:
        logger.info("[SUPABASE] Not configured, skipping cloud storage")

    # 4. Clean up temp files to avoid filling ephemeral disk on Render.
    logger.info("[CLEANUP] Removing temporary files...")
    for fp in file_paths:
        try:
            os.remove(fp)
            logger.debug(f"[CLEANUP] Removed {Path(fp).name}")
        except OSError:
            pass
    logger.info("[CLEANUP] ✓ Temporary files cleaned up")


# ── Legacy sync version (kept for backwards-compat if needed elsewhere) ────────
def load_vector_store(uploaded_files, user_email):
    """Legacy sync version - use load_vector_store_from_data for new code"""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    embed_model = get_cached_embedding_model()  # Use cached model
    file_paths = []
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    for file in uploaded_files:
        save_path = Path(UPLOAD_DIR) / file.filename
        content = file.file.read()
        with open(save_path, "wb") as f:
            f.write(content)
        file_paths.append(str(save_path))
        cursor.execute(
            "INSERT INTO user_files (user_email, custom_name, filename) VALUES (?, ?, ?)",
            (user_email, file.filename, file.filename),
        )

    conn.commit()
    conn.close()
    _process_file_paths(file_paths, user_email, embed_model)