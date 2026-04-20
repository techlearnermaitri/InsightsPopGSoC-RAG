import os
import time
from pathlib import Path
from dotenv import load_dotenv
from tqdm.auto import tqdm
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

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
 
UPLOAD_DIR = "./uploaded_docs"
try:
    os.makedirs(UPLOAD_DIR, exist_ok=True)
except Exception as e:
    import sys
    print(f"[WARNING] Failed to create upload directory: {e}", file=sys.stderr)
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
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="dotproduct",
            spec=spec
        )
        while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
            print("Waiting for index to be ready...")  
            time.sleep(2)

    return pc.Index(PINECONE_INDEX_NAME)

from server.database import DB_FILE
import sqlite3

# Load, split, embed and upsert pdf content
def load_vector_store(uploaded_files, user_email):
    # Ensure upload directory exists before saving files
    try:
        os.makedirs(UPLOAD_DIR, exist_ok=True)
    except Exception as e:
        raise ValueError(f"Failed to create upload directory: {e}")
    
    # Local embeddings to avoid Gemini quota issues during upload.
    embed_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
    )
    file_paths = []
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # 1. Upload 
    for file in uploaded_files:
        save_path = Path(UPLOAD_DIR) / file.filename
        content = file.file.read() # Read the bytes
        with open(save_path, "wb") as f:
            f.write(content)
        file_paths.append(str(save_path))
        
        # Save to SQLite DB
        cursor.execute(
            "INSERT INTO user_files (user_email, custom_name, filename) VALUES (?, ?, ?)",
            (user_email, file.filename, file.filename)
        )
    
    conn.commit()
    conn.close()
    
    # 2. Split
    for file_path in file_paths:
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)

        texts = [chunk.page_content for chunk in chunks]
        metadata = []
        for chunk in chunks:
            meta = chunk.metadata.copy()
            meta["text"] = chunk.page_content
            meta["user_email"] = user_email
            metadata.append(meta)
        
        # FIXED BUG: Use len(chunks) instead of chunks
        ids = [f"{Path(file_path).stem}_{i}" for i in range(len(chunks))]
        
        # 3. Embedding 
        print("Embedding and upserting chunks...")
        embeddings = embed_model.embed_documents(texts)
        
        # 4. Upsert
        print("Upserting embeddings...")
        index = get_pinecone_index()
        with tqdm(total=len(embeddings), desc="Upserting to Pinecone") as progress:
            # zip needs to be converted to a list for Pinecone
            vectors_to_upsert = list(zip(ids, embeddings, metadata))
            index.upsert(vectors=vectors_to_upsert)
            progress.update(len(embeddings))

        print(f"Finished processing for {file_path}")