import os
import time
from pathlib import Path
from dotenv import load_dotenv
from tqdm.auto import tqdm
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")  

# Safeguard to prevent hard-to-read crashes
if not GOOGLE_API_KEY or not PINECONE_API_KEY:
    raise ValueError("Missing API Keys! Please check your .env file location and contents.")

PINECONE_ENV = "us-east-1"
PINECONE_INDEX_NAME = "insights-pop"

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
 
UPLOAD_DIR = "./uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True) 

# Initialize pinecone instance
pc = Pinecone(api_key=PINECONE_API_KEY)
spec = ServerlessSpec(cloud="aws", region=PINECONE_ENV)

existing_indexes = (i["name"] for i in pc.list_indexes())

if PINECONE_INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=768, 
        metric="dotproduct",
        spec=spec
    )
    while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
        print("Waiting for index to be ready...")  
        time.sleep(2)

index = pc.Index(PINECONE_INDEX_NAME)

# Load, split, embed and upsert pdf content
def load_vector_store(uploaded_files):
    embed_model = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2-preview",
        output_dimensionality=768,
        google_api_key=GOOGLE_API_KEY
    )
    file_paths = []

    # 1. Upload 
    for file in uploaded_files:
        save_path = Path(UPLOAD_DIR) / file.filename
        content = file.file.read() # Read the bytes
        with open(save_path, "wb") as f:
            f.write(content)
        file_paths.append(str(save_path))
    
    # 2. Split
    for file_path in file_paths:
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)

        texts = [chunk.page_content for chunk in chunks]
        metadata = [chunk.metadata for chunk in chunks]
        
        # FIXED BUG: Use len(chunks) instead of chunks
        ids = [f"{Path(file_path).stem}_{i}" for i in range(len(chunks))]
        
        # 3. Embedding 
        print("Embedding and upserting chunks...")
        embeddings = embed_model.embed_documents(texts)
        
        # 4. Upsert
        print("Upserting embeddings...")
        with tqdm(total=len(embeddings), desc="Upserting to Pinecone") as progress:
            # zip needs to be converted to a list for Pinecone
            vectors_to_upsert = list(zip(ids, embeddings, metadata))
            index.upsert(vectors=vectors_to_upsert)
            progress.update(len(embeddings))

        print(f"Finished processing for {file_path}")