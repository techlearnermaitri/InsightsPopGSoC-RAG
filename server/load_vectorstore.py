import os
import time
from pathlib import Path
from dotenv import load_dotenv
from tqdm.auto import tqdm
from pinecone import Pinecone , ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")  
PINECONE_ENV="us-east-1"
PINECONE_INDEX_NAME="insightspop-rag"

os.environ["GOOGLE_API_KEY"]=GOOGLE_API_KEY
 
UPLOAD_DIR="./uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True) #automatically create the directory if it doesn't exist


#initialize pinecone instance
pc=Pinecone(api_key=PINECONE_API_KEY)
spec=ServerlessSpec(
    cloud="aws",region=PINECONE_ENV)

existing_indexes=(i["name"] for i in pc.list_indexes())


if PINECONE_INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=768 #because we use google gen ai embeddings model which return 768 dimensions, 
        metric="dotproduct",
        spec=spec
    )
    while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
        print("Waiting for index to be ready...") #relaxation time 
        time.sleep(2)

index=pc.Index(PINECONE_INDEX_NAME)