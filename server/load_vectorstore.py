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


#load , split , embed and upsert pdf conetent
def load_vector_store(uploaded_files):
    embed_model=GoogleGenerativeAIEmbeddings(model="models/text-embedding-001")
    file_paths=[]

    #1. step1 upload 
    for file in uploaded_files:
        save_path=Path(UPLOAD_DIR)/file.filename
        with open(save_path,"wb") as f :
            f.write(file.file.read())
        file_paths.append(str(save_path))
    
    #2.split
    for file_path in file_paths:
        loader=PyPDFLoader(file_path)
        documents=loader.load()

        splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        chunks=splitter.split_documents(documents)

        texts=[chunk.page_content for chunk in chunks]
        metadata =[chunk.metadata for chunk in chunks]
        ids=[f"{Path(file_path).stem}_{i}" for i in range(chunks)]
#3.embedding 
        print("embedding and upserting chunks")
        embeddings=embed_model.embed_documents(texts)
#4.upsert
        print("upserting embedings")
        with tqdm(total=len(embeddings),desc="upserting to pinecone") as progress:
            index.upsert(vectors=zip(ids,embeddings,metadata))
            progress.update(len(embeddings))

        print(f"Finished processing for {file_path}")