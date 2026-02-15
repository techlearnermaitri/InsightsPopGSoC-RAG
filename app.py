
import os
from fastapi import FastAPI , UploadFile, File , HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
import google.generativeai as genai
import json

app=FastAPI()

#react frontend to api
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

