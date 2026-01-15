import streamlit as st
import os
import pymupdf as pdf
import pandas as pd
import base64


#langchian imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchian_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchai.prompts import PromptTemplate

from datetime  import datetime

#text chunking imports
