#this will handle llm response and prompt
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY=os.getenv("GROQ_API_KEY")

def get_llm_chain(retriever):
    groq_model = os.getenv(
        "GROQ_MODEL_NAME",
        # Groq has decommissioned `llama3-70b-8192` (model_decommissioned 400).
        # Recommended replacement per Groq deprecations:
        "llama-3.3-70b-versatile",
    )
    llm=ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=groq_model,
    )
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        You are **InsightsPopRAG**, an AI-powered research assistant trained to answer questions based only on uploaded documents.

        Your job is to provide clear, accurate, and structured answers based **only on the provided context**.

        ---

        🧠 **User Question**:
        {question}

        ---

        📚 **Context**:
        {context}

        ---

        📝 **Answer Guidelines**:
        - Respond in a professional and structured format.
        - Base your answer strictly on the context.
        - If the answer is not found in the context, say:
        "I couldn't find relevant information in the provided documents."
        - Do NOT invent information.
        - If possible, reference the source from the context.
        - Keep answers concise but informative.

        ---

        ✅ **Answer**:
        """
        )
    return RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True,
#for citatios will be goods
)    
    