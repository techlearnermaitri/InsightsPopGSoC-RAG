#this will handle llm response and prompt
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables — try .env files but don't fail if missing (Render uses env vars directly)
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    load_dotenv(dotenv_path=_env_path)
else:
    # Try root .env as fallback
    _root_env_path = Path(__file__).parent.parent.parent / ".env"
    if _root_env_path.exists():
        load_dotenv(dotenv_path=_root_env_path)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def get_llm_chain(retriever):
    if not GROQ_API_KEY:
        raise ValueError(
            "Missing GROQ_API_KEY! Please set this environment variable. "
            "On Render, add it to your Environment variables. "
            "Locally, add it to a .env file in the project root."
        )
    
    groq_model = os.getenv(
        "GROQ_MODEL_NAME",
        "llama-3.3-70b-versatile",
    )
    llm=ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=groq_model,
        temperature=0.3
    )
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        You are **InsightsPopRAG**, an AI-powered research assistant.

        Your primary job is to answer the user's question using the provided document **context**.
        However, if the context DOES NOT contain the answer, you are allowed to use your general knowledge and the live internet search context (if provided) to help the user.

        ---

        🧠 **User Question**:
        {question}

        ---

        📚 **Context (from uploaded documents and/or web search)**:
        {context}

        ---

        📝 **Answer Guidelines**:
        1. **Context-Driven Answers**: Prioritize extracting exact details from the provided `Context` above, and explicitly cite the filename.
        2. **Formatting (CRITICAL)**: You must use **Markdown Typography**. Never write long unstructured paragraphs. Use bold headers, numbered lists, bullet points, and data tables whenever possible to make answers extremely clean, legible, and easy for humans to read. Break up large amounts of text.
        3. **Open-Ended Collaboration**: If the user asks for brainstorms or deep analyses, DO IT automatically using your vast pre-trained LLM knowledge.
        4. **Live Web Searches**: If the context mentions "Live Internet Web Search", weave these internet facts into your answer and cite it.
        5. **Zero Complaints**: Never apologize for only having "excerpts". Just answer thoughtfully.

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
    )
    