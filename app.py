import streamlit as st
import os
from utils.pdf_extractor import extract_text_from_pdf, save_extracted_text

st.set_page_config(page_title="InsightPop", layout="centered")

st.title("📄 InsightPop")
st.subheader("Upload research PDFs and ask smarter questions")

uploaded_file = st.file_uploader("Upload a research PDF", type=["pdf"])

if uploaded_file:
    os.makedirs("data/uploads", exist_ok=True)
    pdf_path = os.path.join("data/uploads", uploaded_file.name)

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("PDF uploaded successfully")

    with st.spinner("Extracting full document text..."):
        extracted_text = extract_text_from_pdf(pdf_path)
        text_path = save_extracted_text(extracted_text, uploaded_file.name)

    st.success("Text extracted from entire PDF ✅")

    st.text_area(
        "Preview extracted text",
        extracted_text[:2000],
        height=300
    )

from rag_backend.chunker import chunk_text, save_chunks

with st.spinner("Chunking extracted text..."):
    chunks = chunk_text(extracted_text, chunk_size=1000, overlap=200)
    chunk_files = save_chunks(chunks, uploaded_file.name)

st.success(f"Text split into {len(chunks)} chunks ✅")
