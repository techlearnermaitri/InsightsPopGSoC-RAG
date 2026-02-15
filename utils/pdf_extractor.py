#pdf extractor
import fitz  # PyMuPDF
import os

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts full text from a PDF file (all pages).
    """
    doc = fitz.open(pdf_path)
    full_text = []

    for page in doc:
        text = page.get_text()
        if text.strip():
            full_text.append(text)

    doc.close()
    return "\n".join(full_text)


def save_extracted_text(text: str, filename: str):
    output_dir = "data/extracted_text"
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"{filename}.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

    return output_path





