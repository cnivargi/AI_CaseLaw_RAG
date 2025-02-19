import fitz  # PyMuPDF
import os

def extract_text_from_pdf(pdf_path):
    """Extracts text from a given PDF file."""
    doc = fitz.open(pdf_path)
    text = ""
    
    for page in doc:
        text += page.get_text("text") + "\n"
    
    return text.strip()

def process_pdf_folder(pdf_folder):
    """Processes all PDFs in a folder and extracts text."""
    pdf_texts = {}

    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, filename)
            print(f"📄 Processing: {filename}")
            text = extract_text_from_pdf(pdf_path)
            pdf_texts[filename] = text
    
    return pdf_texts

if __name__ == "__main__":
    pdf_folder = "./case_laws"  # Folder where new case laws are stored
    extracted_texts = process_pdf_folder(pdf_folder)

    for pdf, text in extracted_texts.items():
        print(f"\n🔍 {pdf} - Extracted {len(text)} characters")
