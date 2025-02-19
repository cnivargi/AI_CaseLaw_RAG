import os
import fitz  # PyMuPDF
import chromadb
import tiktoken

# Define paths
pdf_folder = "D:/GenAI2025/AI_CaseLaw_RAG/case_laws"
chroma_db_path = "D:/GenAI2025/AI_CaseLaw_RAG/chromadb"

# Initialize ChromaDB client (Persistent storage)
chroma_client = chromadb.PersistentClient(path=chroma_db_path)

# Create or get the collection
collection_name = "case_laws"
collection = chroma_client.get_or_create_collection(name=collection_name)

# Initialize tokenizer for chunking
tokenizer = tiktoken.get_encoding("cl100k_base")  # Using OpenAI's tokenizer for efficient chunking

def chunk_text(text, chunk_size=500):
    """Splits text into chunks of approximately 'chunk_size' tokens each."""
    tokens = tokenizer.encode(text)
    chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
    return [tokenizer.decode(chunk) for chunk in chunks]

def extract_text_from_pdf(pdf_path):
    """Extracts text from a given PDF file."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text.strip()

def store_pdfs_in_chromadb():
    """Extracts text from PDFs and stores them in ChromaDB."""
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]

    if not pdf_files:
        print("⚠️ No PDFs found in the directory!")
        return

    print(f"📄 Found {len(pdf_files)} PDFs. Processing...")

    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_folder, pdf_file)
        print(f"🔍 Extracting text from: {pdf_file}")

        text = extract_text_from_pdf(pdf_path)
        if not text:
            print(f"⚠️ No text found in {pdf_file}. Skipping...")
            continue

        chunks = chunk_text(text)
        print(f"✅ Extracted and chunked {len(chunks)} text segments from {pdf_file}")

        # Add chunks to ChromaDB
        for idx, chunk in enumerate(chunks):
            doc_id = f"{pdf_file}-{idx}"
            collection.add(
                ids=[doc_id],
                documents=[chunk],
                metadatas=[{"filename": pdf_file, "chunk_id": idx}]
            )

        print(f"✅ Stored {len(chunks)} chunks from {pdf_file} into ChromaDB!")

if __name__ == "__main__":
    store_pdfs_in_chromadb()
    print("🎉 All PDFs processed and stored successfully!")
