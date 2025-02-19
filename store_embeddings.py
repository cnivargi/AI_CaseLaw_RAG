import os
import chromadb
from sentence_transformers import SentenceTransformer
from pdf_processor import process_pdf_folder  # Import text extraction function

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Load embedding model
embedding_model_name = "BAAI/bge-small-en-v1.5"  # Optimized for retrieval
embedding_model = SentenceTransformer(embedding_model_name)

# Create or load collection in ChromaDB
collection = chroma_client.get_or_create_collection(name="case_laws")

def add_pdfs_to_chromadb(pdf_folder):
    """Extracts text from PDFs, generates embeddings, and stores them in ChromaDB."""
    pdf_texts = process_pdf_folder(pdf_folder)  # Extract text from PDFs

    for pdf_name, text in pdf_texts.items():
        print(f"🔹 Storing: {pdf_name} in ChromaDB")

        # Generate embedding for text
        embedding = embedding_model.encode(text).tolist()

        # Store in ChromaDB
        collection.add(
            ids=[pdf_name],
            documents=[text],
            metadatas=[{"source": pdf_name}],
            embeddings=[embedding]
        )

if __name__ == "__main__":
    pdf_folder = "./case_laws"
    add_pdfs_to_chromadb(pdf_folder)
    print("✅ PDFs stored successfully in ChromaDB!")
