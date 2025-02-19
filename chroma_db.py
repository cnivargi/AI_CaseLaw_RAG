import chromadb

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_db")  # Saves data persistently

# Create (or get) a collection for storing case law embeddings
case_law_collection = chroma_client.get_or_create_collection(name="case_laws")

print("✅ ChromaDB initialized and ready!")
