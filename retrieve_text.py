import chromadb
from sentence_transformers import SentenceTransformer

# Define paths
chroma_db_path = "D:/GenAI2025/AI_CaseLaw_RAG/chromadb"

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path=chroma_db_path)

# Load the collection
collection_name = "case_laws"
collection = chroma_client.get_or_create_collection(name=collection_name)

# Load Sentence Transformer model for embedding queries
model = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve_case_law(query, top_k=5):
    """Retrieves top_k most relevant case law chunks based on a query."""
    query_embedding = model.encode(query).tolist()  # Convert to list format
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    # Extract and format results
    retrieved_texts = results["documents"][0] if results["documents"] else []
    metadata = results["metadatas"][0] if results["metadatas"] else []

    print("\n🔍 **Top Case Law Matches:**")
    for idx, (text, meta) in enumerate(zip(retrieved_texts, metadata)):
        print(f"\n📄 Match {idx+1}: (From {meta['filename']}, Chunk {meta['chunk_id']})")
        print(text[:500] + "...")  # Print first 500 characters for preview

if __name__ == "__main__":
    user_query = input("Enter your search query: ")
    retrieve_case_law(user_query)
