import chromadb
from sentence_transformers import SentenceTransformer

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="case_laws")

# Load the same embedding model used for storage
embedding_model_name = "BAAI/bge-small-en-v1.5"
embedding_model = SentenceTransformer(embedding_model_name)

def retrieve_case_laws(query, top_k=3):
    """Fetches the most relevant case laws for a given query."""
    query_embedding = embedding_model.encode(query).tolist()

    # Search ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    matched_docs = results["documents"][0]
    matched_metadata = results["metadatas"][0]

    print("\n🔍 **Search Results:**")
    for idx, doc in enumerate(matched_docs):
        print(f"\n📄 Case Law {idx+1}: {matched_metadata[idx]['source']}\n{doc[:500]}...")  # Display first 500 chars

    return matched_docs

if __name__ == "__main__":
    while True:
        query = input("\n❓ Enter your legal query (or type 'exit' to quit): ")
        if query.lower() == "exit":
            break
        retrieve_case_laws(query)
