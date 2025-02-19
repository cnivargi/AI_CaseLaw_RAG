import chromadb
from sentence_transformers import SentenceTransformer
import ollama

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="case_laws")

# Load embedding model
embedding_model_name = "BAAI/bge-small-en-v1.5"
embedding_model = SentenceTransformer(embedding_model_name)

# Function to retrieve relevant case laws
def retrieve_case_laws(query, top_k=3):
    """Fetches the most relevant case laws for a given query."""
    query_embedding = embedding_model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    matched_docs = results["documents"][0]
    matched_metadata = results["metadatas"][0]

    context = "\n\n".join([f"📜 Case Law {idx+1}: {matched_metadata[idx]['source']}\n{doc}" for idx, doc in enumerate(matched_docs)])

    return context

# Function to generate response using Phi-4 via Ollama
def generate_response(query):
    """Generates a legal answer using Phi-4 based on retrieved case laws."""
    case_laws = retrieve_case_laws(query)

    prompt = f"""
    You are a highly skilled legal assistant. Answer the following legal question based on the given case laws.

    🔍 **Question:** {query}

    📖 **Relevant Case Laws:** 
    {case_laws}

    Provide a well-structured legal explanation in simple terms.
    """

    response = ollama.chat(model="llama3:8b", messages=[{"role": "user", "content": prompt}])
    
    return response["message"]["content"]

# Run the query-answer system
if __name__ == "__main__":
    while True:
        query = input("\n❓ Enter your legal query (or type 'exit' to quit): ")
        if query.lower() == "exit":
            break
        answer = generate_response(query)
        print("\n🤖 **AI Legal Assistant Answer:**\n")
        print(answer)
