from flask import Flask, request, jsonify
from flask_cors import CORS
import chromadb
import ollama

app = Flask(__name__)
CORS(app)

# 🔹 Explicitly set the ChromaDB path
CHROMA_DB_PATH = "D:/GenAI2025/AI_CaseLaw_RAG/chromadb"

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

# Select the collection storing legal case laws
collection = chroma_client.get_collection("case_laws")

def retrieve_case_law(query):
    """ Retrieves top 3 relevant case laws from ChromaDB. """
    try:
        results = collection.query(
            query_texts=[query], 
            n_results=3  # Fetch 3 best matches
        )

        if results and results["documents"]:
            retrieved_cases = "\n\n".join([doc[0] for doc in results["documents"]])
            return retrieved_cases.encode('utf-8', 'ignore').decode('utf-8')  # Fix encoding
        else:
            return "No relevant case law found in the database."
    
    except Exception as e:
        print(f"⚠️ ChromaDB Retrieval Error: {e}")
        return "Error retrieving case law."

def query_llama3(prompt):
    """ Sends query to Llama 3-8B running via Ollama. """
    try:
        print(f"🔹 Sending query to Llama 3-8B: {prompt}")

        response = ollama.chat(
            model="llama3:8b",
            messages=[{"role": "user", "content": prompt}],
            options={"num_keep": 20}  # Keep 20 tokens from previous response
        )

        if "message" in response:
            return response["message"]["content"]
        else:
            return "Error: No valid response from Llama 3."

    except Exception as e:
        print(f"⚠️ Llama 3 Error: {e}")
        return f"Error: {str(e)}"


@app.route("/api/search", methods=["GET"])
def search_case_law():
    """ Handles search queries and returns legal case law analysis. """
    query = request.args.get("query", "").strip()
    if not query:
        return jsonify({"response": "Error: No query provided."})

    print(f"🔍 Searching for: {query}")

    # Step 1: Retrieve case law from ChromaDB
    retrieved_text = retrieve_case_law(query)

    # Step 2: Construct structured legal analysis prompt
    detailed_prompt = f"""
You are a legal expert specializing in case law interpretation. 
Given the following retrieved case laws, analyze them and include references:

- **Case Summary**: Summarize the facts and legal issues.
- **Core Legal Principles**: Explain the key legal doctrines applied.
- **Reasoning Behind the Judgment**: Discuss the court’s logic.
- **Relevant Past Cases**: Compare with similar precedents.
- **Impact & Precedent**: Explain how this case affects future rulings.
- **Final Conclusion**: Provide a concise legal opinion.

Ensure that you **cite the case name and reference number** when making statements.

### Query:
{query}

### Case Laws:
{retrieved_text}

Provide a structured legal analysis with proper citations.
"""

    # Step 3: Get AI-generated legal analysis
    ai_response = query_llama3(detailed_prompt)

    return jsonify({"response": ai_response})


if __name__ == "__main__":
    app.run(debug=True)
