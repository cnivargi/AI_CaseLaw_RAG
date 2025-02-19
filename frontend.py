import streamlit as st
import requests

# API Endpoint
API_URL = "http://127.0.0.1:5000/search"

# Streamlit UI
st.set_page_config(page_title="AI Case Law Search", layout="wide")

st.title("🔍 AI-Powered Case Law Search")

# User query input
query = st.text_input("Enter your legal query:", "")

if st.button("Search"):
    if query:
        with st.spinner("Searching case laws..."):
            response = requests.get(API_URL, params={"query": query})
            if response.status_code == 200:
                data = response.json()
                if data["results"]:
                    st.subheader("📄 Search Results")
                    for i, result in enumerate(data["results"], start=1):
                        st.write(f"**{i}. {result['filename']} (Chunk {result['chunk_id']})**")
                        st.write(result["text"])
                        st.markdown("---")
                else:
                    st.warning("No relevant case laws found.")
            else:
                st.error("Error fetching results from backend.")
    else:
        st.warning("Please enter a search query.")

# Run using:
# streamlit run frontend.py
