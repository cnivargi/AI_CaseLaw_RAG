import streamlit as st
import requests

# Streamlit UI
st.set_page_config(page_title="AI Case Law Assistant", layout="wide")

st.title("📜 AI-Powered Case Law Querying System")

# Input for user query
user_query = st.text_area("🔍 Ask your legal question:")

# Submit button
if st.button("Get Answer"):
    if user_query.strip():
        with st.spinner("Processing... ⏳"):
            response = requests.post("http://127.0.0.1:5000/query", json={"query": user_query})
            if response.status_code == 200:
                st.success("✅ Response received!")
                st.write(response.json()["answer"])
            else:
                st.error("⚠️ Error fetching response. Please try again.")
    else:
        st.warning("⚠️ Please enter a question.")

# Upload PDFs to dynamically update ChromaDB
st.subheader("📂 Upload New Case Law PDFs")
uploaded_files = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=["pdf"])

if uploaded_files:
    for uploaded_file in uploaded_files:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
        response = requests.post("http://127.0.0.1:5000/upload", files=files)
        if response.status_code == 200:
            st.success(f"✅ {uploaded_file.name} added successfully!")
        else:
            st.error(f"⚠️ Failed to add {uploaded_file.name}.")
