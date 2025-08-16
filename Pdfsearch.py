import os
import PyPDF2
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss

# =============================================
# 1. SETUP - Using Pre-trained Model
# =============================================

# Smaller pre-trained model (faster loading, still effective)
MODEL_NAME = "paraphrase-MiniLM-L3-v2"  # ~80MB (vs original 496MB)
PDF_FOLDER = "/Users/jaiaakaash/Documents/study"

# =============================================
# 2. LOAD MODEL (Cached for performance)
# =============================================

@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_NAME)

model = load_model()  # This will cache after first run

# =============================================
# 3. PROCESS PDF FILES
# =============================================

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file"""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        return " ".join([page.extract_text() or "" for page in reader.pages])

documents = []
file_names = []

for filename in os.listdir(PDF_FOLDER):
    if filename.endswith(".pdf"):
        try:
            text = extract_text_from_pdf(os.path.join(PDF_FOLDER, filename))
            if text.strip():  # Only add if text exists
                documents.append(text)
                file_names.append(filename)
        except Exception as e:
            st.warning(f"⚠️ Error reading {filename}: {str(e)}")

if not documents:
    st.error("❌ No valid PDF documents found. Please check:")
    st.write(f"- Folder path: {PDF_FOLDER}")
    st.write("- Ensure PDFs contain extractable text")
    st.stop()

# =============================================
# 4. CREATE SEARCH INDEX
# =============================================

embeddings = model.encode(documents)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

# =============================================
# 5. STREAMLIT UI
# =============================================

st.title("🔍 PDF Search Engine")
st.success(f"✅ Loaded {len(documents)} documents")

query = st.text_input("Search your PDFs:", placeholder="Type your question or keywords...")

if query:
    # Get top 3 most relevant results
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k=3)
    
    st.subheader("📄 Top Results:")
    for rank, idx in enumerate(indices[0]):
        with st.expander(f"#{rank+1}: {file_names[idx]} (Score: {1-distances[0][rank]:.2f})"):
            st.write(documents[idx][:1000])  # Show first 1000 chars
            st.download_button(
                label="📥 Download Full Text",
                data=documents[idx],
                file_name=f"extract_{file_names[idx]}.txt",
                mime="text/plain"
            )

# =============================================
# 6. DEBUG INFO (Visible in Advanced Mode)
# =============================================

with st.expander("ℹ️ System Information"):
    st.write(f"Model: {MODEL_NAME}")
    st.write(f"PDF Folder: {PDF_FOLDER}")
    st.write(f"Total Text Chars: {sum(len(d) for d in documents):,}")