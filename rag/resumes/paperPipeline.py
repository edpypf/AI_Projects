import os
import glob
from sentence_transformers import SentenceTransformer
import fitz
import numpy as np
from fastapi import FastAPI
import faiss

def extract_text_from_pdf(pdf_path: str) -> str:
    # open a PDF file and extract text as a single string
    doc = fitz.open(pdf_path)
    pages=[]
    for page in doc:
        page_text = page.get_text()
        pages.append(page_text)
    full_text = "\n".join(pages)
    return full_text

def chunk_text(text: str, max_tokens: int = 64, overlap: int=50) -> list[str]:
    tokens = text.split()
    chunks = []
    step = max_tokens - overlap
    for i in range(0, len(tokens), step):
        chunk = tokens[i:i + max_tokens]
        if chunk:
            chunks.append(" ".join(chunk))
    return chunks

def get_pdf_files(folder: str) -> list[str]:
    # Find all PDF files in the folder
    return glob.glob(os.path.join(folder, "*.pdf"))

# Set your folder path here
pdf_folder = "resumes"  # Change to your folder name

all_chunks = []
for pdf_path in get_pdf_files(pdf_folder):
    full_text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(full_text, max_tokens=64, overlap=50)
    all_chunks.extend(chunks)

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(all_chunks)

dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)  # L2 distance index
index.add(np.array(embeddings, dtype=np.float32))  # Add embeddings to the index

# Example: search for a query embeeding
query_embedding = embeddings[0]  # Replace with your query embedding
k=3
distances, indices = index.search(np.array([query_embedding], dtype=np.float32), k) 

app = FastAPI()
@app.get("/search")
async def search(query: str):
    # Receive a query, embed it, retrieve top-3 passages, and return them
    query_embedding = model.encode([query])[0]
    # Perform FAISS search
    k = 3
    distances, indices = index.search(np.array([query_embedding], dtype=np.float32), k)
    results = [all_chunks[i] for i in indices[0]]
    return {"query": query, "results": results} 

# uvicorn paperPipeline:app --reload
# http://127.0.0.1:8000/search?query="what"

