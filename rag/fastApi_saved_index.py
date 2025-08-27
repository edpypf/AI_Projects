import pickle
import faiss
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index("faiss.index")
with open("chunks.pkl", "rb") as f:
    all_chunks = pickle.load(f)

app = FastAPI()
@app.get("/search")
async def search(query: str):
    query_embedding = model.encode([query])[0]
    k = 3
    distances, indices = index.search(np.array([query_embedding], dtype=np.float32), k)
    results = [all_chunks[i] for i in indices[0]]
    return {"query": query, "results": results}    