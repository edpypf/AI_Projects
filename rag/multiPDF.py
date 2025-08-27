import os
import glob
import fitz
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle

def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    return "\n".join([page.get_text() for page in doc])

def chunk_text(text: str, max_tokens: int = 64, overlap: int = 50) -> list[str]:
    tokens = text.split()
    chunks = []
    step = max_tokens - overlap
    for i in range(0, len(tokens), step):
        chunk = tokens[i:i + max_tokens]
        if chunk:
            chunks.append(" ".join(chunk))
    return chunks

pdf_folder = "resumes"
all_chunks = []
for pdf_path in glob.glob(os.path.join(pdf_folder, "*.pdf")):
    full_text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(full_text, max_tokens=64, overlap=50)
    all_chunks.extend(chunks)

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(all_chunks)
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings, dtype=np.float32))

# Save index and chunks
faiss.write_index(index, "faiss.index")
with open("chunks.pkl", "wb") as f:
    pickle.dump(all_chunks, f)

