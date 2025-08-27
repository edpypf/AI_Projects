import os
import glob
from sentence_transformers import SentenceTransformer
import fitz
import numpy as np
from fastapi import FastAPI
import faiss
import sqlite3

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

def create_sqlite_obj(conn):
    conn = sqlite3.connect("demo.db")
    cur = conn.cursor()
    # Define a table for document metadata 
    cur.execute("""
                CREATE TABLE if not exists documents (
                    doc_id    INTEGER PRIMARY KEY,
                    title     TEXT,
                    author    TEXT,
                    year      INTEGER,
                    keywords  TEXT
                );
                """)
    #FTS5 table for text
    cur.execute("""
                CREATE VIRTUAL TABLE if not exists doc_chunks USING fts5(
                    content,                      -- chunk text
                    content='documents',          -- external content table
                    content_rowid='doc_id'        -- link to documents.doc_id
                );
                """)
    conn.commit()

def insert_sqlite_obj(conn, documents):
    cur = conn.cursor()
    for doc in documents:
    # insert metadata
        cur.execute(""" INSERT INTO documents (title, author, year, keywords) VALUES (?, ?, ?, ?)""",
                    (doc["title"], doc["author"], doc["year"], doc["keywords"]))
        doc_id = cur.lastrowid

        # insert chunks 
        for ch in doc.get("chunks", []):
           conn.execute("INSERT INTO doc_chunks(rowid, content) VALUES (?, ?)", (doc["doc_id"], doc["chunk_text"]))
    conn.commit()

def keyword_query(conn, search_terms):
    cur = conn.cursor()
    if search_terms:
    # query based on terms
        doc_id, title = cur.execute(""" SELECT doc_id, title
                        FROM documents
                        JOIN doc_chunks ON documents.doc_id = doc_chunks.rowid
                        WHERE doc_chunks MATCH 'search_terms'
                        LIMIT 5;""")
    return doc_id, title

def BM25(documents, query):
    from rank_bm25 import BM250kapi
    tokenized_docs = [doc.split() for doc in documents]
    bm25 = BM250kapi(tokenized_docs)  
    tokenized_query = query.split()
    top_docs = bm25.get_top_n(tokenized_query, documents, n=3)
    return top_docs

def hybrid_score(vec_score, key_score, alpha=0.5):
    # Assume vec_score and key_score are normalized(0-1)
    return alpha * vec_score + (1-alpha)*key_score

def reRanking_Top_Result(faiss_results, keyword_scores):
    combined = []
    for doc, v_score in faiss_results:
        k_score = keyword_scores.get(doc, 0.0)
        combined_score = hybrid_score(v_score, k_score, alpha=0.6)
        combined.append((doc, combined_score))
    combined.sort(key=lambda x: x[1], reverse=True)
    top_k = combined[:3]
    return top_k 

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

conn = sqlite3.connect("demo.db")
create_sqlite_obj(conn)

documents = []
for idx, chunk in enumerate(all_chunks):
    doc = {
        "doc_id": idx + 1,
        "title": f"Chunk {idx+1}",
        "author": "Unknown",
        "year": 2025,
        "keywords": "",
        "chunks": [chunk],
        "chunk_text": chunk
    }
    documents.append(doc)

insert_sqlite_obj(conn, documents=documents)


@app.get("/hybrid_search")
async def hybrid_search(query: str, k: int = 3):
    # 1. Compute query embedding for FAISS
    query_embedding = model.encode([query])[0]
    # faiss_results: list of tuples (chunk_text, score)
    faiss_results = [(all_chunks[i], distances[0][idx]) for idx, i in enumerate(indices[0])]

    # 2. Get top-k from FAISS and top-k from SQLite FTS/BM25
    keyword_scores = keyword_query(conn, query)
    top_k = reRanking_Top_Result(faiss_results, keyword_scores)
    # 3. Merge scores (as above) and select final top-k documents
    return {"results": top_k} 

# uvicorn paperPipeline:app --reload
# http://127.0.0.1:8000/search?query="what skills are required as data analyst"

