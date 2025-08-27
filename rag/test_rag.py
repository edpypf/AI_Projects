import os
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from dotenv import load_dotenv, find_dotenv

class RAGClass:
    def __init__(self, resume_path: str):
        """Initialize the RAGClass with the path to the resume PDF."""
        self.resume_path = resume_path
        self.documents = []
        self.text_chunks = []
        self.vectorstore = None
        self.retriever = None
        self.qa_chain = None

    def load_documents(self):
        """Load the resume and store them in self.documents and return it."""
        loader = TextLoader(self.resume_path)
        self.documents = loader.load()
        print(f"Loaded {len(self.documents)} documents from {self.resume_path}")
        for i, doc in enumerate(self.documents):
            preview = doc.page_content[:200].replace('\n', '')  # Preview first 200 characters
            print(f"Document {i+1} content preview: {preview}{'...' if len(doc.page_content) > 200 else ''}")
        return self.documents

    def split_documents(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """Split the loaded documents into smaller chunks. returns the list of text chunks."""
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.text_chunks = text_splitter.split_documents(self.documents)
        print(f"Split documents into {len(self.text_chunks)} text chunks.")
        for i, chunk in enumerate(self.text_chunks):
            formatted_text = chunk.page_content.replace('. ', '.')
            print(f"Chunk {i+1} : {formatted_text}")
        return self.text_chunks

    def create_vectorstore(self):
        """Create a vector store from the text chunks using OpenAI embeddings. returns the vectorstore object."""
        if not self.text_chunks:
            raise ValueError("Text chunks are empty. Please split documents first.")
        embeddings = OpenAIEmbeddings()
        self.vectorstore = Chroma.from_documents(self.text_chunks, embedding=embeddings)
        print("Vector store created with OpenAI embeddings.")
        print(f"Vectorstore contains {len(self.vectorstore)} documents.")
        for i, doc in enumerate(self.text_chunks):
            formatted_text = doc.page_content.replace('. ', '.')
            print(f"Vectorstore Document {i+1} : {formatted_text}")
        return self.vectorstore

# Test the code
try:
    # Load environment variables
    _env_path = find_dotenv(usecwd=True)
    load_dotenv(_env_path, override=True)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key.startswith("YOUR_") or api_key.strip() == "":
        raise RuntimeError(f"Please set your OpenAI API key in the .env file as OPENAI_API_KEY.:{api_key}|{_env_path}")
    os.environ["OPENAI_API_KEY"] = api_key

    rag = RAGClass(resume_path="my_text_file.txt")
    print(rag)

    rag.load_documents()
    rag.split_documents()
    rag.create_vectorstore()
    print("SUCCESS: Vector store created successfully!")
    
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {str(e)}")
    import traceback
    traceback.print_exc()
