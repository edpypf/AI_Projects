import os
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from dotenv import load_dotenv, find_dotenv
import tempfile
import shutil
import time

class RAGClass:
    def __init__(self, resume_path: str):
        """Initialize the RAGClass with the path to the resume PDF."""
        self.resume_path = resume_path
        self.documents = []
        self.text_chunks = []
        self.vectorstore = None
        self.retriever = None
        self.qa_chain = None
        # Create a temporary directory for Chroma
        self.persist_directory = tempfile.mkdtemp()

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
        
        print("Creating OpenAI embeddings...")
        embeddings = OpenAIEmbeddings()
        
        print("Creating Chroma vector store...")
        try:
            # Use a persistent directory to avoid memory issues
            self.vectorstore = Chroma.from_documents(
                documents=self.text_chunks, 
                embedding=embeddings,
                persist_directory=self.persist_directory
            )
            print("Vector store created with OpenAI embeddings.")
            print(f"Vectorstore contains {len(self.text_chunks)} documents.")
            
            for i, doc in enumerate(self.text_chunks):
                formatted_text = doc.page_content.replace('. ', '.')
                print(f"Vectorstore Document {i+1} : {formatted_text}")
                
        except Exception as e:
            print(f"Error creating vector store: {e}")
            # Clean up temp directory on error
            if os.path.exists(self.persist_directory):
                shutil.rmtree(self.persist_directory)
            raise
            
        return self.vectorstore

    
    def setup_retriever(self):
        """Set up the retriever from the vector store for similarity search. Returns the retriever object."""
        if not self.vectorstore:
            raise ValueError("Vector store is not created. Please create vector store first.")
        self.retriever = self.vectorstore.as_retriever()
        print("Retriever set up for similarity search.")
        print(f"<b>Retriever is ready to use: {self.retriever}</b>")
        return self.retriever
    
    def setup_qa_chain(self, llm: ChatOpenAI):
        """Set up the QA chain using the retriever and a language model. returns the QA chain object."""
        if not self.retriever:
            raise ValueError("Retriever is not set up. Please set up retriever first.")
        llm = llm or ChatOpenAI(model_name="gpt-4", temperature=0)
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True
        )
        print("QA chain set up with the language model.")
        print(f"<b>QA Chain is ready to use: {self.qa_chain}</b>")
        return self.qa_chain
    
    def answer_qa(self, query: str):
        """Run the QA chain with a query and return the answer and source documents."""
        if not self.qa_chain:
            raise ValueError("QA chain is not set up. Please set up QA chain first.")
        result = self.qa_chain(query)
        print(f"<b>Query: </b>{query}")
        print(f"<b>Answer: </b>{result['result']}")
        return result
    
    def evaluate(self, query: str, ground_truths: list):
        """Evaluate the QA chain using a list of query and ground truths. returns the accuracy as a float."""
        if len(query) != len(ground_truths):
            print(f"<b>Evaluating {len(query)} queries...</b>")
            print(f"<b>Ground Truths: </b> {len(ground_truths)}")
            raise ValueError("Query and ground truths must have the same length.")
        if self.qa_chain is None:
            raise ValueError("QA chain is not set up. Please set up QA chain first.")
        correct = 0
        for idx, (query, ground_truth) in enumerate(zip(query, ground_truths)):
            answer = self.qa_chain(query)
            print(f"<b>Query {idx+1}: </b>{query}Expected: {ground_truth} Model Answer: {answer}")
            if ground_truth.lower() in answer['result'].lower():
                correct += 1
        accuracy = correct / len(query) if query else 0
        
        print(f"<b>Overall Accuracy: </b>{accuracy:.2f}")
        return accuracy
    
# Test the code
# Load environment variables
_env_path = find_dotenv(usecwd=True)
load_dotenv(_env_path, override=True)
api_key = os.getenv("OPENAI_API_KEY")
if not api_key or api_key.startswith("YOUR_") or api_key.strip() == "":
    raise RuntimeError(f"Please set your OpenAI API key in the .env file as OPENAI_API_KEY.:{api_key}|{_env_path}")
os.environ["OPENAI_API_KEY"] = api_key

print("Initializing RAG class...")
rag = RAGClass(resume_path="my_text_file.txt")

print("Loading documents...")
rag.load_documents()

print("Splitting documents...")
rag.split_documents()

print("Creating vector store...")
rag.create_vectorstore()

print("setting up retriever...")
rag.setup_retriever()

print("Setting up QA chain...")
rag.setup_qa_chain(llm=ChatOpenAI(model_name="gpt-4", temperature=0))

# rag.answer_qa("What is Retrieval-Augmented Generation?")
sample_queries = ["Define RAG.", "Explain vector databases."]
sample_ground_truths = ["Retrieval-Augmented Generation", "Vector databases store embeddings"]
rag.evaluate(sample_queries, sample_ground_truths)

