from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma, FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

# 1) LOAD your resume & docs
resume = PyPDFLoader("./resumes/Vincent_CV_2022.pdf").load()
extras = TextLoader("./resumes/PL.txt").load()
docs = resume + extras

# 2) SPLIT the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(docs)

# 3) EMBED + INDEX
embeddings = OpenAIEmbeddings(openai_api_key=api_key)
vectorstore = FAISS.from_documents(docs, embeddings)

# 4) MAKE RAG chain
llm = OpenAI(temperature=0)
agent = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
)

def ask_me(question):
    print(agent.run(question))

# Example usage
if __name__ == "__main__":
    ask_me("what kind of python project have you done?")

