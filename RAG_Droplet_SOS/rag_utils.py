import os
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
import boto3
from dotenv import load_dotenv

load_dotenv()

SPACES_BUCKET = os.getenv("SPACES_BUCKET")
FAISS_LOCAL_DIR = "vectorstore"

spaces = boto3.client(
    "s3",
    region_name=os.getenv("SPACES_REGION"),
    endpoint_url=os.getenv("SPACES_ENDPOINT"),
    aws_access_key_id=os.getenv("SPACES_KEY"),
    aws_secret_access_key=os.getenv("SPACES_SECRET")
)

def upload_vectorstore_to_spaces():
    spaces.upload_file(f"{FAISS_LOCAL_DIR}/index.faiss", SPACES_BUCKET, "index.faiss")
    spaces.upload_file(f"{FAISS_LOCAL_DIR}/index.pkl", SPACES_BUCKET, "index.pkl")

def download_vectorstore_from_spaces():
    if not os.path.exists(FAISS_LOCAL_DIR):
        os.makedirs(FAISS_LOCAL_DIR)
    spaces.download_file(SPACES_BUCKET, "index.faiss", f"{FAISS_LOCAL_DIR}/index.faiss")
    spaces.download_file(SPACES_BUCKET, "index.pkl", f"{FAISS_LOCAL_DIR}/index.pkl")

def embed_text(text: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = [Document(page_content=chunk) for chunk in splitter.split_text(text)]
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(FAISS_LOCAL_DIR)
    upload_vectorstore_to_spaces()
    return vectorstore

def append_to_vectorstore(vectorstore, text: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    new_docs = [Document(page_content=chunk) for chunk in splitter.split_text(text)]
    vectorstore.add_documents(new_docs)
    vectorstore.save_local(FAISS_LOCAL_DIR)
    upload_vectorstore_to_spaces()

def load_vectorstore():
    try:
        download_vectorstore_from_spaces()
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return FAISS.load_local(FAISS_LOCAL_DIR, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"Failed to load from Spaces: {e}")
        return None

def get_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    llm = Ollama(model="tinyllama")  # Adjust as needed
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)