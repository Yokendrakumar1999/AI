Here is a **complete, step-by-step end-to-end guide** to build a **FastAPI-based RAG (Retrieval-Augmented Generation)** system on **Windows** using:

* 📁 File upload (PDF, CSV, TXT)
* 🧠 HuggingFace Embeddings + FAISS vector store
* 🤖 LangChain RAG + Ollama (TinyLlama)
* 💡 Persistent vector store (even after restart)
* ⚙️ Python, FastAPI, Uvicorn

---

# ✅ Step-by-Step RAG App with FastAPI + LangChain + Ollama

---

## 🔹 Step 1: Install Python and Create Environment

### 1.1 ✅ Install Python

* Download: [https://www.python.org/downloads/](https://www.python.org/downloads/)
* **CHECK** ✅ "Add Python to PATH" during installation

### 1.2 ✅ Open Terminal and Create Project Folder

```powershell
cd C:\Users\YourName\Desktop
mkdir RAG_Local_Ollama
cd RAG_Local_Ollama
```

### 1.3 ✅ Create Virtual Environment

```powershell
python -m venv venv
.\venv\Scripts\activate
```

---

## 🔹 Step 2: Install Dependencies

### 2.1 ✅ Create `requirements.txt`

```txt
fastapi
uvicorn
langchain
langchain-community
langchain-huggingface
transformers
faiss-cpu
pypdf
pandas
ollama
```

### 2.2 ✅ Install with pip

```powershell
pip install -r requirements.txt
```

---

## 🔹 Step 3: Project Structure

```
RAG_Local_Ollama/
├── main.py
├── file_utils.py
├── rag_utils.py
├── requirements.txt
├── uploaded_files/         ← auto-created
├── venv/
```

---

## 🔹 Step 4: Add Core Files

### 4.1 `file_utils.py`

```python
from PyPDF2 import PdfReader
import pandas as pd

def extract_text_from_pdfs(paths):
    text = ""
    for path in paths:
        reader = PdfReader(path)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

def extract_text_from_csvs(paths):
    text = ""
    for path in paths:
        df = pd.read_csv(path)
        text += df.to_string()
    return text

def extract_text_from_txts(paths):
    text = ""
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            text += f.read()
    return text
```

---

### 4.2 `rag_utils.py`

```python
import os
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA

EMBED_MODEL = "all-MiniLM-L6-v2"
VECTORSTORE_DIR = "vectorstore"

def embed_text(text: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = [Document(page_content=chunk) for chunk in splitter.split_text(text)]
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(VECTORSTORE_DIR)
    return vectorstore

def load_vectorstore():
    if not os.path.exists(f"{VECTORSTORE_DIR}/index.faiss"):
        return None
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return FAISS.load_local(VECTORSTORE_DIR, embeddings)

def get_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever()
    llm = Ollama(model="tinyllama")
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
```

---

### 4.3 `main.py`

```python
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from typing import List
import os
import shutil

from file_utils import extract_text_from_pdfs, extract_text_from_csvs, extract_text_from_txts
from rag_utils import embed_text, load_vectorstore, get_rag_chain

app = FastAPI()

UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

vectorstore = load_vectorstore()
qa_chain = get_rag_chain(vectorstore) if vectorstore else None

@app.post("/upload/")
async def upload_files(files: List[UploadFile] = File(...)):
    global vectorstore, qa_chain
    all_text = ""

    for file in files:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        if file.filename.endswith(".pdf"):
            all_text += extract_text_from_pdfs([file_path])
        elif file.filename.endswith(".csv"):
            all_text += extract_text_from_csvs([file_path])
        elif file.filename.endswith(".txt"):
            all_text += extract_text_from_txts([file_path])

    if not all_text.strip():
        return JSONResponse(content={"error": "No valid text extracted"}, status_code=400)

    vectorstore = embed_text(all_text)
    qa_chain = get_rag_chain(vectorstore)

    return {"message": "Files uploaded and processed successfully."}

@app.get("/ask/")
async def ask(query: str):
    global qa_chain
    if not qa_chain:
        return JSONResponse(content={"error": "Upload documents first."}, status_code=400)
    return {"answer": qa_chain.run(query)}
```

---

## 🔹 Step 5: Install and Start Ollama

### 5.1 ✅ Install Ollama

Download and install from: [https://ollama.com/](https://ollama.com/)

### 5.2 ✅ Run TinyLlama model

```bash
ollama pull tinyllama
ollama run tinyllama
```

Leave this running in a separate terminal.

---

## 🔹 Step 6: Run FastAPI Server

```bash
uvicorn main:app --reload
```

---

## 🔹 Step 7: Test Your App

### ✅ Upload Files

POST to: `http://127.0.0.1:8000/upload/`
Use a tool like [Postman](https://www.postman.com/) or Swagger UI (auto opens at `http://127.0.0.1:8000/docs`)

### ✅ Ask Questions

GET: `http://127.0.0.1:8000/ask/?query=your-question`

---

## 🔒 Bonus: Persistent VectorStore

Every time the app runs, it loads saved FAISS vectors from disk (`vectorstore/`). So uploaded data survives app restarts.

---

## ✅ You now have a full local RAG system!

Let me know if you want:

* 🐳 Docker support
* ☁️ AWS deployment
* 🧪 Swagger/Postman testing scripts
* 🧠 Switch to OpenAI instead of Ollama
