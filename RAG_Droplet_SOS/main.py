# main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from typing import List
import os, shutil

from file_utils import extract_text_from_pdfs, extract_text_from_csvs, extract_text_from_txts
from rag_utils import embed_text, get_rag_chain, load_vectorstore, append_to_vectorstore

app = FastAPI()
UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

vectorstore = None
qa_chain = None

@app.on_event("startup")
async def startup_event():
    global vectorstore, qa_chain
    vectorstore = load_vectorstore()
    if vectorstore:
        qa_chain = get_rag_chain(vectorstore)

@app.post("/upload/")
async def upload_files(files: List[UploadFile] = File(...)):
    global vectorstore, qa_chain
    saved_text = ""
    for file in files:
        path = os.path.join(UPLOAD_DIR, file.filename)
        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        if file.filename.endswith(".pdf"):
            saved_text += extract_text_from_pdfs([path])
        elif file.filename.endswith(".csv"):
            saved_text += extract_text_from_csvs([path])
        elif file.filename.endswith(".txt"):
            saved_text += extract_text_from_txts([path])

    if saved_text.strip():
        if vectorstore:
            append_to_vectorstore(vectorstore, saved_text)
        else:
            vectorstore = embed_text(saved_text)
        qa_chain = get_rag_chain(vectorstore)

    return {"message": "Files processed and vectorstore updated."}

@app.get("/ask/")
async def ask(q: str):
    if not qa_chain:
        return JSONResponse(status_code=400, content={"error": "Vector store not ready."})
    answer = qa_chain.invoke({"query": q})  # Updated for langchain 0.1+
    return {"answer": answer}
