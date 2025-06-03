from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from typing import List
import os
import shutil

from file_utils import extract_text_from_pdfs, extract_text_from_csvs, extract_text_from_txts
from rag_utils import embed_text, get_rag_chain

app = FastAPI()
UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

vectorstore = None
qa_chain = None

@app.post("/upload/")
async def upload_files(files: List[UploadFile] = File(...)):
    saved_pdfs = []
    saved_csvs = []
    saved_txts = []

    for file in files:
        file_location = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_location, "wb") as f:
            shutil.copyfileobj(file.file, f)

        if file.filename.lower().endswith(".pdf"):
            saved_pdfs.append(file_location)
        elif file.filename.lower().endswith(".csv"):
            saved_csvs.append(file_location)
        elif file.filename.lower().endswith(".txt"):
            saved_txts.append(file_location)

    if not (saved_pdfs or saved_csvs or saved_txts):
        return JSONResponse(content={"error": "No valid files uploaded."}, status_code=400)

    text = ""
    if saved_pdfs:
        text += extract_text_from_pdfs(saved_pdfs)
    if saved_csvs:
        text += extract_text_from_csvs(saved_csvs)
    if saved_txts:
        text += extract_text_from_txts(saved_txts)

    global vectorstore, qa_chain
    vectorstore = embed_text(text)
    qa_chain = get_rag_chain(vectorstore)

    return JSONResponse(content={"message": "Files uploaded and embedded."})

@app.post("/ask/")
async def ask_question(question: str):
    if not qa_chain:
        return JSONResponse(content={"error": "Upload files first."}, status_code=400)
    answer = qa_chain.run(question)
    return JSONResponse(content={"answer": answer})
