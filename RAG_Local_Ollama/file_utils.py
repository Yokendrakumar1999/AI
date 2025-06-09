import fitz  # PyMuPDF
import pandas as pd

def extract_text_from_pdfs(file_paths):
    all_text = ""
    for file_path in file_paths:
        with fitz.open(file_path) as doc:
            for page in doc:
                all_text += page.get_text()
    return all_text

def extract_text_from_csvs(file_paths):
    all_text = ""
    for path in file_paths:
        try:
            df = pd.read_csv(path)
            all_text += df.astype(str).apply(lambda x: " ".join(x), axis=1).str.cat(sep="\n") + "\n"
        except Exception as e:
            print(f"Error reading {path}: {e}")
    return all_text

def extract_text_from_txts(file_paths):
    all_text = ""
    for path in file_paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                all_text += f.read() + "\n"
        except Exception as e:
            print(f"Error reading {path}: {e}")
    return all_text
