from PyPDF2 import PdfReader
import csv

def extract_text_from_pdfs(paths):
    text = ""
    for path in paths:
        try:
            reader = PdfReader(path)
            for page in reader.pages:
                text += page.extract_text() or ""
        except Exception as e:
            print(f"PDF extraction failed: {e}")
    return text

def extract_text_from_csvs(paths):
    text = ""
    for path in paths:
        try:
            with open(path, newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    text += ' '.join(row) + '\n'
        except Exception as e:
            print(f"CSV extraction failed: {e}")
    return text

def extract_text_from_txts(paths):
    text = ""
    for path in paths:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                text += f.read() + '\n'
        except Exception as e:
            print(f"TXT extraction failed: {e}")
    return text
