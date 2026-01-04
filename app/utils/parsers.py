import os

from docx import Document as DocxDocument
from pypdf import PdfReader


def parse_file_to_text(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        reader = PdfReader(file_path)
        return " ".join([page.extract_text() for page in reader.pages])
    elif ext == ".docx":
        doc = DocxDocument(file_path)
        return " ".join([para.text for para in doc.paragraphs])
    else:  # .txt
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
