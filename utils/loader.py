# utils/loader.py

import os
from llama_index.readers.file import PyMuPDFReader, DocxReader
from llama_index.core import Document

def load_all_documents(folder_path: str):
    documents = []

    for file_name in os.listdir(folder_path):
        path = os.path.join(folder_path, file_name)
        if file_name.endswith(".pdf"):
            documents += PyMuPDFReader().load(file_path=path)
        elif file_name.endswith(".docx"):
            documents += DocxReader().load(file_path=path)
        elif file_name.endswith(".txt"):
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
            documents.append(Document(text=text))

    return documents
