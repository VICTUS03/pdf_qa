import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader 
from llama_parse import LlamaParse
from langchain_core.documents import Document
import streamlit as st

from dotenv import load_dotenv

load_dotenv()

def get_key(key_name):
    try:
        if key_name in st.secrets:
            return st.secrets[key_name]
        
        return os.environ.get(key_name)
        
    except Exception:
        return None

# LlamaParse
try:
    parser = LlamaParse(
        result_type="markdown",
        num_workers=4,
        verbose=True,
        api_key= get_key("LLAMA_CLOUD_API_KEY") 
    )
except Exception as e:
    print(f"⚠️ LlamaParse Init Failed: {e}")
    parser = None

def save_uploaded_files(uploaded_files):
    file_paths = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            file_paths.append(tmp.name)
    return file_paths


#  for pypdfloader

# #  # UPDATED: Handles a list of file paths
# def load_pdfs(file_paths):
#     all_documents = []
#     for path in file_paths:
#         loader = PyPDFLoader(path)
#         # Add this PDF's pages to the main list
#         all_documents.extend(loader.load())
#     return all_documents




def load_pdfs(file_paths):
    all_documents = []
    # Get threshold from a variable so it's easy to change later
    PAGE_LIMIT_FOR_LLAMA = 20 

    for path in file_paths:
        try:
            # Page Count 
            reader = PdfReader(path)
            page_count = len(reader.pages)
            
            # small document, use high-quality LlamaParse
            if page_count <= PAGE_LIMIT_FOR_LLAMA:
                if parser and os.getenv("LLAMA_CLOUD_API_KEY"):
                    print(f"✨ Small doc ({page_count} pgs): Using LlamaParse for {path}")
                    llama_docs = parser.load_data(path)
                    for l_doc in llama_docs:
                        all_documents.append(
                            Document(page_content=l_doc.text, metadata=l_doc.metadata)
                        )
                else:
                    raise Exception("LlamaParse missing config")

            else:
                print(f"⏩ Large doc ({page_count} pgs): Switching to Fast PyPDF for {path}")
                loader = PyPDFLoader(path)
                all_documents.extend(loader.load())

        except Exception as e:
            # Global Fallback
            print(f"⚠️ Primary parsing failed for {path}: {e}. Falling back to PyPDF.")
            loader = PyPDFLoader(path)
            all_documents.extend(loader.load())
            
    if not all_documents:
        raise ValueError("❌ No content extracted from PDFs.")
        
    return all_documents
