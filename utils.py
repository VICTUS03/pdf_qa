import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader 
from llama_parse import LlamaParse
from langchain_core.documents import Document

from dotenv import load_dotenv

load_dotenv()

# LlamaParse
try:
    parser = LlamaParse(
        result_type="markdown",
        num_workers=4,
        verbose=True,
        api_key=os.getenv("LLAMA_CLOUD_API_KEY") 
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
    for path in file_paths:
        try:
            # 1. Try LlamaParse first (High Quality)
            if parser and os.getenv("LLAMA_CLOUD_API_KEY"):
                print(f"📄 Parsing with LlamaParse: {path}")
                llama_docs = parser.load_data(path)
                
                # Convert LlamaIndex docs to LangChain docs
                for l_doc in llama_docs:
                    all_documents.append(
                        Document(
                            page_content=l_doc.text,
                            metadata=l_doc.metadata
                        )
                    )
            else:
                raise Exception("LlamaParse skipped (No API Key)")
                
        except Exception as e:
            # 2. Fallback to PyPDF (Standard)
            print(f"⚠️ LlamaParse failed/skipped: {e}. Falling back to PyPDF.")
            loader = PyPDFLoader(path)
            all_documents.extend(loader.load())
            
    # CRITICAL CHECK
    if not all_documents:
        raise ValueError("❌ No content could be extracted from these PDFs. They might be empty or scanned images.")
        
    return all_documents






