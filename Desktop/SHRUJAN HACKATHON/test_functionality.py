# Test pages (note: actual filenames have numbers, can't import directly)
import os

# Test if page files exist
pages_to_check = [
    ("pages/1_Document_QA.py", "Document Q&A"),
    ("pages/2_Document_Summary.py", "Document Summary"),
    ("pages/3_Analytics_Dashboard.py", "Analytics Dashboard")
]

for file_path, page_name in pages_to_check:
    if os.path.exists(file_path):
        print(f"✅ {page_name} page file exists")
    else:
        print(f"❌ {page_name} page file missing: {file_path}")

# Test core imports
try:
    import streamlit as st
    print("✅ Streamlit imports successfully")
except ImportError as e:
    print(f"❌ Streamlit import failed: {e}")

try:
    from utils.rag_qa import process_documents, get_answer
    print("✅ RAG QA utils import successfully")
except ImportError as e:
    print(f"❌ RAG QA utils import failed: {e}")
