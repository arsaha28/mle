"""
08 - Document Loaders
=====================
Concept: Load data into LangChain Document objects before embedding.
Each Document has page_content (text) and metadata (source, page, etc).

Loaders covered:
  - TextLoader       -> plain .txt files
  - PyPDFLoader      -> PDF files (page-by-page)
  - CSVLoader        -> CSV files (row-by-row)
  - DirectoryLoader  -> all files in a folder
  - WebBaseLoader    -> scrape a webpage
"""

import os
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
PDF_PATH = os.path.join(DATA_DIR, "laptop_policy.pdf")

def show_docs(docs, label, max_chars=200):
    print(f"\n=== {label} ===")
    print(f"Total documents loaded: {len(docs)}")
    for i, doc in enumerate(docs[:2]):
        print(f"\n[Doc {i}] metadata: {doc.metadata}")
        print(f"content preview: {doc.page_content[:max_chars]}...")

# 1. TextLoader
txt_path = os.path.join(DATA_DIR, "sample.txt")
with open(txt_path, "w") as f:
    f.write("Artificial Intelligence is transforming industries worldwide.\n"
            "LangChain makes it easier to build applications powered by LLMs.\n")

from langchain_community.document_loaders import TextLoader
show_docs(TextLoader(txt_path).load(), "TextLoader")

# 2. PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader
show_docs(PyPDFLoader(PDF_PATH).load(), "PyPDFLoader (page-by-page)")

# 3. CSVLoader
csv_path = os.path.join(DATA_DIR, "sample.csv")
with open(csv_path, "w") as f:
    f.write("name,department,policy_violation\n")
    f.write("Alice,Engineering,No violations\n")
    f.write("Bob,Marketing,Used personal cloud storage\n")
    f.write("Dave,Sales,Installed unapproved software\n")

from langchain_community.document_loaders import CSVLoader
show_docs(CSVLoader(file_path=csv_path).load(), "CSVLoader (row-by-row)")

# 4. DirectoryLoader
from langchain_community.document_loaders import DirectoryLoader
show_docs(DirectoryLoader(DATA_DIR, glob="*.txt").load(), "DirectoryLoader (all .txt files)")

# 5. WebBaseLoader
from langchain_community.document_loaders import WebBaseLoader
show_docs(WebBaseLoader("https://en.wikipedia.org/wiki/LangChain").load(), "WebBaseLoader (Wikipedia)")

print("\nAll loaders demonstrated. Documents are ready to be split and embedded.")
