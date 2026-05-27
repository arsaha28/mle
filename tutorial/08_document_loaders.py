"""
08 - Document Loaders
=====================
Concept: Before an LLM can answer questions about your data, you need to load
that data into LangChain Document objects. Each Document has:
  - page_content  → the actual text
  - metadata      → source, page number, etc.

Loaders covered:
  - TextLoader         → plain .txt files
  - PyPDFLoader        → PDF files (page-by-page)
  - CSVLoader          → CSV files (row-by-row)
  - DirectoryLoader    → load all files in a folder
"""

import os

from dotenv import load_dotenv

load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
PDF_PATH = os.path.join(DATA_DIR, "laptop_policy.pdf")


# ── Helper ───────────────────────────────────────────────────────────────────────────────
def show_docs(docs: list, label: str, max_chars: int = 200) -> None:
    print(f"\n=== {label} ===")
    print(f"Total documents loaded: {len(docs)}")
    for i, doc in enumerate(docs[:2]):  # show first 2 only
        print(f"\n[Doc {i}] metadata: {doc.metadata}")
        print(f"content preview: {doc.page_content[:max_chars]}...")


# ── 1. TextLoader ───────────────────────────────────────────────────────────────────
# First create a sample .txt file
txt_path = os.path.join(DATA_DIR, "sample.txt")
with open(txt_path, "w") as f:
    f.write(
        "Artificial Intelligence is transforming industries worldwide.\n"
        "From healthcare to finance, AI models are being deployed at scale.\n"
        "LangChain makes it easier to build applications powered by LLMs.\n"
        "Developers can chain together prompts, tools, and models with ease.\n"
    )

from langchain_community.document_loaders import TextLoader  # noqa: E402

text_loader = TextLoader(txt_path)
docs = text_loader.load()
show_docs(docs, "TextLoader")
# TextLoader loads the entire file as a single Document.

# ── 2. PyPDFLoader ────────────────────────────────────────────────────────────────
from langchain_community.document_loaders import PyPDFLoader  # noqa: E402

pdf_loader = PyPDFLoader(PDF_PATH)
docs = pdf_loader.load()
show_docs(docs, "PyPDFLoader (page-by-page)")
# Each page becomes a separate Document with 'page' in metadata.

# ── 3. CSVLoader ──────────────────────────────────────────────────────────────────
csv_path = os.path.join(DATA_DIR, "sample.csv")
with open(csv_path, "w") as f:
    f.write("name,department,policy_violation\n")
    f.write("Alice,Engineering,No violations\n")
    f.write("Bob,Marketing,Used personal cloud storage\n")
    f.write("Carol,HR,No violations\n")
    f.write("Dave,Sales,Installed unapproved software\n")

from langchain_community.document_loaders import CSVLoader  # noqa: E402

csv_loader = CSVLoader(file_path=csv_path)
docs = csv_loader.load()
show_docs(docs, "CSVLoader (row-by-row)")
# Each row becomes a separate Document, with column values in page_content.

# ── 4. DirectoryLoader ───────────────────────────────────────────────────────────────
from langchain_community.document_loaders import DirectoryLoader  # noqa: E402

# loader_cls=TextLoader avoids the default UnstructuredFileLoader which
# requires libmagic and makes network calls.
dir_loader = DirectoryLoader(DATA_DIR, glob="*.txt", loader_cls=TextLoader)
docs = dir_loader.load()
show_docs(docs, "DirectoryLoader (all .txt files)")
# Useful for loading entire knowledge bases stored as files.

print("\n✅ All loaders demonstrated. Documents are ready to be split and embedded.")
