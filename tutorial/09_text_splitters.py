"""
09 - Text Splitters
===================
Concept: LLMs have context window limits. Splitters break documents into
smaller chunks that can be embedded and retrieved individually.

Key parameters:
  - chunk_size    -> max characters (or tokens) per chunk
  - chunk_overlap -> characters shared between consecutive chunks

Splitters covered:
  - CharacterTextSplitter
  - RecursiveCharacterTextSplitter (recommended default)
  - TokenTextSplitter
  - MarkdownHeaderTextSplitter
  - HTMLHeaderTextSplitter
"""

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
PDF_PATH = os.path.join(DATA_DIR, "laptop_policy.pdf")

raw_docs = PyPDFLoader(PDF_PATH).load()
full_text = "\n\n".join(doc.page_content for doc in raw_docs)

def show_chunks(chunks, label):
    print(f"\n=== {label} ===")
    print(f"Total chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks[:2]):
        print(f"  [{i}] ({len(chunk.page_content)} chars) {chunk.page_content[:120].replace(chr(10), ' ')}...")

# 1. CharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter
show_chunks(CharacterTextSplitter(separator="\n\n", chunk_size=500, chunk_overlap=50).create_documents([full_text]),
            "CharacterTextSplitter (separator='\\n\\n')")

# 2. RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
show_chunks(RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).create_documents([full_text]),
            "RecursiveCharacterTextSplitter (recommended)")

# 3. TokenTextSplitter
from langchain_text_splitters import TokenTextSplitter
show_chunks(TokenTextSplitter(chunk_size=150, chunk_overlap=20).create_documents([full_text]),
            "TokenTextSplitter (150 tokens/chunk)")

# 4. MarkdownHeaderTextSplitter
from langchain_text_splitters import MarkdownHeaderTextSplitter
md_text = """# Company Laptop Policy
## 1. Purpose
This policy establishes guidelines for laptop use.
## 2. Acceptable Use
Employees may use laptops for business tasks.
### 2.1 Software
Only install approved software.
## 3. Security
Full-disk encryption must be enabled."""
md_chunks = MarkdownHeaderTextSplitter(headers_to_split_on=[("#","h1"),("##","h2"),("###","h3")]).split_text(md_text)
print(f"\n=== MarkdownHeaderTextSplitter ===\nTotal chunks: {len(md_chunks)}")
for c in md_chunks:
    print(f"  metadata={c.metadata} | {c.page_content[:80]}...")

# 5. HTMLHeaderTextSplitter
from langchain_text_splitters import HTMLHeaderTextSplitter
html_text = "<html><body><h1>Laptop Policy</h1><p>Usage guidelines.</p><h2>Security</h2><p>Enable encryption and use VPN.</p></body></html>"
html_chunks = HTMLHeaderTextSplitter(headers_to_split_on=[("h1","header1"),("h2","header2")]).split_text(html_text)
print(f"\n=== HTMLHeaderTextSplitter ===\nTotal chunks: {len(html_chunks)}")
for c in html_chunks:
    print(f"  metadata={c.metadata} | {c.page_content[:80]}...")

print("\nSplitting complete. Next: embed and store in a vector database.")
