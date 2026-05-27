"""
09 - Text Splitters
===================
Concept: LLMs have a context window limit. Loaded documents are often too large
to fit in a single prompt. Splitters break documents into smaller *chunks* that
can be embedded and retrieved individually.

Why chunk_overlap matters:
  Imagine a sentence that falls exactly on a chunk boundary — it gets cut in half.
  chunk_overlap copies the last N characters of one chunk into the start of the next,
  so no sentence is ever completely lost at a boundary.

  Example with chunk_size=100, chunk_overlap=20:
    Chunk 0: characters   0–100
    Chunk 1: characters  80–180   ← starts 20 chars back (the overlap)
    Chunk 2: characters 160–260

Key parameters:
  - chunk_size     → maximum characters (or tokens) per chunk
  - chunk_overlap  → characters shared between consecutive chunks

Splitters covered:
  1. CharacterTextSplitter          → split on a single separator character
  2. RecursiveCharacterTextSplitter → tries multiple separators in order (recommended)
  3. TokenTextSplitter              → split by token count, not character count
  4. MarkdownHeaderTextSplitter     → split Markdown by heading hierarchy
  5. HTMLHeaderTextSplitter         → split HTML by heading tags

Rule of thumb:
  - Use RecursiveCharacterTextSplitter for almost everything (it's the safest default)
  - Use TokenTextSplitter when you need to stay within an exact token budget
  - Use Markdown/HTML splitters when your source has clear structural headings
"""

import os

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
PDF_PATH = os.path.join(DATA_DIR, "laptop_policy.pdf")

# Load the PDF once and join all pages into a single text string.
# All splitters below operate on this same text so you can compare results.
raw_docs = PyPDFLoader(PDF_PATH).load()
full_text = "\n\n".join(doc.page_content for doc in raw_docs)
print(f"Full text length: {len(full_text)} characters across {len(raw_docs)} pages\n")


def show_chunks(chunks: list, label: str) -> None:
    """Print a summary and preview of the first two chunks."""
    print(f"\n=== {label} ===")
    print(f"Total chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks[:2]):
        preview = chunk.page_content[:150].replace("\n", " ")
        print(f"  [{i}] ({len(chunk.page_content)} chars) {preview}...")


# ── 1. CharacterTextSplitter ────────────────────────────────────────────────────────────────
# The simplest splitter. Splits ONLY on the single separator you give it.
# If the separator doesn't appear in the text, you get one massive chunk.
#
# When to use: quick prototyping or when your text has a known, reliable separator.
# Avoid for production — the separator might not always appear where you expect.
from langchain_text_splitters import CharacterTextSplitter  # noqa: E402

char_splitter = CharacterTextSplitter(
    separator="\n\n",   # split at blank lines (paragraph breaks)
    chunk_size=500,     # maximum 500 characters per chunk
    chunk_overlap=50,   # repeat last 50 characters in the next chunk
)
chunks = char_splitter.create_documents([full_text])
show_chunks(chunks, "1. CharacterTextSplitter (separator='\\n\\n')")
# Notice: chunk count depends entirely on how many blank lines exist in the PDF.
# If a section has no blank lines, CharacterTextSplitter won't split it further
# even if it exceeds chunk_size.


# ── 2. RecursiveCharacterTextSplitter ──────────────────────────────────────────────────────
# The recommended default. Tries a list of separators in order:
#   ["\n\n", "\n", " ", ""]
# It starts with the most meaningful break (paragraph) and falls back to
# smaller breaks only if a chunk still exceeds chunk_size.
#
# This means chunks are as semantically complete as possible — it won't split
# mid-sentence unless absolutely necessary.
from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: E402

recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    # separators default: ["\n\n", "\n", " ", ""]
    # You can customise: separators=[".", "\n", " "]
)
chunks = recursive_splitter.create_documents([full_text])
show_chunks(chunks, "2. RecursiveCharacterTextSplitter (recommended default)")
# Compare chunk count with CharacterTextSplitter — Recursive usually produces
# more, smaller, cleaner chunks because it never leaves an oversized paragraph
# intact.


# ── 3. TokenTextSplitter ──────────────────────────────────────────────────────────────────
# Splits by TOKEN count rather than character count.
#
# Why this matters: LLM context limits are measured in TOKENS, not characters.
# 1 token ≈ 4 characters on average for English text.
# A chunk of 500 characters might be ~125 tokens.
#
# Use TokenTextSplitter when you need to guarantee a chunk fits within a
# specific token budget (e.g. embedding model limit of 8191 tokens).
from langchain_text_splitters import TokenTextSplitter  # noqa: E402

token_splitter = TokenTextSplitter(
    chunk_size=150,    # 150 tokens per chunk (≈600 characters on average)
    chunk_overlap=20,  # 20-token overlap between chunks
)
chunks = token_splitter.create_documents([full_text])
show_chunks(chunks, "3. TokenTextSplitter (150 tokens/chunk)")
# Notice: chunk sizes in characters will vary — some tokens are short ("a"),
# some are long ("encryption"). The token count is what stays consistent.


# ── 4. MarkdownHeaderTextSplitter ────────────────────────────────────────────────────────
# Splits Markdown documents at heading boundaries (#, ##, ###).
# The heading text is stored in the chunk's METADATA — not stripped away.
#
# This is powerful for documentation, wikis, and policy docs written in Markdown
# because you can later filter or cite chunks by their section name.
#
# Note: this splitter takes raw text, not Document objects — use split_text().
from langchain_text_splitters import MarkdownHeaderTextSplitter  # noqa: E402

markdown_text = """
# Company Laptop Policy

## 1. Purpose
This policy establishes guidelines for the acceptable use of company laptops.

## 2. Acceptable Use
Employees may use laptops for business tasks and limited personal use.

### 2.1 Software
Only install software approved by the IT department.

### 2.2 Internet
Avoid accessing inappropriate websites.

## 3. Security
All devices must have full-disk encryption enabled.
"""

md_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#",  "h1"),   # top-level heading → stored as metadata["h1"]
        ("##", "h2"),   # section heading   → stored as metadata["h2"]
        ("###","h3"),   # subsection        → stored as metadata["h3"]
    ]
)
chunks = md_splitter.split_text(markdown_text)
print("\n=== 4. MarkdownHeaderTextSplitter ===")
print(f"Total chunks: {len(chunks)}")
for chunk in chunks:
    # metadata tells you WHICH section this chunk came from
    # page_content is the body text under that heading
    print(f"  metadata={chunk.metadata}")
    print(f"  content: {chunk.page_content[:80]}...")
    print()
# Key observation: the heading text appears in metadata, not in page_content.
# This lets you cite the exact section in your RAG answer.


# ── 5. HTMLHeaderTextSplitter ─────────────────────────────────────────────────────────────
# Same idea as MarkdownHeaderTextSplitter but for HTML documents.
# Splits on <h1>, <h2>, etc. and stores heading text in metadata.
#
# Use when loading web pages, exported HTML docs, or CMS content.
# Strips HTML tags from the content — you get clean text with structural metadata.
from langchain_text_splitters import HTMLHeaderTextSplitter  # noqa: E402

html_text = """
<html>
<body>
<h1>Laptop Policy</h1>
<p>This document outlines our laptop usage policy.</p>
<h2>Acceptable Use</h2>
<p>Use your laptop for work tasks and limited personal use during breaks.</p>
<h2>Security</h2>
<p>Enable full-disk encryption and use the company VPN on public networks.</p>
</body>
</html>
"""
html_splitter = HTMLHeaderTextSplitter(
    headers_to_split_on=[
        ("h1", "header1"),  # <h1> text → metadata["header1"]
        ("h2", "header2"),  # <h2> text → metadata["header2"]
    ]
)
chunks = html_splitter.split_text(html_text)
print("\n=== 5. HTMLHeaderTextSplitter ===")
print(f"Total chunks: {len(chunks)}")
for chunk in chunks:
    print(f"  metadata={chunk.metadata}")
    print(f"  content: {chunk.page_content[:80]}...")
    print()
# HTML tags are stripped — page_content contains only the visible text.


# ── Summary: which splitter to choose ────────────────────────────────────────────────
print("\n--- Splitter comparison summary ---")
print("CharacterTextSplitter       → simple, fast, rigid. Use for prototyping.")
print("RecursiveCharacterTextSplitter → smart fallback chain. USE THIS by default.")
print("TokenTextSplitter           → precise token budgeting. Use for embedding limits.")
print("MarkdownHeaderTextSplitter  → structure-aware. Use for Markdown docs.")
print("HTMLHeaderTextSplitter      → structure-aware. Use for HTML/web content.")
print("\n✅ Splitting complete. Next step: embed chunks and store in a vector database.")
