"""
10 - Embeddings & Vector Stores
================================
Concept: An *embedding* converts text into a list of numbers (a vector) that
captures semantic meaning. The key insight is that similar meanings produce
vectors that are mathematically close — this is what makes semantic search work.

What is a vector?
  A vector is just a list of floats, e.g. [0.021, -0.134, 0.087, ...].
  text-embedding-3-small produces 1536 numbers per text.
  Two texts with similar meaning will have similar numbers at each position.

Why not just use keyword search?
  Keyword search: "laptop security" only matches documents with those exact words.
  Semantic search: "laptop security" also finds "device encryption policy" and
  "endpoint protection requirements" — because they MEAN similar things.

Flow:
  text → embed() → vector [1536 floats] → stored in index
  query → embed() → vector [1536 floats] → find nearest stored vectors → return docs

Covered:
  1. OpenAIEmbeddings          → call the OpenAI API to embed text
  2. FAISS                     → fast in-memory index (no server needed)
  3. Similarity search         → find k nearest documents
  4. Similarity search + score → same, but see how relevant each result is
  5. Save / load FAISS         → persist the index to disk and reload it
  6. Chroma                    → persistent local vector store with a simple API
  7. Direct cosine similarity  → understand the maths behind the search
"""

import os

import numpy as np
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
PDF_PATH = os.path.join(DATA_DIR, "laptop_policy.pdf")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index")
CHROMA_PATH = os.path.join(DATA_DIR, "chroma_db")

# ── Prepare documents ───────────────────────────────────────────────────────────────────
# Load the PDF and split into chunks — same pattern as file 09.
# Each chunk will be embedded individually and stored as one vector in the index.
print("Loading and splitting PDF...")
docs = PyPDFLoader(PDF_PATH).load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)
print(f"  → {len(chunks)} chunks ready for embedding")

# ── Embeddings model ────────────────────────────────────────────────────────────────────
# OpenAIEmbeddings wraps the OpenAI embeddings API.
# text-embedding-3-small:
#   - Output: 1536-dimensional vector per text
#   - Fast and cheap — ideal for tutorials and production RAG pipelines
#   - Other option: text-embedding-3-large (3072 dims, higher quality, more cost)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Quick demo: what does an embedding actually look like?
sample_vec = embeddings.embed_query("What is a laptop policy?")
print(f"\nEmbedding demo: {len(sample_vec)} floats, first 5: {[round(x,4) for x in sample_vec[:5]]}")
# → 1536 numbers that encode the meaning of that sentence


# ── 1. FAISS — in-memory vector store ──────────────────────────────────────────────────
# FAISS (Facebook AI Similarity Search) builds an index entirely in memory.
# No database server needed — ideal for prototyping and small datasets.
#
# from_documents() does two things in one call:
#   a) Embeds every chunk by calling the OpenAI API (one API call per batch)
#   b) Builds the FAISS index from those vectors
print("\n=== 1. FAISS — building the index ===")
from langchain_community.vectorstores import FAISS  # noqa: E402

faiss_store = FAISS.from_documents(chunks, embeddings)
print(f"  Index built with {faiss_store.index.ntotal} vectors")

# similarity_search() embeds the query and returns the k nearest document chunks.
# It does NOT call an LLM — this is pure vector maths.
query = "What are employees prohibited from doing on company laptops?"
results = faiss_store.similarity_search(query, k=3)
print(f"\nQuery: '{query}'")
print(f"Top {len(results)} results:")
for i, doc in enumerate(results):
    print(f"  [{i}] page={doc.metadata.get('page')} | {doc.page_content[:120]}...")


# ── 2. Similarity search with relevance scores ────────────────────────────────────────
# Same as similarity_search() but also returns a score for each result.
# Score range: 0.0 (unrelated) → 1.0 (identical meaning).
# Use scores to filter out low-quality results in production.
print("\n=== 2. Similarity search with relevance scores ===")
results_with_scores = faiss_store.similarity_search_with_relevance_scores(query, k=3)
print(f"Query: '{query}'")
for doc, score in results_with_scores:
    bar = "█" * int(score * 20)  # visual bar
    print(f"  score={score:.4f} {bar:<20} | {doc.page_content[:80]}...")
# A score above ~0.75 generally indicates a genuinely relevant chunk.
# If all scores are low, the document may not contain an answer.


# ── 3. Save and reload FAISS index ───────────────────────────────────────────────────────
# Embedding all documents takes time and API calls. Save the index once
# and reload it on subsequent runs — no re-embedding needed.
#
# save_local() writes two files:
#   faiss_index/index.faiss  → the binary vector index
#   faiss_index/index.pkl    → the document store (text + metadata)
print("\n=== 3. Saving and loading FAISS index ===")
faiss_store.save_local(FAISS_INDEX_PATH)
print(f"  Saved to {FAISS_INDEX_PATH}/")

# allow_dangerous_deserialization=True is required because .pkl files can
# contain arbitrary Python objects — LangChain forces you to opt in explicitly.
loaded_store = FAISS.load_local(
    FAISS_INDEX_PATH,
    embeddings,
    allow_dangerous_deserialization=True,
)
result = loaded_store.similarity_search("password requirements", k=1)
print(f"  Reloaded OK — {loaded_store.index.ntotal} vectors in index")
print(f"  Sample result: {result[0].page_content[:100]}...")


# ── 4. Chroma — persistent vector store ───────────────────────────────────────────────
# Chroma is a dedicated vector database that persists automatically to disk.
# Unlike FAISS (which you must save manually), Chroma writes every addition
# to persist_directory immediately.
#
# Chroma vs FAISS:
#   FAISS  → faster for pure ANN search; manual save/load; single process
#   Chroma → auto-persistent; supports metadata filtering; easier API
print("\n=== 4. Chroma — persistent vector store ===")
from langchain_community.vectorstores import Chroma  # noqa: E402

chroma_store = Chroma.from_documents(
    chunks,
    embeddings,
    persist_directory=CHROMA_PATH,     # automatically saved here
    collection_name="laptop_policy",   # logical namespace within Chroma
)
print(f"  Collection '{chroma_store._collection.name}' created at {CHROMA_PATH}/")

result = chroma_store.similarity_search("remote work security requirements", k=2)
print(f"  Results for 'remote work security requirements':")
for doc in result:
    print(f"    → page={doc.metadata.get('page')} | {doc.page_content[:120]}...")


# ── 5. Direct cosine similarity ───────────────────────────────────────────────────────────
# Under the hood, FAISS and Chroma find similar vectors using cosine similarity:
#   cos(θ) = (A · B) / (|A| × |B|)
#   Result: -1 (opposite) to +1 (identical direction) — in practice 0 to 1
#
# This section shows the raw maths so you can see exactly how the search works.
# In production you'd never write this yourself — the vector store handles it.
print("\n=== 5. Direct cosine similarity (the maths behind search) ===")

def cosine(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

texts = [
    "Employees must use the VPN on public Wi-Fi networks.",      # relevant
    "The sky is blue and the sun is bright today.",              # irrelevant
    "All devices require full-disk encryption when travelling.", # relevant
]
query_text = "network security requirements for remote workers"

vecs = embeddings.embed_documents(texts)
query_vec = embeddings.embed_query(query_text)

print(f"Query: '{query_text}'\n")
for text, vec in zip(texts, vecs):
    sim = cosine(query_vec, vec)
    bar = "█" * int(sim * 30)
    print(f"  {sim:.4f} {bar}")
    print(f"         {text}\n")
print("  → High score = semantically close to the query.")
print("  → The sky sentence scores low despite being valid English — meaning differs.")
print("\n✅ Done. Next step (file 11): use retrievers to query these stores in a chain.")
