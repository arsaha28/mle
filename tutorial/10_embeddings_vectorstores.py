"""
10 - Embeddings & Vector Stores
================================
Concept: Embeddings convert text to vectors that capture meaning.
Similar texts have vectors that are mathematically close.
Vector stores find the closest vectors to a query efficiently.

Flow: text -> embed -> store -> query -> retrieve similar texts

Covered:
  - OpenAIEmbeddings
  - FAISS (in-memory)
  - Chroma (persistent)
  - Similarity search with scores
  - Save/load FAISS index
  - Direct cosine similarity comparison
"""

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
PDF_PATH = os.path.join(DATA_DIR, "laptop_policy.pdf")
FAISS_PATH = os.path.join(DATA_DIR, "faiss_index")

print("Loading and splitting PDF...")
docs = PyPDFLoader(PDF_PATH).load()
chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)
print(f"  -> {len(chunks)} chunks ready")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 1. FAISS
print("\n=== 1. FAISS - in-memory vector store ===")
from langchain_community.vectorstores import FAISS
faiss_store = FAISS.from_documents(chunks, embeddings)
query = "What are employees prohibited from doing on company laptops?"
results = faiss_store.similarity_search(query, k=3)
for i, doc in enumerate(results):
    print(f"  [{i}] page={doc.metadata.get('page')} | {doc.page_content[:100]}...")

# 2. Similarity search with scores
print("\n=== 2. Similarity search with relevance scores ===")
for doc, score in faiss_store.similarity_search_with_relevance_scores(query, k=3):
    print(f"  score={score:.4f} | {doc.page_content[:100]}...")

# 3. Save and reload FAISS
print("\n=== 3. Save and reload FAISS index ===")
faiss_store.save_local(FAISS_PATH)
loaded = FAISS.load_local(FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
print(f"  Reloaded OK. Sample: {loaded.similarity_search('password', k=1)[0].page_content[:80]}...")

# 4. Chroma
print("\n=== 4. Chroma - persistent vector store ===")
from langchain_community.vectorstores import Chroma
chroma_store = Chroma.from_documents(chunks, embeddings, persist_directory=os.path.join(DATA_DIR, "chroma_db"), collection_name="laptop_policy")
for doc in chroma_store.similarity_search("remote work security", k=2):
    print(f"  -> {doc.page_content[:100]}...")

# 5. Direct cosine similarity
print("\n=== 5. Direct embedding comparison ===")
import numpy as np
texts = ["Employees must use the VPN on public networks.", "The sky is blue.", "All devices require encryption."]
vecs = embeddings.embed_documents(texts)
query_vec = embeddings.embed_query("network security for remote workers")
def cosine(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
for text, vec in zip(texts, vecs):
    print(f"  sim={cosine(query_vec, vec):.4f} | {text}")
