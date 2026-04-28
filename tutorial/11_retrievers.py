"""
11 - Retrievers
===============
Concept: A retriever wraps a vector store and fetches relevant documents.
Different strategies trade off speed, diversity, and relevance.

Retrievers covered:
  - Basic VectorStoreRetriever      -> top-k similarity
  - MMR Retriever                   -> diversity-aware
  - MultiQueryRetriever             -> query expansion
  - ContextualCompressionRetriever  -> extract only relevant parts
  - SelfQueryRetriever              -> natural language metadata filters
"""

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
PDF_PATH = os.path.join(DATA_DIR, "laptop_policy.pdf")

print("Building vector store...")
docs = PyPDFLoader(PDF_PATH).load()
chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(chunks, embeddings)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
QUERY = "What security measures are required for remote workers?"

# 1. Basic
print("\n=== 1. Basic Retriever (top-k) ===")
basic_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
for doc in basic_retriever.invoke(QUERY):
    print(f"  -> {doc.page_content[:120]}...")

# 2. MMR
print("\n=== 2. MMR Retriever (diversity-aware) ===")
mmr_retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.7})
for doc in mmr_retriever.invoke(QUERY):
    print(f"  -> {doc.page_content[:120]}...")

# 3. MultiQueryRetriever
print("\n=== 3. MultiQueryRetriever (query expansion) ===")
from langchain.retrievers import MultiQueryRetriever
multi_retriever = MultiQueryRetriever.from_llm(retriever=basic_retriever, llm=llm)
docs_retrieved = multi_retriever.invoke(QUERY)
print(f"Retrieved {len(docs_retrieved)} docs (merged from multiple query variants)")
for doc in docs_retrieved[:3]:
    print(f"  -> {doc.page_content[:120]}...")

# 4. ContextualCompressionRetriever
print("\n=== 4. ContextualCompressionRetriever ===")
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
compression_retriever = ContextualCompressionRetriever(
    base_compressor=LLMChainExtractor.from_llm(llm),
    base_retriever=basic_retriever,
)
for doc in compression_retriever.invoke(QUERY):
    print(f"  -> {doc.page_content[:200]}...")

# 5. SelfQueryRetriever
print("\n=== 5. SelfQueryRetriever (natural language -> metadata filter) ===")
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.vectorstores import Chroma
from langchain.chains.query_constructor.base import AttributeInfo
metadata_field_info = [
    AttributeInfo(name="page", description="Page number in the PDF", type="integer"),
    AttributeInfo(name="source", description="Source file path", type="string"),
]
self_query_retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=Chroma.from_documents(chunks, embeddings, collection_name="self_query_demo"),
    document_contents="Company laptop usage policy sections",
    metadata_field_info=metadata_field_info,
)
results = self_query_retriever.invoke("What does page 2 say about software installation?")
for doc in results[:2]:
    print(f"  page={doc.metadata.get('page')} | {doc.page_content[:120]}...")
