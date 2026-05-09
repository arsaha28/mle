"""
12 - Basic RAG (Retrieval-Augmented Generation)
================================================
Concept: RAG = retrieve relevant documents + pass them as context to an LLM.
Reduces hallucination by grounding answers in real documents.

Pipeline:
  PDF -> load -> split -> embed -> store
                                     |
  question -> retrieve -> prompt -> LLM -> answer
"""

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
PDF_PATH = os.path.join(DATA_DIR, "laptop_policy.pdf")

print("Step 1: Loading PDF...")
docs = PyPDFLoader(PDF_PATH).load()
print(f"  -> {len(docs)} pages")

print("Step 2: Splitting...")
chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)
print(f"  -> {len(chunks)} chunks")

print("Step 3: Embedding and indexing...")
vectorstore = FAISS.from_documents(chunks, OpenAIEmbeddings(model="text-embedding-3-small"))
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
print("  -> Vector store ready")

rag_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful HR assistant. Answer using ONLY the context below. "
     "If not found, say 'I could not find that in the policy.'\n\nContext:\n{context}"),
    ("human", "{question}"),
])
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = StrOutputParser()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    RunnableParallel(context=(retriever | format_docs), question=RunnablePassthrough())
    | rag_prompt | llm | parser
)

questions = [
    "What activities are prohibited on company laptops?",
    "What should an employee do if their laptop is stolen?",
    "Are employees allowed to use personal cloud storage?",
    "What is the password length requirement?",
]

print("\n" + "="*60 + "\nRAG Q&A - Company Laptop Policy\n" + "="*60)
for q in questions:
    print(f"\nQ: {q}")
    print(f"A: {rag_chain.invoke(q)}")

print("\n=== Source chunks for last question ===")
result = RunnableParallel(answer=rag_chain, sources=retriever).invoke(questions[-1])
print("Answer:", result["answer"])
for doc in result["sources"]:
    print(f"  page={doc.metadata.get('page')} | {doc.page_content[:100]}...")
