"""
13 - Advanced RAG Patterns
===========================
Concept: Production RAG needs more sophistication than basic retrieval.

Patterns covered:
  1. Conversational RAG  -> remembers previous turns
  2. Multi-query RAG     -> generates query variants for better recall
  3. RAG with reranking  -> scores and filters retrieved chunks
"""

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
PDF_PATH = os.path.join(DATA_DIR, "laptop_policy.pdf")

docs = PyPDFLoader(PDF_PATH).load()
chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)
vectorstore = FAISS.from_documents(chunks, OpenAIEmbeddings(model="text-embedding-3-small"))
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = StrOutputParser()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Pattern 1: Conversational RAG
print("=" * 60 + "\nPattern 1: Conversational RAG\n" + "=" * 60)
condense_chain = ChatPromptTemplate.from_messages([
    ("system", "Rewrite the follow-up as a standalone question using the chat history."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
]) | llm | parser

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer using only the context below.\n\nContext:\n{context}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

def conversational_rag(question, history):
    standalone_q = condense_chain.invoke({"question": question, "history": history}) if history else question
    context = format_docs(retriever.invoke(standalone_q))
    return (rag_prompt | llm | parser).invoke({"context": context, "history": history, "question": question})

history = []
for q in ["What are the main security requirements?", "What about when working remotely?", "Do they need a VPN?"]:
    print(f"\nUser: {q}")
    answer = conversational_rag(q, history)
    print(f"Assistant: {answer}")
    history += [HumanMessage(content=q), AIMessage(content=answer)]

# Pattern 2: Multi-query RAG
print("\n" + "=" * 60 + "\nPattern 2: Multi-query RAG\n" + "=" * 60)
expand_chain = ChatPromptTemplate.from_messages([
    ("system", "Generate 3 different versions of the question for better retrieval. One per line."),
    ("human", "{question}"),
]) | llm | parser

def multi_query_retrieve(question):
    alternatives = expand_chain.invoke({"question": question})
    queries = [question] + [q.strip() for q in alternatives.strip().split("\n") if q.strip()]
    print(f"  Queries used: {queries}")
    seen, all_docs = set(), []
    for q in queries:
        for doc in retriever.invoke(q):
            key = doc.page_content[:50]
            if key not in seen:
                seen.add(key)
                all_docs.append(doc)
    return all_docs

question = "How should employees protect company data?"
print(f"\nQ: {question}")
retrieved = multi_query_retrieve(question)
print(f"  {len(retrieved)} unique docs retrieved")
answer_prompt = ChatPromptTemplate.from_messages([("system", "Answer using only the context.\n\nContext:\n{context}"), ("human", "{question}")])
print("A:", (answer_prompt | llm | parser).invoke({"context": format_docs(retrieved), "question": question}))

# Pattern 3: RAG with Reranking
print("\n" + "=" * 60 + "\nPattern 3: RAG with Reranking\n" + "=" * 60)
rerank_prompt = ChatPromptTemplate.from_messages([
    ("system", "Score how relevant this document is to the query from 0-10. Reply with just the integer."),
    ("human", "Query: {query}\n\nDocument:\n{document}"),
])

def rerank(query, docs, top_n=3):
    scored = []
    for doc in docs:
        try:
            score = int((rerank_prompt | llm | parser).invoke({"query": query, "document": doc.page_content}).strip())
        except ValueError:
            score = 0
        scored.append((score, doc))
    return [doc for _, doc in sorted(scored, reverse=True)[:top_n]]

question = "What happens if an employee violates the laptop policy?"
print(f"\nQ: {question}")
candidates = vectorstore.as_retriever(search_kwargs={"k": 8}).invoke(question)
top_docs = rerank(question, candidates)
print(f"  Kept top {len(top_docs)} after reranking")
print("A:", (answer_prompt | llm | parser).invoke({"context": format_docs(top_docs), "question": question}))
