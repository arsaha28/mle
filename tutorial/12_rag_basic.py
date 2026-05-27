"""
12 - Basic RAG (Retrieval-Augmented Generation)
================================================
Concept: RAG = Retrieve relevant documents + pass them as context to an LLM.
Instead of the model relying only on its training data, it reads real documents
at query time. This reduces hallucination and keeps answers grounded.

Why RAG instead of just asking the LLM directly?
  Without RAG: the LLM answers from its training data only. It cannot know:
    - Your company's internal policies
    - Documents created after its training cutoff
    - Real-time or frequently updated information
  With RAG: relevant text is pulled from YOUR documents and injected into the
  prompt as context. The LLM reads that context and answers from it.
  Hallucination drops because the model is quoting your documents, not guessing.

The full pipeline in two phases:
  Indexing phase (done once, usually offline):
    PDF → load pages → split into chunks → embed each chunk → store in vector DB

  Query phase (done per user question):
    question → embed → find similar chunks → inject into prompt → LLM → answer

Visual:
  [PDF]
    │ load
    ▼
  [pages]  ←── one Document per page
    │ split
    ▼
  [chunks] ←── 500-char overlapping windows
    │ embed
    ▼
  [vectors] ←── 1536-float representations of meaning
    │ FAISS index
    ▼
  [retriever]
    │ ← question flows IN here
    ▼
  [top-k chunks] → format → {context}
                              │
  question ─────────────────►│
                              ▼
                         [RAG prompt]
                              │
                              ▼
                            [LLM]
                              │
                              ▼
                           [answer]

Key components used in this file:
  - PyPDFLoader                  → load the PDF (file 08)
  - RecursiveCharacterTextSplitter → split into chunks (file 09)
  - OpenAIEmbeddings + FAISS     → embed and index (file 10)
  - vectorstore.as_retriever()   → wrap in retriever interface (file 11)
  - ChatPromptTemplate           → structure the prompt with {context} + {question}
  - RunnableParallel             → run retriever and passthrough simultaneously
  - RunnablePassthrough          → pass the question through unchanged
  - ChatOpenAI + StrOutputParser → generate and extract the answer
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

# ── Step 1: Load ──────────────────────────────────────────────────────────────
# PyPDFLoader returns one Document per page.
# Each Document has:
#   doc.page_content → the text on that page
#   doc.metadata     → {"source": "path/to/file.pdf", "page": 0, ...}
#
# For the indexing pipeline we need the text; metadata is carried forward so
# we can later cite which page an answer came from.
print("Step 1: Loading PDF...")
docs = PyPDFLoader(PDF_PATH).load()
print(f"  → Loaded {len(docs)} pages")


# ── Step 2: Split ─────────────────────────────────────────────────────────────
# A full page of text is too large to be a useful retrieval unit.
# We split each page into 500-character chunks with 50-char overlap.
#
# Why 500 chars?
#   - Small enough to be specific (embedding captures one idea)
#   - Large enough to contain a complete sentence or two
#   - The embedding model can handle up to 8191 tokens — 500 chars is ~125 tokens
#
# Why overlap?
#   - A sentence split across a boundary loses its meaning
#   - Overlap ensures no sentence is completely missing from every chunk
print("Step 2: Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)
print(f"  → {len(chunks)} chunks")
# Each chunk inherits the parent document's metadata (source, page number).
# This is how we can later say "this answer came from page 3".


# ── Step 3: Embed + Store ─────────────────────────────────────────────────────
# from_documents() does two things:
#   a) Calls the OpenAI embeddings API to convert each chunk → 1536-float vector
#   b) Builds a FAISS index from those vectors (in memory)
#
# as_retriever() wraps the index in the standard retriever interface.
# search_kwargs={"k": 4} means: return the 4 most similar chunks per query.
#
# Choosing k:
#   k=2 → tight, precise context; may miss relevant details
#   k=4 → balanced (default); covers most questions
#   k=8 → broad context; more tokens consumed; diminishing returns
print("Step 3: Embedding and indexing...")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
print("  → Vector store ready")


# ── Step 4: Build the RAG prompt ──────────────────────────────────────────────
# The RAG prompt has two variables:
#   {context}  → the retrieved chunks, joined into a single string
#   {question} → the user's original question
#
# System message design choices:
#   "Answer using ONLY the context" → prevents hallucination from training data
#   "If the answer is not in the context, say ..." → graceful fallback instead
#   of making something up
#
# The context is injected in the system message so the LLM treats it as
# authoritative background knowledge, not as a user message.
rag_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful HR assistant. Answer questions about the company laptop policy "
     "using ONLY the context provided below. If the answer is not in the context, say "
     "'I could not find that information in the policy.'\n\n"
     "Context:\n{context}"),
    ("human", "{question}"),
])

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# temperature=0 → deterministic, factual answers
# For a Q&A bot over documents, we want consistency, not creativity.

parser = StrOutputParser()
# StrOutputParser extracts the text content from the AIMessage the LLM returns.
# Without it, you'd get an AIMessage object instead of a plain string.


# ── Step 5: Assemble the RAG chain ────────────────────────────────────────────
# format_docs joins the retrieved chunks with double newlines.
# This is the {context} string that goes into the prompt.
# Each chunk is separated clearly so the LLM can read them as distinct passages.
def format_docs(docs: list) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

# RunnableParallel runs two sub-chains simultaneously on the SAME input (the question):
#   "context"  branch: question → retriever (finds chunks) → format_docs (joins them)
#   "question" branch: question → RunnablePassthrough() (returns it unchanged)
#
# The output is a dict: {"context": "chunk1\n\nchunk2\n\n...", "question": "..."}
# This dict exactly matches the two variables in rag_prompt.
#
# RunnablePassthrough is needed because LCEL pipes always take one input.
# Without it there would be no way to pass the question forward while the
# retriever branch is running.
rag_chain = (
    RunnableParallel(
        context=(retriever | format_docs),
        question=RunnablePassthrough(),
    )
    | rag_prompt
    | llm
    | parser
)

# Full data flow, step by step:
#   "What is the password length requirement?"         ← invoke() input
#     │
#     ├─► retriever.invoke(question)
#     │       → FAISS finds 4 nearest chunks
#     │       → format_docs joins them into one string
#     │       → stored as context="..."
#     │
#     └─► RunnablePassthrough()
#             → question="What is the password length requirement?"
#     │
#     ▼
#   {"context": "...", "question": "What is the password..."}
#     │
#     ▼
#   rag_prompt.format_messages(...)
#     → [SystemMessage(content="...Context:\n..."), HumanMessage(content="...")]
#     │
#     ▼
#   llm.invoke([SystemMessage, HumanMessage])
#     → AIMessage(content="Passwords must be at least 12 characters...")
#     │
#     ▼
#   parser.invoke(AIMessage)
#     → "Passwords must be at least 12 characters..."    ← final output


# ── Step 6: Ask questions ─────────────────────────────────────────────────────
# These four questions cover different parts of the policy document.
# Each call: embed question → retrieve 4 chunks → fill prompt → LLM → answer.
questions = [
    "What activities are prohibited on company laptops?",
    "What should an employee do if their laptop is stolen?",
    "Are employees allowed to use personal cloud storage for work files?",
    "What is the password length requirement?",
]

print("\n" + "="*60)
print("RAG Q&A — Company Laptop Policy")
print("="*60)
for question in questions:
    print(f"\nQ: {question}")
    answer = rag_chain.invoke(question)
    print(f"A: {answer}")
# Notice: answers are grounded in the document.
# If you ask a question whose answer isn't in the PDF (e.g. "What is the CEO's name?")
# the model should respond with the fallback: "I could not find that information..."
# — it does NOT make something up. That's the value of the system message constraint.


# ── Step 7: Show which source chunks were used ────────────────────────────────
# In production you usually want to show users WHERE an answer came from.
# This "sources" pattern runs the retriever a second time to get the raw chunks.
#
# RunnableParallel here runs two independent chains on the same question:
#   "answer"  → the full RAG chain (returns a string)
#   "sources" → just the retriever (returns a list of Documents)
#
# Both branches receive the same question string as input.
# The result is a dict with both keys populated.
print("\n=== Source chunks for last question ===")
source_chain = RunnableParallel(
    answer=rag_chain,
    sources=retriever,
)
result = source_chain.invoke(questions[-1])
print("Answer:", result["answer"])
print("\nSources used:")
for doc in result["sources"]:
    print(f"  page={doc.metadata.get('page')} | {doc.page_content[:100]}...")
# You can see exactly which page and which text the answer was drawn from.
# This is essential for trust and auditability in enterprise RAG applications.


# ── Summary ───────────────────────────────────────────────────────────────────
print("\n--- RAG pipeline recap ---")
print("Load     → PyPDFLoader: one Document per page")
print("Split    → RecursiveCharacterTextSplitter: 500-char chunks, 50-char overlap")
print("Embed    → OpenAIEmbeddings: each chunk → 1536-float vector")
print("Index    → FAISS: in-memory nearest-neighbour index")
print("Retrieve → as_retriever(k=4): top-4 similar chunks per question")
print("Prompt   → ChatPromptTemplate: injects {context} + {question}")
print("Generate → ChatOpenAI: reads context, produces grounded answer")
print("Parse    → StrOutputParser: extracts plain text from AIMessage")
print("\n✅ Done. Next step (file 13): add memory so the RAG chain can handle follow-up questions.")
