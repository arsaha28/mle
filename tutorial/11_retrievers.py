"""
11 - Retrievers
===============
Concept: A *retriever* is a standardised interface that takes a query string
and returns a list of relevant Document objects. It wraps a vector store but
adds smarter strategies on top of raw similarity search.

Why do we need retrievers beyond basic similarity search?
  Basic search: returns the k most similar chunks — but they may all say the
  same thing (redundant), miss relevant chunks phrased differently, or include
  mostly-irrelevant sentences buried inside a long chunk.

  Retrievers fix each of these problems:
    MMR                  → removes redundant results (diversity)
    MultiQueryRetriever  → catches relevant chunks with different phrasing (recall)
    ContextualCompression → strips irrelevant sentences from chunks (precision)
    SelfQueryRetriever   → filters by metadata so only the right pages are searched

All retrievers share the same interface: retriever.invoke(query) → list[Document]
This means you can swap one for another in a RAG chain without changing anything else.

Retrievers covered:
  1. Basic VectorStoreRetriever      → simple top-k similarity search
  2. MMR Retriever                   → Maximum Marginal Relevance (diversity-aware)
  3. MultiQuery                      → generates multiple query variants, merges results
  4. Contextual Compression          → strips irrelevant parts from returned chunks
  5. Metadata Filtering              → translates natural language filters into metadata queries
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

# ── Shared setup ────────────────────────────────────────────────────────────────
# Build one FAISS vector store and reuse it across all retriever examples.
# This avoids re-embedding the same document multiple times.
print("Building vector store...")
docs = PyPDFLoader(PDF_PATH).load()
chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(chunks, embeddings)
print(f"  → {len(chunks)} chunks indexed\n")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# The same query is used across all retrievers so you can directly compare results.
QUERY = "What security measures are required for remote workers?"


# ── 1. Basic VectorStoreRetriever ─────────────────────────────────────────────
# The simplest retriever — embeds the query and returns the k most similar chunks.
# No LLM involved; just vector maths. Fast and cheap.
#
# as_retriever() converts any vector store into a retriever.
# search_kwargs passes options to the underlying similarity_search() call.
#
# When to use: default choice when speed matters and query phrasing is reliable.
print("=== 1. Basic Retriever (top-k similarity) ===")
basic_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
docs_retrieved = basic_retriever.invoke(QUERY)
print(f"Query: '{QUERY}'")
print(f"Retrieved {len(docs_retrieved)} docs:")
for doc in docs_retrieved:
    print(f"  page={doc.metadata.get('page')} | {doc.page_content[:120]}...")
# Limitation: if two chunks say nearly the same thing, both get returned.
# That's wasted context — the LLM sees duplicate information.


# ── 2. MMR Retriever ─────────────────────────────────────────────────────────────
# MMR = Maximum Marginal Relevance.
# Instead of returning the k most similar chunks, it iteratively selects chunks
# that are BOTH relevant to the query AND different from already-selected chunks.
#
# How it works:
#   1. Fetch a larger candidate pool (fetch_k=10)
#   2. Pick the most relevant chunk first
#   3. For each remaining pick, score = λ * relevance - (1-λ) * similarity_to_selected
#   4. Repeat until k chunks are selected
#
# lambda_mult controls the trade-off:
#   0.0 → maximise diversity (ignore relevance entirely)
#   1.0 → maximise relevance (same as basic search)
#   0.5 → balanced (default)
#
# When to use: when basic search returns redundant chunks covering the same point.
print("\n=== 2. MMR Retriever (diversity-aware) ===")
mmr_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,           # number of chunks to return
        "fetch_k": 10,    # candidate pool size before MMR selection
        "lambda_mult": 0.7,  # 0.7 = slightly favour relevance over diversity
    },
)
docs_retrieved = mmr_retriever.invoke(QUERY)
print(f"Retrieved {len(docs_retrieved)} diverse docs (lambda_mult=0.7):")
for doc in docs_retrieved:
    print(f"  page={doc.metadata.get('page')} | {doc.page_content[:120]}...")
# Compare with basic retriever above — MMR results should cover different aspects
# of the topic rather than repeating the same point from different angles.


# ── 3. MultiQuery (query expansion) ──────────────────────────────────────────────
# Problem: a single query phrasing might miss relevant chunks that use
# different vocabulary. "remote work security" won't find a chunk that says
# "off-site device protection" even if it's highly relevant.
#
# Solution: use the LLM to generate 3 alternative phrasings of your query,
# run all of them, then merge and deduplicate the results.
#
# This is implemented here using core LangChain primitives (PromptTemplate + LLM)
# so you can see exactly what happens under the hood — no opaque wrapper needed.
#
# Cost: 1 extra LLM call per retrieval (to generate the query variants).
# Benefit: higher recall — more relevant chunks are found.
#
# When to use: when your query is ambiguous or could be phrased many ways.
print("\n=== 3. MultiQuery (query expansion) ===")
from langchain_core.prompts import PromptTemplate  # noqa: E402
from langchain_core.output_parsers import StrOutputParser  # noqa: E402

# Step 1: ask the LLM to rephrase the query 3 ways
variant_prompt = PromptTemplate.from_template(
    "Generate exactly 3 alternative phrasings of the following question to help "
    "retrieve relevant documents. Output only the questions, one per line, no numbering.\n\n"
    "Question: {question}"
)
variant_chain = variant_prompt | llm | StrOutputParser()

variants_raw = variant_chain.invoke({"question": QUERY})
variants = [v.strip() for v in variants_raw.strip().split("\n") if v.strip()]
print(f"Generated query variants:")
for v in variants:
    print(f"  - {v}")

# Step 2: run original + all variants, deduplicate results by content
seen_keys: set = set()
all_docs = []
for q in [QUERY] + variants:
    for doc in basic_retriever.invoke(q):
        key = doc.page_content[:80]
        if key not in seen_keys:
            seen_keys.add(key)
            all_docs.append(doc)

print(f"\nOriginal query: '{QUERY}'")
print(f"Retrieved {len(all_docs)} unique docs after merging all query variants:")
for doc in all_docs[:3]:
    print(f"  page={doc.metadata.get('page')} | {doc.page_content[:120]}...")
# Notice the result count is usually higher than basic_retriever — more coverage.
# The deduplication step ensures each chunk appears at most once even if multiple
# query variants retrieve the same chunk.


# ── 4. Contextual Compression (extract relevant parts) ────────────────────────
# Problem: retrieved chunks often contain a mix of relevant and irrelevant sentences.
# A 500-character chunk about password policy might start with 3 relevant sentences
# and end with 2 sentences about something completely different.
#
# Solution: after retrieval, pass each chunk through an LLM that extracts ONLY
# the sentences relevant to the query — discarding the rest.
#
# Pipeline:
#   Query → basic_retriever → raw chunks → LLM extractor → compressed chunks
#
# This is implemented here with a direct LCEL chain so you can see the mechanics.
#
# Cost: 1 extra LLM call per retrieved chunk (to compress it).
# Benefit: the LLM receives a tighter, more focused context — fewer tokens wasted.
#
# When to use: when chunk quality matters more than retrieval speed.
print("\n=== 4. Contextual Compression (extract relevant parts) ===")
from langchain_core.documents import Document  # noqa: E402

extract_prompt = PromptTemplate.from_template(
    "Given the document below and a question, extract ONLY the sentences that "
    "directly help answer the question. If nothing is relevant, output exactly: IRRELEVANT\n\n"
    "Question: {question}\n\nDocument:\n{document}\n\nRelevant extract:"
)
extract_chain = extract_prompt | llm | StrOutputParser()

raw_docs = basic_retriever.invoke(QUERY)
compressed_docs = []
for doc in raw_docs:
    extract = extract_chain.invoke({"question": QUERY, "document": doc.page_content})
    if extract.strip().upper() != "IRRELEVANT":
        compressed_docs.append(Document(page_content=extract.strip(), metadata=doc.metadata))

print(f"Retrieved {len(compressed_docs)} compressed docs:")
for doc in compressed_docs:
    print(f"  page={doc.metadata.get('page')} | {doc.page_content[:200]}...")
# Each result is shorter than the original 500-char chunk —
# only the sentences that directly answer the query are kept.


# ── 5. Metadata Filtering (self-query concept) ────────────────────────────────
# Problem: basic search ignores metadata. You can't say "only search page 2"
# or "only look at sections from the security chapter".
#
# Concept: parse the natural language query into TWO parts:
#   a) Semantic query — the part to search by embedding similarity
#   b) Metadata filter — structured conditions applied BEFORE the vector search
#
# Example: "What does page 2 say about software installation?"
#   → semantic query:  "software installation"
#   → metadata filter: page == 2
#
# Implemented here with a simple LLM-based parser + manual filter applied to
# Chroma (which supports metadata filtering, unlike FAISS).
print("\n=== 5. Metadata Filtering (self-query concept) ===")
import json  # noqa: E402
import re    # noqa: E402
from langchain_community.vectorstores import Chroma  # noqa: E402

chroma_store = Chroma.from_documents(
    chunks, embeddings, collection_name="self_query_demo"
)

# Ask the LLM to extract the page filter and semantic query from natural language.
parse_prompt = PromptTemplate.from_template(
    "Extract the page number filter and search query from this question.\n"
    "Return valid JSON only, with keys 'page' (integer or null) and 'query' (string).\n\n"
    "Question: {question}\n\nJSON:"
)
parse_chain = parse_prompt | llm | StrOutputParser()

nl_query = "What does page 2 say about software installation?"
parsed_raw = parse_chain.invoke({"question": nl_query})

# Strip markdown code fences if the LLM wrapped the JSON
parsed_raw = re.sub(r"```(?:json)?|```", "", parsed_raw).strip()
parsed = json.loads(parsed_raw)
semantic_query = parsed.get("query", nl_query)
page_filter = parsed.get("page")

print(f"Natural language: '{nl_query}'")
print(f"  → semantic query: '{semantic_query}'")
print(f"  → page filter:    {page_filter}")

# Apply the metadata filter in Chroma then run similarity search
if page_filter is not None:
    results = chroma_store.similarity_search(
        semantic_query, k=3, filter={"page": page_filter}
    )
else:
    results = chroma_store.similarity_search(semantic_query, k=3)

print(f"\nRetrieved {len(results)} docs via metadata-filtered search:")
for doc in results[:2]:
    print(f"  page={doc.metadata.get('page')} | {doc.page_content[:120]}...")
# All results should be from page 2 — the filter restricts the search space
# BEFORE the vector similarity step runs.


# ── Summary: when to use each retriever ────────────────────────────────────────
print("\n--- Retriever decision guide ---")
print("Basic          → default; fast; use when query phrasing is reliable")
print("MMR            → use when basic returns redundant/repetitive chunks")
print("MultiQuery     → use when the query could be phrased many different ways")
print("Compression    → use when you need tight, precise context for the LLM")
print("Metadata filter→ use when you need to filter by metadata (page, date, category)")
print("\n✅ Done. Next step (file 12): wire a retriever into a full RAG chain.")
