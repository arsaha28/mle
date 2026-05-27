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
  3. MultiQueryRetriever             → generates multiple query variants, merges results
  4. ContextualCompressionRetriever  → strips irrelevant parts from returned chunks
  5. SelfQueryRetriever              → translates natural language filters into metadata queries
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


# ── 3. MultiQueryRetriever ───────────────────────────────────────────────────────
# Problem: a single query phrasing might miss relevant chunks that use
# different vocabulary. "remote work security" won't find a chunk that says
# "off-site device protection" even if it's highly relevant.
#
# Solution: use the LLM to generate 3 alternative phrasings of your query,
# run all of them, then merge and deduplicate the results.
#
# Cost: 1 extra LLM call per retrieval (to generate the query variants).
# Benefit: higher recall — more relevant chunks are found.
#
# When to use: when your query is ambiguous or could be phrased many ways.
print("\n=== 3. MultiQueryRetriever (query expansion) ===")
try:
    from langchain.retrievers.multi_query import MultiQueryRetriever
except ImportError:
    from langchain_community.retrievers.multi_query import MultiQueryRetriever  # noqa: E402
import logging  # noqa: E402

# Enable logging to see the generated query variants in the output
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

multi_retriever = MultiQueryRetriever.from_llm(
    retriever=basic_retriever,  # underlying retriever to run each variant against
    llm=llm,                    # LLM used to generate the query variants
)
docs_retrieved = multi_retriever.invoke(QUERY)
print(f"\nOriginal query: '{QUERY}'")
print(f"Retrieved {len(docs_retrieved)} docs after merging all query variants:")
for doc in docs_retrieved[:3]:
    print(f"  page={doc.metadata.get('page')} | {doc.page_content[:120]}...")
# The log output shows the 3 generated query variants before the results.
# Notice the result count is usually higher than basic_retriever — more coverage.


# ── 4. ContextualCompressionRetriever ─────────────────────────────────────────────
# Problem: retrieved chunks often contain a mix of relevant and irrelevant sentences.
# A 500-character chunk about password policy might start with 3 relevant sentences
# and end with 2 sentences about something completely different.
#
# Solution: after retrieval, pass each chunk through an LLM compressor that
# extracts ONLY the sentences relevant to the query.
#
# Pipeline:
#   Query → basic_retriever → raw chunks → LLMChainExtractor → compressed chunks
#
# Cost: 1 extra LLM call per retrieved chunk (to compress it).
# Benefit: the LLM receives a tighter, more focused context — fewer tokens wasted.
#
# When to use: when chunk quality matters more than retrieval speed.
print("\n=== 4. ContextualCompressionRetriever (extract relevant parts) ===")
try:
    from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
    from langchain.retrievers.document_compressors import LLMChainExtractor
except ImportError:
    from langchain_community.retrievers.contextual_compression import ContextualCompressionRetriever  # noqa: E402
    from langchain_community.retrievers.document_compressors import LLMChainExtractor  # noqa: E402

# LLMChainExtractor reads each chunk and returns only the relevant sentences.
compressor = LLMChainExtractor.from_llm(llm)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,   # the LLM that does the extraction
    base_retriever=basic_retriever,  # the retriever that fetches raw chunks first
)
docs_retrieved = compression_retriever.invoke(QUERY)
print(f"Retrieved {len(docs_retrieved)} compressed docs:")
for doc in docs_retrieved:
    print(f"  page={doc.metadata.get('page')} | {doc.page_content[:200]}...")
# Each result should be shorter than the original 500-char chunk —
# only the sentences that directly answer the query are kept.


# ── 5. SelfQueryRetriever ────────────────────────────────────────────────────────
# Problem: basic search ignores metadata. You can't say "only search page 2"
# or "only look at sections from the security chapter".
#
# Solution: the LLM parses your natural language query into TWO things:
#   a) Semantic query — the part to search by embedding similarity
#   b) Metadata filter — structured conditions applied BEFORE the vector search
#
# Example: "What does page 2 say about software installation?"
#   → semantic query:  "software installation"
#   → metadata filter: page == 2
#
# The vector search runs only on chunks that pass the filter — much more precise.
#
# Requires: a vector store that supports metadata filtering (Chroma, Pinecone, etc.)
# FAISS does not support metadata filtering, so we use Chroma here.
print("\n=== 5. SelfQueryRetriever (natural language → metadata filter) ===")
from langchain.retrievers.self_query.base import SelfQueryRetriever  # noqa: E402
from langchain.chains.query_constructor.base import AttributeInfo  # noqa: E402
from langchain_community.vectorstores import Chroma  # noqa: E402

# Tell the LLM what metadata fields exist so it can build the right filter.
metadata_field_info = [
    AttributeInfo(
        name="page",
        description="The page number in the policy PDF (integer, starts at 0)",
        type="integer",
    ),
    AttributeInfo(
        name="source",
        description="The file path of the source PDF document",
        type="string",
    ),
]

# Build a Chroma store (supports metadata filtering, unlike FAISS)
chroma_store = Chroma.from_documents(
    chunks, embeddings, collection_name="self_query_demo"
)

self_query_retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=chroma_store,
    document_contents="Company laptop usage policy sections",
    metadata_field_info=metadata_field_info,
    verbose=True,  # prints the parsed query + filter so you can see what the LLM generated
)

results = self_query_retriever.invoke("What does page 2 say about software installation?")
print(f"\nRetrieved {len(results)} docs via self-query:")
for doc in results[:2]:
    print(f"  page={doc.metadata.get('page')} | {doc.page_content[:120]}...")
# Watch the verbose output — you'll see the LLM extract:
#   query="software installation"  filter={"page": {"$eq": 2}}


# ── Summary: when to use each retriever ────────────────────────────────────────
print("\n--- Retriever decision guide ---")
print("Basic          → default; fast; use when query phrasing is reliable")
print("MMR            → use when basic returns redundant/repetitive chunks")
print("MultiQuery     → use when the query could be phrased many different ways")
print("Compression    → use when you need tight, precise context for the LLM")
print("SelfQuery      → use when you need to filter by metadata (page, date, category)")
print("\n✅ Done. Next step (file 12): wire a retriever into a full RAG chain.")
