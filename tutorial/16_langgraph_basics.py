"""
16 - LangGraph Basics
=====================
Concept: LangGraph builds stateful, multi-step workflows as a directed graph.
Each node is a Python function that reads from and writes to a shared state dict.
Edges define which node runs next.

Why LangGraph instead of a plain LCEL chain?
  LCEL chains are linear pipelines: A | B | C. They work well for simple flows
  but cannot loop back, branch conditionally, or pause for human input.

  LangGraph adds:
    Cycles / loops    → a node can route back to an earlier node (e.g. retry logic)
    Conditional edges → the next node is chosen at runtime based on state values
    Shared state      → every node reads from and writes to the same dict
    Human-in-the-loop → a node can pause and wait for external input
    Visibility        → the graph structure is inspectable and printable

Core building blocks:
  TypedDict     → defines the shape of the shared state (what keys exist)
  StateGraph    → the graph builder you add nodes and edges to
  Node          → a plain Python function: (state) → dict of updates
  Edge          → a fixed connection: always go from node A to node B
  compile()     → freezes the graph into a runnable object
  START / END   → special sentinel nodes marking entry and exit

How state works:
  - You define a TypedDict with all the keys the graph will use
  - graph.invoke(initial_state) puts the initial dict into the state
  - Each node RETURNS a dict of keys it wants to update
  - LangGraph merges those updates into the running state automatically
  - Nodes that don't update a key leave it unchanged
  - By the time graph.invoke() returns, the state contains every key
    that was written by any node

Execution model:
  invoke(input) → runs nodes in order → returns final state dict
  The graph runs synchronously, one node at a time, in topological order.
"""

from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = StrOutputParser()


# ── Example 1: Minimal linear graph ────────────────────────────────────────────
# The simplest possible LangGraph: two nodes connected in a straight line.
#
# Graph structure:
#   START → write_outline → write_article → END
#
# State schema (SimpleState):
#   topic   → input: the subject to write about (set before invoke)
#   outline → written by write_outline, read by write_article
#   article → written by write_article, returned as final output
#
# Key pattern: each node reads what it needs from state, calls the LLM,
# and returns ONLY the key(s) it produced. It does not need to return
# the full state — LangGraph merges partial updates automatically.
print("=" * 55)
print("Example 1: Linear graph (START → outline → article → END)")
print("=" * 55)

class SimpleState(TypedDict):
    topic: str    # provided at invoke time
    outline: str  # produced by write_outline
    article: str  # produced by write_article


def write_outline(state: SimpleState) -> dict:
    """Node 1: turn a topic into a bullet-point outline."""
    print(f"  [write_outline] topic='{state['topic']}'")
    # state["topic"] was set by the caller via graph.invoke({"topic": "..."})
    outline = (
        ChatPromptTemplate.from_messages([
            ("human", "Write a 3-bullet outline about: {topic}")
        ])
        | llm
        | parser
    ).invoke({"topic": state["topic"]})
    # Returning only {"outline": ...} — LangGraph merges this into state.
    # state["topic"] remains unchanged; state["article"] stays unset for now.
    return {"outline": outline}


def write_article(state: SimpleState) -> dict:
    """Node 2: expand the outline into a short article."""
    print(f"  [write_article] outline received ({len(state['outline'])} chars)")
    # state["outline"] was written by write_outline in the previous step.
    # LangGraph guarantees this node only runs after write_outline finishes.
    article = (
        ChatPromptTemplate.from_messages([
            ("human", "Write a 2-paragraph article from this outline:\n{outline}")
        ])
        | llm
        | parser
    ).invoke({"outline": state["outline"]})
    return {"article": article}


# Build the graph ───────────────────────────────────────────────────────────────────
# Step 1: create a builder with the state schema
builder = StateGraph(SimpleState)

# Step 2: register nodes — each node has a name and a function
builder.add_node("write_outline", write_outline)
builder.add_node("write_article", write_article)

# Step 3: connect nodes with edges
# START is the entry point — the first node to run
builder.add_edge(START, "write_outline")
builder.add_edge("write_outline", "write_article")
# END signals the graph to stop and return the current state
builder.add_edge("write_article", END)

# Step 4: compile — validates the graph and returns a runnable object
# After compile(), the structure is frozen. You cannot add more nodes/edges.
graph = builder.compile()

# Run the graph — initial state only needs the keys used by the first node.
# LangGraph fills in the rest as nodes execute.
result = graph.invoke({"topic": "the benefits of drinking water"})

# result is the final state dict — all keys are populated
print("\nOutline:\n", result["outline"])
print("\nArticle:\n", result["article"])


# ── Example 2: Three-node pipeline ─────────────────────────────────────────────
# A longer linear pipeline showing how state accumulates across nodes.
#
# Graph structure:
#   START → research → draft → improve → END
#
# State schema (PipelineState):
#   topic    → input
#   summary  → produced by research
#   draft    → produced by draft (reads summary)
#   improved → produced by improve (reads draft)
#
# This pattern — research → draft → refine — is a common LangGraph idiom
# for content generation pipelines. Each stage builds on the previous one.
print("\n" + "=" * 55)
print("Example 2: Three-node pipeline (research → draft → improve)")
print("=" * 55)

class PipelineState(TypedDict):
    topic: str     # provided at invoke time
    summary: str   # key facts about the topic
    draft: str     # raw social media post
    improved: str  # polished post with hashtags


def research(state: PipelineState) -> dict:
    """Node 1: gather key facts about the topic."""
    print("  [research] fetching facts...")
    summary = (
        ChatPromptTemplate.from_messages([
            ("human", "List the 5 most important facts about: {topic}")
        ])
        | llm | parser
    ).invoke({"topic": state["topic"]})
    return {"summary": summary}
    # Only "summary" is updated — "draft" and "improved" stay as empty strings
    # until their respective nodes run.


def draft(state: PipelineState) -> dict:
    """Node 2: turn the research summary into a first-draft social post."""
    print("  [draft] writing post...")
    post = (
        ChatPromptTemplate.from_messages([
            ("human", "Write a concise social media post from these facts:\n{summary}")
        ])
        | llm | parser
    ).invoke({"summary": state["summary"]})
    return {"draft": post}


def improve(state: PipelineState) -> dict:
    """Node 3: polish the draft and add hashtags."""
    print("  [improve] polishing...")
    better = (
        ChatPromptTemplate.from_messages([
            ("human", "Make this post more engaging and add 2 relevant hashtags:\n{draft}")
        ])
        | llm | parser
    ).invoke({"draft": state["draft"]})
    return {"improved": better}


# Build and run in one shot — same pattern as Example 1, just more nodes
pipeline = StateGraph(PipelineState)
for name, fn in [("research", research), ("draft", draft), ("improve", improve)]:
    pipeline.add_node(name, fn)
pipeline.add_edge(START, "research")
pipeline.add_edge("research", "draft")
pipeline.add_edge("draft", "improve")
pipeline.add_edge("improve", END)

result = pipeline.compile().invoke({"topic": "renewable energy"})
print("\nFinal post:\n", result["improved"])
# Notice: result also contains "summary" and "draft" — the full state is always
# returned, not just the last node's output. You can inspect intermediate steps.
print("\nIntermediate summary (from research node):\n", result["summary"][:200], "...")


# ── Example 3: Inspect the graph structure ───────────────────────────────────────
# After compile(), you can introspect the graph without any extra packages.
# get_graph().nodes lists every registered node by name.
# get_graph().edges lists every directed connection.
print("\n" + "=" * 55)
print("Example 3: Graph introspection")
print("=" * 55)

print("\nNodes:", list(graph.get_graph().nodes.keys()))
print("Edges:", [(e.source, e.target) for e in graph.get_graph().edges])
# Edges output: [('__start__', 'write_outline'), ('write_outline', 'write_article'),
#                ('write_article', '__end__')]
# Tip: install grandalf (`pip install grandalf`) to also call
# graph.get_graph().print_ascii() for a text diagram of the graph.


# ── Summary ───────────────────────────────────────────────────────────────────
print("\n--- LangGraph pattern recap ---")
print("1. Define state   → TypedDict with all keys the graph needs")
print("2. Write nodes    → functions that return {key: value} updates")
print("3. Add nodes      → builder.add_node(name, function)")
print("4. Add edges      → builder.add_edge(from_node, to_node)")
print("5. Compile        → graph = builder.compile()")
print("6. Run            → result = graph.invoke(initial_state_dict)")
print("\nNext step (file 17): conditional edges — route to different nodes based on state.")
