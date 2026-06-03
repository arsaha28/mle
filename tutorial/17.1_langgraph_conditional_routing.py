"""
17.1 - LangGraph Conditional Routing
=====================================
Concept: In a linear graph every node always leads to the same next node.
Conditional routing lets the graph choose WHICH node runs next based on the
current state — like an if/else branch in your workflow.

How it works:
  1. A node runs and writes something to state (e.g. a classification result)
  2. A router function reads that state value and returns the NAME of the next node
  3. add_conditional_edges() wires the router so LangGraph calls it after the node

Signature of a router function:
  def route(state: MyState) -> str:
      return "node_a"  or  "node_b"  (must match a registered node name or END)

The router does NOT update state — it only inspects it and returns a string.
It is never added as a node; it is passed directly to add_conditional_edges().

Graph structure for this example:
  START
    │
  classify          ← runs first, writes state["category"]
    │
  route()           ← inspects state["category"], returns next node name
   ├── "technical"  → answer_technical → END
   └── "general"    → answer_general   → END

Why use this pattern?
  - Different questions need different prompts / personas / models
  - You want one entry point but multiple specialised handlers
  - The routing decision itself is dynamic (made by an LLM or rule at runtime)

When to use:
  Any time you need "if X, go here; else go there" in your workflow.
"""

from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = StrOutputParser()


# ── State ─────────────────────────────────────────────────────────────────────────
class RouterState(TypedDict):
    question: str   # input: the question to answer
    category: str   # written by classify: "technical" or "general"
    answer: str     # written by whichever answer node runs


# ── Nodes ────────────────────────────────────────────────────────────────────────
def classify(state: RouterState) -> dict:
    """Node: ask the LLM to label the question as technical or general.

    This is the only node that runs before the branch. Its job is to write
    state["category"] so the router function can read it.
    """
    result = (
        ChatPromptTemplate.from_messages([
            ("system", "Reply with one word only: 'technical' or 'general'."),
            ("human", "{question}"),
        ])
        | llm | parser
    ).invoke({"question": state["question"]})

    # Normalise — LLM might say "Technical." so lower + keyword check
    category = "technical" if "technical" in result.lower() else "general"
    print(f"  [classify] '{state['question'][:50]}' → '{category}'")
    return {"category": category}


def answer_technical(state: RouterState) -> dict:
    """Node: answer with a software-engineer persona (detailed, precise)."""
    print("  [answer_technical]")
    answer = (
        ChatPromptTemplate.from_messages([
            ("system", "You are an experienced software engineer. Give a detailed, accurate answer."),
            ("human", "{question}"),
        ])
        | llm | parser
    ).invoke({"question": state["question"]})
    return {"answer": answer}


def answer_general(state: RouterState) -> dict:
    """Node: answer with a friendly assistant persona (simple, approachable)."""
    print("  [answer_general]")
    answer = (
        ChatPromptTemplate.from_messages([
            ("system", "You are a friendly assistant. Give a simple, easy-to-understand answer."),
            ("human", "{question}"),
        ])
        | llm | parser
    ).invoke({"question": state["question"]})
    return {"answer": answer}


# ── Router function ────────────────────────────────────────────────────────────────
# This is NOT a node — it is a plain function passed to add_conditional_edges().
# LangGraph calls it after "classify" runs, reads its return value, and jumps
# to the node with that name.
#
# The Literal type hint tells LangGraph (and type checkers) exactly which
# node names this function can return — useful for validation and IDE support.
def route(state: RouterState) -> Literal["answer_technical", "answer_general"]:
    """Router: choose the answer node based on the category in state."""
    if state["category"] == "technical":
        return "answer_technical"
    return "answer_general"


# ── Build the graph ────────────────────────────────────────────────────────────────
ROUTER_GRAPH = StateGraph(RouterState)

ROUTER_GRAPH.add_node("classify", classify)
ROUTER_GRAPH.add_node("answer_technical", answer_technical)
ROUTER_GRAPH.add_node("answer_general", answer_general)

ROUTER_GRAPH.add_edge(START, "classify")
ROUTER_GRAPH.add_edge("answer_technical", END)
ROUTER_GRAPH.add_edge("answer_general", END)

# Conditional edge — after "classify", call route() to decide what runs next
ROUTER_GRAPH.add_conditional_edges("classify", route)

app = ROUTER_GRAPH.compile()


# ── Run ───────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("17.1 - Conditional Routing")
print("=" * 60)

questions = [
    "What is a REST API?",           # → technical
    "What should I have for lunch?",  # → general
]

for q in questions:
    print(f"\nQ: {q}")
    result = app.invoke({"question": q})
    print(f"Category: {result['category']}")
    print(f"A: {result['answer'][:300]}")

print("\n--- Pattern recap ---")
print("add_conditional_edges(node, router_fn) → router_fn reads state, returns next node name")
print("Router fn is NOT a node — it runs between nodes to decide the path")
print("Use Literal return type to document which nodes are reachable")
