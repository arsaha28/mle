"""
17.2 - LangGraph Parallel Branches (Fan-out / Fan-in)
======================================================
Concept: Multiple nodes can run simultaneously by connecting them both from
the same source node. This is called "fan-out". When they all finish, a single
downstream node reads all their outputs — this is "fan-in".

Graph structure:
             START
               │
  ┌────────────┴────────────┐   ← fan-out: both get_pros and get_cons
  │                         │     receive state at the same time
get_pros               get_cons
  │                         │
  └────────────┬────────────┘   ← fan-in: merge waits for BOTH to finish
             merge
               │
              END

How LangGraph handles fan-out / fan-in:
  - add_edge(START, "get_pros") and add_edge(START, "get_cons") both fire
    when the graph enters, sending the same state to both nodes
  - Both nodes write different keys: get_pros → {"pros": ...},
    get_cons → {"cons": ...}
  - LangGraph merges both updates into the shared state
  - "merge" only runs once both upstream nodes have finished
  - When merge runs, state already contains both "pros" and "cons"

Why use this pattern?
  - Parallelism: independent tasks run simultaneously instead of sequentially
  - In a sequential A → B → C pipeline, total time = time(A) + time(B) + time(C)
  - In a fan-out A → {B, C} → D pipeline, total time = time(A) + max(B,C) + time(D)
  - For LLM calls (which are network-bound), this can halve execution time

Important constraint:
  Fan-out branches must write to DIFFERENT state keys.
  If get_pros and get_cons both tried to write "answer", the second write
  would overwrite the first. Each branch owns its own key.

When to use:
  - Gathering multiple independent perspectives on the same input
  - Running several LLM calls that don't depend on each other
  - Any "gather then synthesise" workflow
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


# ── State ─────────────────────────────────────────────────────────────────────────
class ParallelState(TypedDict):
    topic: str      # input: the subject to analyse
    pros: str       # written by get_pros (fan-out branch 1)
    cons: str       # written by get_cons (fan-out branch 2)
    summary: str    # written by merge (fan-in node), reads pros + cons


# ── Nodes ────────────────────────────────────────────────────────────────────────
def get_pros(state: ParallelState) -> dict:
    """Fan-out branch 1: generate the pros of the topic."""
    print("  [get_pros] running...")
    pros = (
        ChatPromptTemplate.from_messages([
            ("human", "List exactly 3 advantages of {topic}. Be concise.")
        ])
        | llm | parser
    ).invoke({"topic": state["topic"]})
    return {"pros": pros}


def get_cons(state: ParallelState) -> dict:
    """Fan-out branch 2: generate the cons of the topic."""
    print("  [get_cons] running...")
    cons = (
        ChatPromptTemplate.from_messages([
            ("human", "List exactly 3 disadvantages of {topic}. Be concise.")
        ])
        | llm | parser
    ).invoke({"topic": state["topic"]})
    return {"cons": cons}


def merge(state: ParallelState) -> dict:
    """Fan-in node: synthesise both branches into a balanced summary."""
    print("  [merge] both branches complete, synthesising...")
    summary = (
        ChatPromptTemplate.from_messages([
            ("human",
             "Write a 2-sentence balanced summary of {topic}.\n\n"
             "Pros:\n{pros}\n\nCons:\n{cons}")
        ])
        | llm | parser
    ).invoke({"topic": state["topic"], "pros": state["pros"], "cons": state["cons"]})
    return {"summary": summary}


# ── Build the graph ────────────────────────────────────────────────────────────────
g = StateGraph(ParallelState)

g.add_node("get_pros", get_pros)
g.add_node("get_cons", get_cons)
g.add_node("merge", merge)

# Fan-out: START connects to BOTH branches
g.add_edge(START, "get_pros")
g.add_edge(START, "get_cons")

# Fan-in: BOTH branches connect to merge
g.add_edge("get_pros", "merge")
g.add_edge("get_cons", "merge")

g.add_edge("merge", END)

app = g.compile()


# ── Run ───────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("17.2 - Parallel Branches (fan-out / fan-in)")
print("=" * 60)

for topic in ["remote work", "electric vehicles"]:
    print(f"\nTopic: '{topic}'")
    result = app.invoke({"topic": topic})
    print(f"\nPros:\n{result['pros']}")
    print(f"\nCons:\n{result['cons']}")
    print(f"\nSummary:\n{result['summary']}")
    print("-" * 40)

print("\n--- Pattern recap ---")
print("add_edge(START, 'A') + add_edge(START, 'B') → A and B run in parallel")
print("add_edge('A', 'C') + add_edge('B', 'C')     → C waits for both A and B")
print("Each branch must write to a DIFFERENT state key to avoid overwriting")
