"""
16 - LangGraph Basics
=====================
Concept: LangGraph builds stateful multi-step workflows as a graph.
Nodes are processing steps; edges define flow between them.

Why LangGraph over plain chains?
  - Cycles / loops
  - Conditional branching based on state
  - Human-in-the-loop pauses
  - Explicit shared state

Core concepts:
  - StateGraph  -> the graph builder
  - TypedDict   -> shared state schema
  - Node        -> Python function that reads/updates state
  - Edge        -> connection between nodes
  - START / END -> entry and exit points
  - compile()   -> turns builder into a runnable graph
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

# Example 1: Minimal linear graph
print("=" * 55 + "\nExample 1: Linear graph (A -> B -> END)\n" + "=" * 55)

class SimpleState(TypedDict):
    topic: str
    outline: str
    article: str

def write_outline(state: SimpleState) -> dict:
    print(f"  [write_outline] topic='{state['topic']}'")
    outline = (ChatPromptTemplate.from_messages([("human", "Write a 3-bullet outline about: {topic}")]) | llm | parser).invoke({"topic": state["topic"]})
    return {"outline": outline}

def write_article(state: SimpleState) -> dict:
    print(f"  [write_article] outline received")
    article = (ChatPromptTemplate.from_messages([("human", "Write a 2-paragraph article from this outline:\n{outline}")]) | llm | parser).invoke({"outline": state["outline"]})
    return {"article": article}

builder = StateGraph(SimpleState)
builder.add_node("write_outline", write_outline)
builder.add_node("write_article", write_article)
builder.add_edge(START, "write_outline")
builder.add_edge("write_outline", "write_article")
builder.add_edge("write_article", END)
graph = builder.compile()

result = graph.invoke({"topic": "the benefits of drinking water"})
print("\nOutline:\n", result["outline"])
print("\nArticle:\n", result["article"])

# Example 2: Three-node pipeline
print("\n" + "=" * 55 + "\nExample 2: Three-node pipeline\n" + "=" * 55)

class PipelineState(TypedDict):
    topic: str
    summary: str
    draft: str
    improved: str

def research(state):
    print("  [research]")
    return {"summary": (ChatPromptTemplate.from_messages([("human", "Key facts about: {topic}")]) | llm | parser).invoke({"topic": state["topic"]})}

def draft(state):
    print("  [draft]")
    return {"draft": (ChatPromptTemplate.from_messages([("human", "Write a social media post from:\n{summary}")]) | llm | parser).invoke({"summary": state["summary"]})}

def improve(state):
    print("  [improve]")
    return {"improved": (ChatPromptTemplate.from_messages([("human", "Make this post more engaging and add 2 hashtags:\n{draft}")]) | llm | parser).invoke({"draft": state["draft"]})}

pipeline = StateGraph(PipelineState)
for name, fn in [("research", research), ("draft", draft), ("improve", improve)]:
    pipeline.add_node(name, fn)
pipeline.add_edge(START, "research")
pipeline.add_edge("research", "draft")
pipeline.add_edge("draft", "improve")
pipeline.add_edge("improve", END)
result = pipeline.compile().invoke({"topic": "renewable energy"})
print("\nFinal post:\n", result["improved"])

# Example 3: Inspect the graph
print("\n" + "=" * 55 + "\nExample 3: Graph structure\n" + "=" * 55)
graph.get_graph().print_ascii()
print("\nNodes:", list(graph.get_graph().nodes.keys()))
