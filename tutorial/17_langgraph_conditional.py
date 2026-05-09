"""
17 - LangGraph Conditional Routing & Branching
===============================================
Concept: Route to different nodes based on state using conditional edges.
A router function inspects state and returns the next node name.

Patterns covered:
  1. Simple conditional edge  -> route based on classification
  2. Parallel branches        -> fan-out / fan-in
  3. Loop with exit condition -> retry until quality threshold met
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

# Pattern 1: Conditional routing
print("=" * 60 + "\nPattern 1: Route by question type\n" + "=" * 60)

class RouterState(TypedDict):
    question: str
    category: str
    answer: str

def classify(state: RouterState) -> dict:
    result = (ChatPromptTemplate.from_messages([("system", "Reply with one word: 'technical' or 'general'."), ("human", "{question}")]) | llm | parser).invoke({"question": state["question"]})
    category = "technical" if "technical" in result.lower() else "general"
    print(f"  [classify] -> '{category}'")
    return {"category": category}

def answer_technical(state: RouterState) -> dict:
    print("  [answer_technical]")
    return {"answer": (ChatPromptTemplate.from_messages([("system", "You are a software engineer. Be detailed."), ("human", "{question}")]) | llm | parser).invoke({"question": state["question"]})}

def answer_general(state: RouterState) -> dict:
    print("  [answer_general]")
    return {"answer": (ChatPromptTemplate.from_messages([("system", "You are a friendly assistant. Be simple."), ("human", "{question}")]) | llm | parser).invoke({"question": state["question"]})}

def route(state: RouterState) -> Literal["answer_technical", "answer_general"]:
    return "answer_technical" if state["category"] == "technical" else "answer_general"

g = StateGraph(RouterState)
g.add_node("classify", classify)
g.add_node("answer_technical", answer_technical)
g.add_node("answer_general", answer_general)
g.add_edge(START, "classify")
g.add_conditional_edges("classify", route)
g.add_edge("answer_technical", END)
g.add_edge("answer_general", END)
router_app = g.compile()

for q in ["What is a REST API?", "What should I have for lunch?"]:
    print(f"\nQ: {q}")
    print(f"A: {router_app.invoke({'question': q})['answer'][:200]}")

# Pattern 2: Parallel branches
print("\n" + "=" * 60 + "\nPattern 2: Parallel branches (pros + cons)\n" + "=" * 60)

class ParallelState(TypedDict):
    topic: str
    pros: str
    cons: str
    summary: str

def get_pros(state):
    print("  [get_pros]")
    return {"pros": (ChatPromptTemplate.from_messages([("human", "List 3 pros of {topic}.")]) | llm | parser).invoke({"topic": state["topic"]})}

def get_cons(state):
    print("  [get_cons]")
    return {"cons": (ChatPromptTemplate.from_messages([("human", "List 3 cons of {topic}.")]) | llm | parser).invoke({"topic": state["topic"]})}

def merge(state):
    print("  [merge]")
    return {"summary": (ChatPromptTemplate.from_messages([("human", "Summarise pros/cons of {topic} in 2 sentences.\nPros:\n{pros}\nCons:\n{cons}")]) | llm | parser).invoke(state)}

pg = StateGraph(ParallelState)
for name, fn in [("get_pros", get_pros), ("get_cons", get_cons), ("merge", merge)]:
    pg.add_node(name, fn)
pg.add_edge(START, "get_pros")
pg.add_edge(START, "get_cons")
pg.add_edge("get_pros", "merge")
pg.add_edge("get_cons", "merge")
pg.add_edge("merge", END)
result = pg.compile().invoke({"topic": "remote work"})
print("Summary:", result["summary"])

# Pattern 3: Retry loop
print("\n" + "=" * 60 + "\nPattern 3: Retry loop (improve until score >= 8)\n" + "=" * 60)

class IterativeState(TypedDict):
    topic: str
    draft: str
    score: int
    iterations: int

def generate(state):
    n = state.get("iterations", 0) + 1
    print(f"  [generate] iteration {n}")
    text = f"Write a one-paragraph explanation of {state['topic']}." if n == 1 else f"Improve this explanation:\n{state['draft']}"
    return {"draft": llm.invoke(text).content, "iterations": n}

def score(state):
    result = (ChatPromptTemplate.from_messages([("system", "Score text clarity 1-10. Reply with just the integer."), ("human", "{draft}")]) | llm | parser).invoke({"draft": state["draft"]})
    s = int(result.strip()) if result.strip().isdigit() else 5
    print(f"  [score] score={s}")
    return {"score": s}

def should_continue(state) -> Literal["generate", END]:
    return END if state["score"] >= 8 or state.get("iterations", 0) >= 3 else "generate"

lg = StateGraph(IterativeState)
lg.add_node("generate", generate)
lg.add_node("score", score)
lg.add_edge(START, "generate")
lg.add_edge("generate", "score")
lg.add_conditional_edges("score", should_continue)
result = lg.compile().invoke({"topic": "quantum computing"})
print(f"\nFinal draft (score={result['score']}, iterations={result['iterations']}):\n{result['draft']}")
