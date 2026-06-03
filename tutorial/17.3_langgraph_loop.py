"""
17.3 - LangGraph Loop (Retry Until Quality Threshold)
======================================================
Concept: LangGraph supports cycles — a node can route back to an earlier node,
creating a loop. This enables "generate → evaluate → improve" patterns where
the graph keeps refining output until a quality condition is met.

This is the key capability that separates LangGraph from plain LCEL chains.
An LCEL chain A | B | C can never loop back. A LangGraph can.

Graph structure:
  START
    │
  generate          ← writes a draft
    │
  score             ← evaluates the draft, writes a numeric score
    │
  should_continue() ← router: checks score and iteration count
   ├── score < 8 AND iterations < 3  → back to generate  (loop!)
   └── score >= 8 OR iterations >= 3 → END

How the loop works:
  1. generate writes state["draft"] and increments state["iterations"]
  2. score reads state["draft"] and writes state["score"]
  3. should_continue reads state["score"] and state["iterations"]
     - if quality is too low and we haven't hit the limit → go back to generate
     - if quality is good or we've tried enough → go to END
  4. On the next generate call, state["draft"] already exists so it improves
     the existing draft instead of writing a new one

Loop exit conditions — always have TWO:
  1. Quality condition  → exit when the result is good enough
  2. Iteration limit    → exit after N tries regardless of quality
  Without the iteration limit, a loop could run forever if the LLM
  never produces a high-scoring draft.

State evolution across iterations:
  iteration 1: draft="first attempt...", score=6  → loop back
  iteration 2: draft="improved...",      score=7  → loop back
  iteration 3: draft="more improved...", score=9  → exit (score >= 8)

When to use:
  - Self-improving content generation
  - Automated retry with validation
  - Agent loops where the agent keeps trying until it succeeds
  - Any "act → check → act again" workflow
"""

from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
parser = StrOutputParser()

MAX_ITERATIONS = 3
QUALITY_THRESHOLD = 8


# ── State ─────────────────────────────────────────────────────────────────────────
class IterativeState(TypedDict):
    topic: str        # input: what to write about
    draft: str        # current best draft (updated each iteration)
    score: int        # clarity score 1–10 from the evaluator LLM
    iterations: int   # how many times generate has run (loop counter)


# ── Nodes ────────────────────────────────────────────────────────────────────────
def generate(state: IterativeState) -> dict:
    """Node: write or improve a draft explanation."""
    n = state.get("iterations", 0) + 1
    print(f"  [generate] iteration {n}")
    if n == 1:
        prompt = f"Write a clear one-paragraph explanation of: {state['topic']}"
    else:
        prompt = (
            f"Improve this explanation to be clearer and more precise. "
            f"Keep it to one paragraph.\n\nCurrent draft:\n{state['draft']}"
        )
    draft = llm.invoke(prompt).content
    return {"draft": draft, "iterations": n}


def score_draft(state: IterativeState) -> dict:
    """Node: evaluate the current draft and assign a clarity score 1–10."""
    result = (
        ChatPromptTemplate.from_messages([
            ("system",
             "You are a writing quality evaluator. "
             "Score the following text for clarity and accuracy on a scale of 1 to 10. "
             "Reply with ONLY the integer score, nothing else."),
            ("human", "{draft}"),
        ])
        | llm | parser
    ).invoke({"draft": state["draft"]})
    s = int(result.strip()) if result.strip().isdigit() else 5
    print(f"  [score] score={s}/10")
    return {"score": s}


# ── Router function ────────────────────────────────────────────────────────────────
def should_continue(state: IterativeState) -> Literal["generate", "__end__"]:
    """Router: loop back to generate or exit."""
    score = state.get("score", 0)
    iterations = state.get("iterations", 0)
    if score >= QUALITY_THRESHOLD:
        print(f"  [router] score {score} >= {QUALITY_THRESHOLD} → EXIT (quality met)")
        return END
    if iterations >= MAX_ITERATIONS:
        print(f"  [router] {iterations} iterations reached limit → EXIT (cap hit)")
        return END
    print(f"  [router] score {score} < {QUALITY_THRESHOLD}, iteration {iterations} → LOOP")
    return "generate"


# ── Build the graph ────────────────────────────────────────────────────────────────
g = StateGraph(IterativeState)

g.add_node("generate", generate)
g.add_node("score", score_draft)

g.add_edge(START, "generate")
g.add_edge("generate", "score")

# Conditional edge that can loop back: "score" → should_continue() → "generate" or END
g.add_conditional_edges("score", should_continue)

app = g.compile()


# ── Run ───────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("17.3 - Loop (generate → score → improve until quality met)")
print("=" * 60)
print(f"\nSettings: threshold={QUALITY_THRESHOLD}/10, max_iterations={MAX_ITERATIONS}")
print("\nRunning loop for topic: 'quantum computing'\n")

result = app.invoke({"topic": "quantum computing"})

print(f"\n--- Final result ---")
print(f"Iterations:  {result['iterations']}")
print(f"Final score: {result['score']}/10")
print(f"\nFinal draft:\n{result['draft']}")

print("\n--- Pattern recap ---")
print("add_conditional_edges('score', should_continue) creates the cycle")
print("Router returns 'generate' to loop, or END to exit")
print("Always include an iteration cap to prevent infinite loops")
print("Use temperature > 0 on the generator so each retry is different")
