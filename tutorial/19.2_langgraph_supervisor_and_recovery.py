"""
19.2 - LangGraph Supervisor Agent + Error Recovery
===================================================
Two production patterns centred on conditional routing:

─────────────────────────────────────────────────────────────────────────────
Pattern A: Supervisor Multi-Agent
─────────────────────────────────────────────────────────────────────────────
Concept: A "supervisor" LLM reads each incoming request and routes it to
the most appropriate specialist agent. Specialists handle narrow domains and
produce the final answer. The supervisor never answers directly — its only
job is deciding WHO should answer.

Graph structure:
  START
    │
  supervisor      ← LLM reads request, picks a specialist, writes next_agent
    │
  route()  ───────┬──────────┬──────────┐
                  │          │          │
               writer     analyst    coder      ← specialist nodes
                  │          │          │
                  └──────────┴──────────┘
                             │
                            END

Key design — supervisor writes a routing key to state:
  supervisor writes state["next_agent"] = "writer" (or "analyst" or "coder").
  The conditional edge is simply: lambda s: s["next_agent"]
  This means the router is stateless — it just reads what the supervisor wrote.
  Adding a new specialist = add its name to AGENTS, register add_node, add_edge to END.

Why specialist agents beat a single LLM:
  - A narrow persona prompt ("You are an expert data analyst...") produces
    better, more focused output than a generic assistant for that domain.
  - Each specialist can use a different model, temperature, or tool set.
  - You can update, test, or replace one specialist without touching others.

State design:
  messages    → accumulates conversation history (add_messages reducer)
  next_agent  → the supervisor's routing decision (node name to jump to)
  final_answer → written by whichever specialist runs

─────────────────────────────────────────────────────────────────────────────
Pattern B: Error Recovery with Fallback
─────────────────────────────────────────────────────────────────────────────
Concept: A node catches its own exceptions, writes an "error" key to state
instead of crashing, and a conditional edge routes to a dedicated recovery
node. Failures degrade gracefully — the caller always receives a valid result.

Graph structure:
  START
    │
  safe_node   ← attempts main logic; on exception writes state["error"]
    │
  route() ──── error?──────────────────────────────────────────────────────┐
    │                                                                       │
   END  ← success path                                           error_handler
                                                                           │
                                                                          END

Key design — safe_node never raises:
  On success: returns {"result": "...", "error": ""}
  On failure: returns {"error": str(e), "result": ""}   ← NEVER re-raises
  The conditional edge checks state["error"]: non-empty → error_handler.

This is the LangGraph equivalent of a try/except around an entire pipeline
step, with the "except" branch as a named, observable graph node.

Why not just use try/except in the caller?
  - The failure is recorded in state — visible in checkpointer traces
  - error_handler is a proper node: it can call an LLM, send alerts, retry
  - The graph structure explicitly documents that failure is a valid path
  - Other nodes can inspect state["error"] downstream if needed

When to use:
  - LLM calls subject to rate limits, content filtering, or timeouts
  - External API calls that should be retried or gracefully skipped
  - Any node where partial failure should degrade gracefully, not crash
"""

from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = StrOutputParser()


# ══════════════════════════════════════════════════════════════════════════════
# Pattern A: Supervisor Multi-Agent
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("19.2 (A) - Supervisor routes requests to specialist agents")
print("=" * 60)


class SupervisorState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    # add_messages reducer: returned messages append to the list, not replace it
    next_agent: str    # routing decision written by supervisor, read by conditional edge
    final_answer: str  # written by whichever specialist node runs


# All valid specialist names. Each becomes a graph node.
# To add a new specialist: add its name here, call add_node, add add_edge to END.
AGENTS = ["writer", "analyst", "coder"]


def supervisor(state: SupervisorState) -> dict:
    """Node: read the latest human message and pick the best specialist.

    The supervisor's only output is next_agent — it never produces the final answer.
    Writing next_agent to state keeps the routing decision inspectable and testable.
    """
    last_human = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        ""
    )
    decision = (
        ChatPromptTemplate.from_messages([
            ("system", f"You are a router. Choose the best specialist from {AGENTS}. Reply with the name only."),
            ("human", "{request}"),
        ])
        | llm | parser
    ).invoke({"request": last_human})

    # Guard against hallucinated names: fall back to "writer" if no match
    chosen = next((a for a in AGENTS if a in decision.lower()), "writer")
    print(f"  [supervisor] '{last_human[:70]}' → '{chosen}'")
    return {"next_agent": chosen}


def make_agent(persona: str):
    """Factory: returns a node function acting as a specialist with the given persona.

    Using a factory avoids repeating nearly-identical node definitions.
    Each specialist receives only HumanMessages — it doesn't need the
    supervisor's routing messages in its context window.
    """
    def agent_fn(state: SupervisorState) -> dict:
        print(f"  [{persona}] generating specialist response...")
        human_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
        answer = llm.invoke([
            SystemMessage(content=f"You are an expert {persona}. Give a focused, professional answer."),
            *human_messages,
        ]).content
        return {
            "final_answer": answer,
            "messages": [AIMessage(content=answer)],  # appended via add_messages
        }
    return agent_fn


# Build the supervisor graph
sg = StateGraph(SupervisorState)
sg.add_node("supervisor", supervisor)

for agent_name in AGENTS:
    sg.add_node(agent_name, make_agent(agent_name))

sg.add_edge(START, "supervisor")

# Conditional edge: next_agent was set by supervisor, so just read it.
# This works because supervisor always writes a valid node name.
sg.add_conditional_edges("supervisor", lambda s: s["next_agent"])

for agent_name in AGENTS:
    sg.add_edge(agent_name, END)

supervisor_app = sg.compile()

# ── Run supervisor example ────────────────────────────────────────────────────
test_requests = [
    "Write a short poem about AI.",                        # → writer
    "Analyse Python vs JavaScript for backends.",          # → analyst
    "Write a Python function to reverse a string.",        # → coder
]

for req in test_requests:
    print(f"\nRequest: {req}")
    result = supervisor_app.invoke({"messages": [HumanMessage(content=req)]})
    print(f"Routed to: {result['next_agent']}")
    print(f"Answer:\n{result['final_answer'][:300]}...")


# ══════════════════════════════════════════════════════════════════════════════
# Pattern B: Error Recovery with Fallback
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("19.2 (B) - Error recovery — graceful fallback on failure")
print("=" * 60)


class RobustState(TypedDict):
    input: str       # the text to process
    result: str      # populated by safe_node on success OR error_handler on failure
    error: str       # empty string on success; error message on failure
    recovered: bool  # True if error_handler ran (useful for caller diagnostics)


def safe_node(state: RobustState) -> dict:
    """Node: attempt the main logic. On any exception, write to error key — never raise.

    Contract:
      Success → {"result": "<llm output>", "error": ""}
      Failure → {"error": "<exception message>", "result": ""}

    Never raising means the graph stays alive and routes to error_handler.
    The caller always receives a valid state dict instead of an exception.
    """
    print(f"  [safe_node] processing: '{state['input']}'")
    try:
        if state["input"].lower() == "fail":
            raise ValueError("Simulated failure — input='fail' triggers this branch")
        result = llm.invoke(f"Summarise in one sentence: {state['input']}").content
        return {"result": result, "error": ""}
    except Exception as e:
        print(f"  [safe_node] caught exception: {e}")
        return {"error": str(e), "result": ""}  # route to error_handler via conditional edge


def error_handler(state: RobustState) -> dict:
    """Node: produce a user-friendly fallback when safe_node failed.

    Only runs when state["error"] is non-empty. In a real system this node
    could: call a backup LLM, log to an alerting service, or attempt a retry
    with different parameters. Here it returns a polite fallback message.
    """
    print(f"  [error_handler] safe_node failed: '{state['error']}'")
    print("  [error_handler] generating fallback response")
    return {
        "result": f"Could not process '{state['input']}'. Please rephrase your input.",
        "recovered": True,
    }


# Build the error-recovery graph
rg = StateGraph(RobustState)
rg.add_node("safe_node", safe_node)
rg.add_node("error_handler", error_handler)

rg.add_edge(START, "safe_node")

# Conditional edge: inspect state["error"] after safe_node.
# Non-empty string → failure → divert to error_handler.
# Empty string     → success → go to END directly.
rg.add_conditional_edges(
    "safe_node",
    lambda s: "error_handler" if s.get("error") else END,
)

rg.add_edge("error_handler", END)
robust_app = rg.compile()

# ── Run error-recovery example ────────────────────────────────────────────────
test_inputs = [
    "the history of artificial intelligence",  # success path
    "fail",                                    # failure path → error_handler
]

for inp in test_inputs:
    print(f"\nInput: '{inp}'")
    result = robust_app.invoke({"input": inp, "error": "", "recovered": False})
    print(f"Result: {result['result']}")
    if result.get("recovered"):
        print("  (note: result came from error recovery fallback)")


# ── Pattern recap ─────────────────────────────────────────────────────────────
print("\n--- Pattern recap ---")
print()
print("Supervisor Multi-Agent:")
print("  supervisor writes next_agent to state (the routing decision)")
print("  conditional edge: lambda s: s['next_agent'] → jumps to that node")
print("  specialist receives only HumanMessages — narrow prompt = better output")
print("  add a specialist: add name to AGENTS, add_node, add_edge(name, END)")
print()
print("Error Recovery:")
print("  safe_node catches exceptions, writes state['error'] — never raises")
print("  conditional edge: non-empty error → error_handler, empty → END")
print("  callers always receive a valid result dict — failures degrade gracefully")
print("  error_handler is a full node: can log, alert, retry, or call a backup LLM")
