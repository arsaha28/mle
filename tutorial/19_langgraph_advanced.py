"""
19 - LangGraph Advanced Patterns
==================================
Three production-grade patterns that appear in real LangGraph deployments.

─────────────────────────────────────────────────────────────────────────────
Pattern 1: Human-in-the-Loop (HITL)
─────────────────────────────────────────────────────────────────────────────
Concept: Pause the graph mid-execution and wait for a human decision before
continuing. The graph is NOT just "ask the user a question in a node" — it
fully suspends, serialises its state to a persistent store (checkpointer),
and resumes later when the human provides input.

Graph structure:
  START
    │
  draft_email     ← LLM writes the draft
    │
  human_review    ← interrupt() suspends here, waits for human
    │
  send_email      ← proceeds or skips depending on approval
    │
   END

How interrupt() works:
  1. The graph runs draft_email normally.
  2. human_review calls interrupt("prompt string") — this immediately raises
     a GraphInterrupt exception inside LangGraph, which pauses execution and
     returns control to the caller. The caller's .invoke() call returns early.
  3. The current state is saved to the MemorySaver checkpointer under the
     thread_id. Think of thread_id as a session ID — it lets you resume
     a specific conversation later.
  4. The human reads the draft (printed before interrupt) and provides input.
  5. email_app.update_state(config, {"approved": True}, as_node="human_review")
     injects the human's decision into the saved state, as if human_review
     had returned {"approved": True}.
  6. email_app.invoke(None, config=config) resumes from the saved checkpoint.
     The None input means "don't add a new message — resume where we left off."
     Execution continues from the next node: send_email.

Why MemorySaver is required:
  Without a checkpointer, interrupt() has nowhere to save state. The graph
  would be garbage-collected between the two invoke() calls and could not
  resume. MemorySaver stores everything in RAM (fine for development/testing).
  In production, use langgraph.checkpoint.postgres.PostgresSaver or
  langgraph.checkpoint.redis.RedisSaver for durable persistence.

When to use:
  - Email / document approval workflows
  - Code review: generate code, pause for human to review, then deploy
  - Any workflow where an irreversible action (send, deploy, purchase) needs
    human sign-off before it runs

─────────────────────────────────────────────────────────────────────────────
Pattern 2: Supervisor Multi-Agent
─────────────────────────────────────────────────────────────────────────────
Concept: A "supervisor" LLM examines each incoming request and routes it to
the most appropriate specialist agent. Each specialist handles a narrow domain
and produces the final answer. The supervisor never answers directly — it
only decides WHO should answer.

Graph structure:
  START
    │
  supervisor      ← LLM reads the request, picks a specialist
    │
  route() ─────────┬─────────┬─────────┐
                   │         │         │
                writer    analyst    coder    ← specialist agents
                   │         │         │
                   └─────────┴────────-┘
                             │
                            END

Why use a supervisor instead of a single LLM:
  - Different tasks need different personas/prompts/models
  - A specialist prompt ("You are an expert data analyst...") produces better
    output than a generic one for that domain
  - The supervisor routing decision is cheap (one fast LLM call); the specialist
    calls are where the quality work happens
  - You can swap individual specialists independently without touching others
  - Scales: add a new specialist by registering a new node + updating the
    supervisor's options list

State design:
  - messages: accumulates conversation history (add_messages reducer)
  - next_agent: set by supervisor, read by the conditional edge router
  - final_answer: written by whichever specialist runs

The conditional edge after supervisor reads state["next_agent"] and returns
it as the node name to jump to. This is a clean pattern: the supervisor
writes its decision to state, and the router is just `lambda s: s["next_agent"]`.

When to use:
  - Multi-domain chatbots (technical support, billing, general Q&A)
  - Document processing pipelines (route to summariser, extractor, classifier)
  - Research agents (route to web-search, code-execution, knowledge-base nodes)

─────────────────────────────────────────────────────────────────────────────
Pattern 3: Error Recovery with Fallback
─────────────────────────────────────────────────────────────────────────────
Concept: A node catches its own exceptions, writes an "error" key to state
instead of crashing, and a conditional edge routes to a dedicated recovery
node. This makes the graph resilient: failures degrade gracefully rather than
propagating exceptions to the caller.

Graph structure:
  START
    │
  safe_node       ← tries the main logic; on exception writes state["error"]
    │
  route() ──── error? ──────────────────────────────────────────────────────┐
    │                                                                        │
   END (success path)                                               error_handler
                                                                            │
                                                                           END

Key design:
  - safe_node never re-raises. It returns {"error": str(e), "result": ""}
    on failure, or {"result": "...", "error": ""} on success.
  - The conditional edge checks state["error"]: non-empty → error_handler,
    empty → END (success).
  - error_handler writes a user-friendly message and sets recovered=True.
  - The caller always gets a valid result dict — it never sees an exception.

This is the LangGraph equivalent of a try/except around an entire pipeline
step, with the "except" branch as a named, observable graph node.

When to use:
  - LLM calls that might fail (rate limits, content filtering, timeouts)
  - External API calls that can be retried or gracefully skipped
  - Any node where a partial failure should degrade gracefully, not crash
"""

from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = StrOutputParser()


# ══════════════════════════════════════════════════════════════════════════════
# Pattern 1: Human-in-the-Loop
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("Pattern 1: Human-in-the-loop (approve before sending)")
print("=" * 60)


class EmailState(TypedDict):
    task: str      # input: what the email should be about
    draft: str     # written by draft_email
    approved: bool # set by human_review after human input
    sent: bool     # set by send_email based on approved flag


def draft_email(state: EmailState) -> dict:
    """Node: ask the LLM to write an email draft for the given task."""
    print("  [draft_email] writing draft...")
    draft = (
        ChatPromptTemplate.from_messages([
            ("system", "You are a professional email writer."),
            ("human", "Write a short email for: {task}"),
        ])
        | llm | parser
    ).invoke({"task": state["task"]})
    return {"draft": draft}


def human_review(state: EmailState) -> dict:
    """Node: show the draft and pause until a human approves or rejects.

    interrupt() raises GraphInterrupt, which suspends the graph and returns
    control to the outer .invoke() call. The string argument is a prompt
    shown to whatever code handles the resumption (UI, CLI, test harness).

    Execution resumes here (after the interrupt() line) only after
    update_state() injects the human's decision and .invoke(None) is called.
    """
    print(f"\n  [human_review] Draft email:\n{'-'*40}\n{state['draft']}\n{'-'*40}")
    # Suspend the graph — returns control to the caller until resumed
    decision = interrupt("Type 'approve' to send, or 'reject' to discard.")
    approved = str(decision).strip().lower() == "approve"
    print(f"  [human_review] Human decision: approved={approved}")
    return {"approved": approved}


def send_email(state: EmailState) -> dict:
    """Node: send or discard based on the human's decision."""
    if state["approved"]:
        print("  [send_email] Sending email... done.")
        return {"sent": True}
    print("  [send_email] Email rejected by human — not sent.")
    return {"sent": False}


# Build the HITL graph
eg = StateGraph(EmailState)
eg.add_node("draft_email", draft_email)
eg.add_node("human_review", human_review)
eg.add_node("send_email", send_email)
eg.add_edge(START, "draft_email")
eg.add_edge("draft_email", "human_review")
eg.add_edge("human_review", "send_email")
eg.add_edge("send_email", END)

# MemorySaver is REQUIRED for interrupt() to work.
# It persists the graph state between the two .invoke() calls.
# thread_id in config identifies which "session" to save/resume.
email_app = eg.compile(checkpointer=MemorySaver())

# ── Run the HITL example ──────────────────────────────────────────────────────
config = {"configurable": {"thread_id": "email_1"}}
task = "Request a meeting with IT to discuss the laptop security policy."

print(f"\nTask: {task}")

# First invoke: runs draft_email, then suspends at human_review's interrupt()
# Returns early (before send_email) — the state is saved to MemorySaver
email_app.invoke({"task": task, "approved": False, "sent": False}, config=config)

print("\n(Simulating human approval — injecting approved=True into saved state...)")

# update_state() writes the human's decision into the saved checkpoint.
# as_node="human_review" tells LangGraph to treat this as if human_review
# returned {"approved": True} — execution will resume AFTER human_review.
email_app.update_state(config, {"approved": True}, as_node="human_review")

# Second invoke with input=None: resumes from the checkpoint.
# Picks up right after human_review, runs send_email, then ends.
result = email_app.invoke(None, config=config)
print(f"Final state: sent={result['sent']}")


# ══════════════════════════════════════════════════════════════════════════════
# Pattern 2: Supervisor Multi-Agent
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Pattern 2: Supervisor routes requests to specialist agents")
print("=" * 60)


class SupervisorState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    # add_messages reducer: new messages append to the list, not overwrite
    next_agent: str    # set by supervisor; read by the conditional edge router
    final_answer: str  # set by whichever specialist node runs


# Registered specialist agents — any name here becomes a node and a valid
# routing target. Adding a new specialist = add its name here + add_node below.
AGENTS = ["writer", "analyst", "coder"]


def supervisor(state: SupervisorState) -> dict:
    """Node: read the latest human message and decide which specialist to call.

    The supervisor's only job is routing — it never produces the final answer.
    It writes next_agent to state, which the conditional edge lambda reads.
    """
    # Extract the most recent human message from the history
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

    # Guard: if the LLM hallucinated a name not in AGENTS, default to writer
    chosen = next((a for a in AGENTS if a in decision.lower()), "writer")
    print(f"  [supervisor] routing '{last_human[:60]}...' → '{chosen}'")
    return {"next_agent": chosen}


def make_agent(persona: str):
    """Factory: returns a node function that acts as a specialist with the given persona.

    Using a factory avoids repeating nearly-identical node functions.
    Each specialist sees only the HumanMessages from history — it doesn't
    need the supervisor's intermediate messages.
    """
    def agent_fn(state: SupervisorState) -> dict:
        print(f"  [{persona}] generating response...")
        # Give the specialist only the human messages — strip supervisor artifacts
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

# The conditional edge reads state["next_agent"] and jumps to that node.
# This works because next_agent is always set to a valid node name by supervisor.
sg.add_conditional_edges("supervisor", lambda s: s["next_agent"])

for agent_name in AGENTS:
    sg.add_edge(agent_name, END)  # each specialist goes directly to END

supervisor_app = sg.compile()

# ── Run the supervisor example ────────────────────────────────────────────────
test_requests = [
    "Write a short poem about AI.",                          # → writer
    "Analyse Python vs JavaScript for backends.",            # → analyst
    "Write a Python function to reverse a string.",          # → coder
]

for req in test_requests:
    print(f"\nRequest: {req}")
    result = supervisor_app.invoke({"messages": [HumanMessage(content=req)]})
    print(f"Answer ({result['next_agent']}):\n{result['final_answer'][:300]}...")


# ══════════════════════════════════════════════════════════════════════════════
# Pattern 3: Error Recovery with Fallback
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Pattern 3: Error recovery — graceful fallback on failure")
print("=" * 60)


class RobustState(TypedDict):
    input: str       # the text to process
    result: str      # final output (set by safe_node on success OR error_handler on failure)
    error: str       # empty on success; error message on failure
    recovered: bool  # True if error_handler ran


def safe_node(state: RobustState) -> dict:
    """Node: attempt the main logic. On any exception, write to error key instead of raising.

    The key design: this node NEVER raises. It catches exceptions and stores
    them in state["error"]. The conditional edge below reads that key to
    decide whether to route to error_handler.

    Returning {"error": str(e)} instead of raising means:
      - The graph stays alive and continues to the error_handler
      - The caller always gets a valid result dict (never a crash)
      - The failure is logged in state for observability / debugging
    """
    print(f"  [safe_node] processing input: '{state['input']}'")
    try:
        if state["input"].lower() == "fail":
            raise ValueError("Simulated failure — input='fail' triggers this branch")
        result = llm.invoke(f"Summarise in one sentence: {state['input']}").content
        return {"result": result, "error": ""}  # success: clear the error key
    except Exception as e:
        print(f"  [safe_node] caught exception: {e}")
        return {"error": str(e), "result": ""}  # failure: write to error key


def error_handler(state: RobustState) -> dict:
    """Node: produce a user-friendly fallback response when safe_node failed.

    Only runs when state["error"] is non-empty. Writes a polite fallback
    message to result and sets recovered=True so the caller can tell
    that the normal path was bypassed.
    """
    print(f"  [error_handler] safe_node failed with: '{state['error']}'")
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

# Conditional edge: inspect state["error"] after safe_node runs.
# Non-empty error → divert to error_handler.
# Empty error     → go straight to END (success path).
rg.add_conditional_edges(
    "safe_node",
    lambda s: "error_handler" if s.get("error") else END,
)

rg.add_edge("error_handler", END)
robust_app = rg.compile()

# ── Run the error-recovery example ───────────────────────────────────────────
test_inputs = [
    "the history of artificial intelligence",  # success path → summarised by LLM
    "fail",                                    # failure path → triggers error_handler
]

for inp in test_inputs:
    print(f"\nInput: '{inp}'")
    result = robust_app.invoke({"input": inp, "error": "", "recovered": False})
    print(f"Result: {result['result']}")
    if result.get("recovered"):
        print("  (note: result came from error recovery fallback)")


# ── Pattern recap ─────────────────────────────────────────────────────────────
print("\n--- Advanced Patterns recap ---")
print()
print("1. Human-in-the-Loop:")
print("   interrupt() suspends the graph and saves state to the checkpointer")
print("   update_state() injects the human decision into the saved checkpoint")
print("   invoke(None, config=...) resumes from where it was paused")
print("   MemorySaver required — it persists state between the two invoke() calls")
print()
print("2. Supervisor Multi-Agent:")
print("   supervisor writes next_agent to state (the routing decision)")
print("   conditional edge reads next_agent and jumps to that specialist node")
print("   each specialist owns its domain — narrow prompt = better output")
print("   add a new specialist by adding to AGENTS, add_node, and add_edge(a, END)")
print()
print("3. Error Recovery:")
print("   safe_node catches exceptions and writes to state['error'] (never raises)")
print("   conditional edge routes to error_handler when error is non-empty")
print("   callers always receive a valid result dict — failures degrade gracefully")
