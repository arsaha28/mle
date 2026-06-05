"""
19.1 - LangGraph Human-in-the-Loop (HITL)
==========================================
Concept: Pause the graph mid-execution and wait for a human decision before
continuing. This is NOT just "ask the user a question inside a node" — the
graph fully suspends, serialises its state to a persistent store (checkpointer),
and resumes later when the human provides input.

Graph structure:
  START
    │
  draft_email     ← LLM writes the email draft
    │
  human_review    ← interrupt() suspends here, waits for human input
    │
  send_email      ← proceeds or discards based on the approval flag
    │
   END

How interrupt() works — step by step:
  1. graph.invoke(initial_state, config=config) starts normally.
  2. draft_email runs and writes state["draft"].
  3. human_review calls interrupt("prompt") — this raises a GraphInterrupt
     exception inside LangGraph, which:
       a. Saves the current state to the checkpointer under thread_id
       b. Returns control to the outer .invoke() call (which returns early)
  4. The human sees the draft (printed before interrupt) and decides.
  5. email_app.update_state(config, {"approved": True}, as_node="human_review")
     injects the human's decision into the saved checkpoint, as if human_review
     had returned {"approved": True}. as_node tells LangGraph that execution
     should resume AFTER human_review, not re-run it.
  6. email_app.invoke(None, config=config) resumes from the checkpoint.
     input=None means "don't add a new message — resume where we left off."
     Execution continues with the next node: send_email.

Why a checkpointer is required:
  Without a checkpointer, interrupt() has nowhere to persist state.
  The graph would be garbage-collected between the two .invoke() calls
  and could not resume. MemorySaver keeps everything in RAM — fine for
  development and testing. For production use a durable backend:
    - langgraph.checkpoint.postgres.PostgresSaver
    - langgraph.checkpoint.redis.RedisSaver

thread_id:
  The "configurable": {"thread_id": "..."} in the config acts as a session ID.
  It identifies WHICH saved conversation to resume. Use a unique ID per
  user or workflow instance so parallel sessions don't interfere.

When to use:
  - Email / document approval: generate → human reviews → send
  - Code review: generate code → human inspects → deploy
  - Purchase authorisation: plan action → manager approves → execute
  - Any workflow where an irreversible action needs human sign-off first
"""

from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = StrOutputParser()


# ── State ─────────────────────────────────────────────────────────────────────
class EmailState(TypedDict):
    task: str      # input: what the email should be about
    draft: str     # written by draft_email
    approved: bool # injected by human via update_state() after interrupt
    sent: bool     # written by send_email based on the approved flag


# ── Nodes ─────────────────────────────────────────────────────────────────────
def draft_email(state: EmailState) -> dict:
    """Node: ask the LLM to write a professional email draft."""
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
    """Node: display the draft and pause until a human approves or rejects.

    interrupt() suspends the graph here. The string argument is the prompt
    shown to whatever interface handles the resumption (web UI, CLI, test harness).
    The return value of interrupt() is whatever was passed to update_state().

    Execution resumes after the interrupt() line only after:
      1. update_state() injects the human decision
      2. invoke(None, config=...) is called to resume
    """
    print(f"\n  [human_review] Draft email:\n{'-'*40}\n{state['draft']}\n{'-'*40}")

    # Suspend the graph — the outer .invoke() call returns here.
    # Resumption passes the human's input as the return value of interrupt().
    decision = interrupt("Review the draft above. Type 'approve' to send, or 'reject' to discard.")

    approved = str(decision).strip().lower() == "approve"
    print(f"  [human_review] Human decision: {'APPROVED' if approved else 'REJECTED'}")
    return {"approved": approved}


def send_email(state: EmailState) -> dict:
    """Node: execute the send (or discard) based on the human's decision."""
    if state["approved"]:
        print("  [send_email] Sending email... done.")
        return {"sent": True}
    print("  [send_email] Email rejected by human — not sent.")
    return {"sent": False}


# ── Build the graph ───────────────────────────────────────────────────────────
eg = StateGraph(EmailState)
eg.add_node("draft_email", draft_email)
eg.add_node("human_review", human_review)
eg.add_node("send_email", send_email)

eg.add_edge(START, "draft_email")
eg.add_edge("draft_email", "human_review")
eg.add_edge("human_review", "send_email")
eg.add_edge("send_email", END)

# MemorySaver is REQUIRED — interrupt() saves state to the checkpointer so
# the graph can be resumed in a separate .invoke() call.
email_app = eg.compile(checkpointer=MemorySaver())


# ── Run ───────────────────────────────────────────────────────────────────────
print("=" * 60)
print("19.1 - Human-in-the-Loop (approve before sending)")
print("=" * 60)

# thread_id identifies this session in the checkpointer
config = {"configurable": {"thread_id": "email_1"}}
task = "Request a meeting with IT to discuss the laptop security policy."

print(f"\nTask: {task}\n")

# Phase 1: run until interrupt() suspends at human_review
# The graph saves state and returns early — send_email has NOT run yet
email_app.invoke({"task": task, "approved": False, "sent": False}, config=config)

print("\n(Simulating human review — injecting approval into saved state...)\n")

# Phase 2: inject the human's decision into the saved checkpoint.
# as_node="human_review" means: treat this as if human_review returned
# {"approved": True}, so execution resumes AFTER human_review.
email_app.update_state(config, {"approved": True}, as_node="human_review")

# Phase 3: resume from the checkpoint — runs send_email, then END
result = email_app.invoke(None, config=config)
print(f"\nFinal state: sent={result['sent']}")


# ── Pattern recap ─────────────────────────────────────────────────────────────
print("\n--- HITL pattern recap ---")
print("compile(checkpointer=MemorySaver()) → enables state persistence across invoke() calls")
print("interrupt('prompt')                → suspends graph, saves state, returns to caller")
print("update_state(config, data, as_node=...) → injects human input into saved checkpoint")
print("invoke(None, config=config)        → resumes from the checkpoint (None = no new input)")
print("thread_id in config                → identifies which saved session to resume")
print("Use PostgresSaver / RedisSaver in production for durable persistence")
