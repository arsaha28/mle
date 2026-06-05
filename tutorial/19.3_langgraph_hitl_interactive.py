"""
19.3 - LangGraph Human-in-the-Loop (Interactive Terminal)
===========================================================
This is the interactive companion to 19.1. The mechanics are identical —
interrupt(), checkpointer, update_state(), invoke(None) — but instead of
simulating the human response programmatically, this script pauses and
waits for you to type "approve" or "reject" in the terminal.

How to run:
  python tutorial/19.3_langgraph_hitl_interactive.py
  → the graph drafts the email, then pauses and prints a prompt
  → you type "approve" or "reject" and press Enter
  → the graph resumes and sends or discards the email

What happens under the hood (same as 19.1):
  1. invoke(initial_state) runs draft_email, then hits interrupt() in
     human_review and returns early. The state (including the draft) is
     saved to MemorySaver under thread_id.
  2. The calling code (this script) reads the draft from the returned state
     and calls Python's built-in input() to collect your decision.
  3. update_state() injects your decision into the saved checkpoint.
  4. invoke(None) resumes the graph from the checkpoint, runs send_email,
     and returns the final state.

The key insight:
  interrupt() does NOT collect input itself — it only suspends the graph.
  It is the CALLER's job to gather the human input (via a web form, API
  endpoint, or as here, Python's input()) and pass it back via update_state().
  This separation keeps LangGraph agnostic about HOW input is collected.

Graph structure:
  START
    │
  draft_email     ← LLM writes the email draft
    │
  human_review    ← interrupt() suspends here
    │               ↑ calling code calls input(), then update_state()
  send_email      ← runs after resumption
    │
   END
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
    task: str      # input: what the email should cover
    draft: str     # written by draft_email; shown to the human before approval
    approved: bool # injected by the calling code after reading terminal input
    sent: bool     # set by send_email


# ── Nodes ─────────────────────────────────────────────────────────────────────
def draft_email(state: EmailState) -> dict:
    """Node: LLM writes a professional email draft for the given task."""
    print("\n  [draft_email] Writing email draft...\n")
    draft = (
        ChatPromptTemplate.from_messages([
            ("system", "You are a professional email writer."),
            ("human", "Write a short email for: {task}"),
        ])
        | llm | parser
    ).invoke({"task": state["task"]})
    return {"draft": draft}


def human_review(state: EmailState) -> dict:
    """Node: suspend the graph and wait for a human decision.

    interrupt() saves the current state to the checkpointer and raises
    GraphInterrupt internally — the outer invoke() call returns early.
    The calling code then collects real terminal input and calls
    update_state() to inject the decision, then invoke(None) to resume.

    When resumption happens, execution continues AFTER the interrupt() call,
    using the value injected by update_state() as the return of interrupt().
    """
    # interrupt() suspends here — the graph does not reach "approved = ..." yet.
    # On resumption, interrupt() returns the value passed via update_state().
    decision = interrupt("Waiting for human review...")

    approved = str(decision).strip().lower() == "approve"
    print(f"\n  [human_review] Decision received: {'APPROVED ✓' if approved else 'REJECTED ✗'}")
    return {"approved": approved}


def send_email(state: EmailState) -> dict:
    """Node: send or discard the email based on the approval flag."""
    if state["approved"]:
        print("  [send_email] Email sent successfully.")
        return {"sent": True}
    print("  [send_email] Email discarded — not sent.")
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

email_app = eg.compile(checkpointer=MemorySaver())


# ── Run ───────────────────────────────────────────────────────────────────────
print("=" * 60)
print("19.3 - Human-in-the-Loop (Interactive Terminal)")
print("=" * 60)

config = {"configurable": {"thread_id": "interactive_email_1"}}
task = "Request a meeting with IT to discuss the laptop security policy."

print(f"\nTask: {task}")

# ── Phase 1: run until interrupt ──────────────────────────────────────────────
# The graph runs draft_email, then hits interrupt() in human_review and
# returns early. The returned state contains the draft written so far.
state_at_interrupt = email_app.invoke(
    {"task": task, "approved": False, "sent": False},
    config=config,
)

# ── Phase 2: collect real human input ────────────────────────────────────────
# The graph is now suspended. We display the draft and wait for terminal input.
# This is where YOUR input happens — no simulation.
print("\n" + "=" * 60)
print("DRAFT EMAIL (requires your review):")
print("=" * 60)
print(state_at_interrupt["draft"])
print("=" * 60)

while True:
    raw = input("\nType 'approve' to send or 'reject' to discard: ").strip().lower()
    if raw in ("approve", "reject"):
        break
    print("  Please type exactly 'approve' or 'reject'.")

# ── Phase 3: inject the decision and resume ───────────────────────────────────
# update_state() writes the human's input into the saved checkpoint.
# as_node="human_review" tells LangGraph that execution should resume
# AFTER human_review — so human_review won't run again.
email_app.update_state(config, {"approved": raw == "approve"}, as_node="human_review")

# invoke(None) resumes from the checkpoint.
# None means "no new initial input — continue from where we left off."
final = email_app.invoke(None, config=config)

# ── Result ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"Final state: sent={final['sent']}, approved={final['approved']}")
print("=" * 60)

print("\n--- What just happened ---")
print("1. invoke(initial)    → graph ran until interrupt(), saved state, returned early")
print("2. input()            → YOU provided the approval decision in the terminal")
print("3. update_state()     → your decision was injected into the saved checkpoint")
print("4. invoke(None)       → graph resumed after human_review, ran send_email, ended")
