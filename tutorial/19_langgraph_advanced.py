"""
19 - LangGraph Advanced Patterns
==================================
Concept: Three production-grade LangGraph patterns.

Patterns covered:
  1. Human-in-the-loop -> pause for human approval before acting
  2. Supervisor agent  -> coordinator routes to specialist sub-agents
  3. Error recovery    -> catch failures and route to a fallback handler
"""

from typing import Annotated, Literal, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, AIMessage
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

# Pattern 1: Human-in-the-loop
print("=" * 60 + "\nPattern 1: Human-in-the-loop (approve before sending)\n" + "=" * 60)

class EmailState(TypedDict):
    task: str
    draft: str
    approved: bool
    sent: bool

def draft_email(state: EmailState) -> dict:
    print("  [draft_email] writing...")
    draft = (ChatPromptTemplate.from_messages([("system", "You are a professional email writer."), ("human", "Write a short email for: {task}")]) | llm | parser).invoke({"task": state["task"]})
    return {"draft": draft}

def human_review(state: EmailState) -> dict:
    print(f"\n  [human_review] Draft:\n{'-'*40}\n{state['draft']}\n{'-'*40}")
    # interrupt() suspends the graph; resume by calling .invoke() again after update_state()
    decision = interrupt("Type 'approve' or 'reject'.")
    approved = str(decision).strip().lower() == "approve"
    print(f"  [human_review] approved={approved}")
    return {"approved": approved}

def send_email(state: EmailState) -> dict:
    if state["approved"]:
        print("  [send_email] Email sent!")
        return {"sent": True}
    print("  [send_email] Rejected, not sent.")
    return {"sent": False}

eg = StateGraph(EmailState)
eg.add_node("draft_email", draft_email)
eg.add_node("human_review", human_review)
eg.add_node("send_email", send_email)
eg.add_edge(START, "draft_email")
eg.add_edge("draft_email", "human_review")
eg.add_edge("human_review", "send_email")
eg.add_edge("send_email", END)
email_app = eg.compile(checkpointer=MemorySaver())

config = {"configurable": {"thread_id": "email_1"}}
task = "Request a meeting with IT to discuss the laptop security policy."
print(f"\nTask: {task}")
email_app.invoke({"task": task, "approved": False, "sent": False}, config=config)
print("\n(Simulating human approval...)")
email_app.update_state(config, {"approved": True}, as_node="human_review")
result = email_app.invoke(None, config=config)
print(f"Final: sent={result['sent']}")

# Pattern 2: Supervisor multi-agent
print("\n" + "=" * 60 + "\nPattern 2: Supervisor routes to specialist agents\n" + "=" * 60)

class SupervisorState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    next_agent: str
    final_answer: str

AGENTS = ["writer", "analyst", "coder"]

def supervisor(state: SupervisorState) -> dict:
    last = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    decision = (ChatPromptTemplate.from_messages([("system", f"Choose one from {AGENTS}. Reply with the name only."), ("human", "{request}")]) | llm | parser).invoke({"request": last})
    chosen = next((a for a in AGENTS if a in decision.lower()), "writer")
    print(f"  [supervisor] routing to '{chosen}'")
    return {"next_agent": chosen}

def make_agent(persona):
    def agent_fn(state):
        print(f"  [{persona}]")
        answer = llm.invoke([SystemMessage(content=f"You are an expert {persona}."), *[m for m in state["messages"] if isinstance(m, HumanMessage)]]).content
        return {"final_answer": answer, "messages": [AIMessage(content=answer)]}
    return agent_fn

sg = StateGraph(SupervisorState)
sg.add_node("supervisor", supervisor)
for a in AGENTS:
    sg.add_node(a, make_agent(a))
sg.add_edge(START, "supervisor")
sg.add_conditional_edges("supervisor", lambda s: s["next_agent"])
for a in AGENTS:
    sg.add_edge(a, END)
supervisor_app = sg.compile()

for req in ["Write a short poem about AI.", "Analyse Python vs JavaScript for backends.", "Write a Python function to reverse a string."]:
    print(f"\nRequest: {req}")
    result = supervisor_app.invoke({"messages": [HumanMessage(content=req)]})
    print(f"Answer:\n{result['final_answer'][:250]}...")

# Pattern 3: Error recovery
print("\n" + "=" * 60 + "\nPattern 3: Error recovery with fallback\n" + "=" * 60)

class RobustState(TypedDict):
    input: str
    result: str
    error: str
    recovered: bool

def safe_node(state: RobustState) -> dict:
    print(f"  [safe_node] processing: '{state['input']}'")
    try:
        if state["input"].lower() == "fail":
            raise ValueError("Simulated failure!")
        return {"result": llm.invoke(f"Summarise in one sentence: {state['input']}").content, "error": ""}
    except Exception as e:
        print(f"  [safe_node] caught: {e}")
        return {"error": str(e), "result": ""}

def error_handler(state: RobustState) -> dict:
    print("  [error_handler] generating fallback")
    return {"result": f"Could not process '{state['input']}'. Please rephrase.", "recovered": True}

rg = StateGraph(RobustState)
rg.add_node("safe_node", safe_node)
rg.add_node("error_handler", error_handler)
rg.add_edge(START, "safe_node")
rg.add_conditional_edges("safe_node", lambda s: "error_handler" if s.get("error") else END)
rg.add_edge("error_handler", END)
robust_app = rg.compile()

for inp in ["the history of artificial intelligence", "fail"]:
    print(f"\nInput: '{inp}'")
    result = robust_app.invoke({"input": inp, "error": "", "recovered": False})
    print(f"Result: {result['result']}")
    if result.get("recovered"):
        print("  (recovered from error)")
