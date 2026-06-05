"""
18 - LangGraph ReAct Agent
===========================
Concept: ReAct (Reason + Act) is the dominant pattern for building LLM agents.
The agent loops between three phases until the task is complete:

  1. Reason  → the LLM sees the full conversation history and decides
               what to do next: call a tool, or give the final answer
  2. Act     → if the LLM requested a tool call, execute the tool
  3. Observe → the tool result is appended to the message history,
               and the loop starts again from Reason

The loop terminates when the LLM produces a response with NO tool calls —
that response IS the final answer.

Graph structure:
  START
    │
  agent          ← LLM call: reason about the next step
    │
  should_use_tool()   ← router: did the LLM request a tool?
   ├── yes  →  tool_node    ← execute the requested tool(s)
   │              │
   │            agent       ← loop back: LLM sees the tool result
   └── no   →  END          ← LLM gave a final answer, stop

Why the loop goes back to "agent" (not directly to END after tool):
  The agent must always see the tool result before deciding what to do.
  tool_node → agent creates the observe-then-reason cycle.
  After seeing the result, the LLM might: call another tool, call a
  different tool, or produce the final text answer.

Key LangGraph concepts used here:

  add_messages reducer
  ─────────────────────
  Ordinary TypedDict keys get overwritten on each state update.
  Messages are different — you want to APPEND new messages, not replace
  the whole list.
  Annotated[list[AnyMessage], add_messages] tells LangGraph to use the
  add_messages reducer: when a node returns {"messages": [new_msg]},
  LangGraph appends new_msg to the existing list instead of overwriting.
  This is how the agent accumulates a conversation history across iterations.

  bind_tools()
  ─────────────
  llm.bind_tools(tools) attaches the tool schemas to every LLM call.
  The LLM doesn't execute tools itself — it returns a structured
  AIMessage with a tool_calls list describing which tool to call and
  with what arguments. The graph's tool_node reads that list and
  actually executes the functions.

  ToolMessage
  ────────────
  After a tool runs, its output is wrapped in a ToolMessage.
  ToolMessage.tool_call_id links the result back to the specific
  tool_call that requested it. The LLM needs this link to match
  results to requests when multiple tools are called in one step.

Message flow across a multi-step example
("How many items in 'red, green, blue'? Then square that number."):

  turn 1 — Reason:
    messages = [HumanMessage("How many items...")]
    LLM sees 1 message, decides to call list_length("red, green, blue")
    → returns AIMessage(tool_calls=[{name:"list_length", args:{...}}])
    messages = [Human, AIMessage(tool_calls=[list_length])]

  turn 1 — Act:
    tool_node calls list_length("red, green, blue") → "3"
    → appends ToolMessage(content="3", tool_call_id=...)
    messages = [Human, AI(tool_calls), ToolMessage("3")]

  turn 2 — Reason:
    LLM sees 3 messages, knows list_length returned 3
    decides to call calculate("3 ** 2")
    → returns AIMessage(tool_calls=[{name:"calculate", args:{...}}])
    messages = [Human, AI(tool_calls), Tool("3"), AI(tool_calls)]

  turn 2 — Act:
    tool_node calls calculate("3 ** 2") → "9"
    → appends ToolMessage(content="9")
    messages = [..., ToolMessage("9")]

  turn 3 — Reason:
    LLM sees 5 messages, has all results
    produces final text: "There are 3 items. 3² = 9."
    → returns AIMessage(content="There are 3 items...")  — NO tool_calls
    should_use_tool() → END

Tools used in this file (no external APIs needed):
  calculate    → evaluate a math expression using Python eval (sanitised)
  word_counter → count words in a string
  text_upper   → uppercase a string
  list_length  → count items in a comma-separated list

When to use the ReAct pattern:
  - The task requires information the LLM doesn't have (databases, APIs, files)
  - The task requires computation the LLM can't do reliably (math, code execution)
  - The task requires a sequence of dependent steps that aren't known upfront
  - Any "think → do → think again" workflow
"""

import re
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# temperature=0 — agents should be deterministic; randomness causes
# inconsistent tool selection and unreliable reasoning chains


# ── Tools ─────────────────────────────────────────────────────────────────────
# Each function decorated with @tool becomes a LangChain tool.
# @tool reads the function name, docstring, and type hints to build the
# JSON schema that gets sent to the LLM via bind_tools().
# The LLM uses the docstring to decide WHEN to call each tool.
# Write docstrings that clearly describe what the tool does and its input format.

@tool
def calculate(expression: str) -> str:
    """Evaluate a safe math expression. Supports +, -, *, /, **, parentheses."""
    # Strip everything except digits, operators, and parentheses to prevent
    # code injection via eval — never eval unsanitised user/LLM input
    clean = re.sub(r"[^\d\s\+\-\*\/\(\)\. ]", "", expression)
    try:
        return str(round(eval(clean), 6))
    except Exception as e:
        return f"Error: {e}"


@tool
def word_counter(text: str) -> str:
    """Count the number of words in the given text."""
    return str(len(text.split()))


@tool
def text_upper(text: str) -> str:
    """Convert a string to uppercase."""
    return text.upper()


@tool
def list_length(items: str) -> str:
    """Return the count of items in a comma-separated list. Example: 'a, b, c' -> '3'"""
    return str(len([i.strip() for i in items.split(",") if i.strip()]))


# Register tools and attach them to the LLM
tools = [calculate, word_counter, text_upper, list_length]

# tool_map lets tool_node look up the actual function by name at runtime.
# The LLM returns tool calls as {"name": "calculate", "args": {...}} dicts;
# tool_map[name] retrieves the callable to execute.
tool_map = {t.name: t for t in tools}

# bind_tools() sends the tool schemas to the LLM with every request.
# The LLM itself never runs tools — it only returns structured requests
# saying "call tool X with args Y". tool_node does the actual execution.
llm_with_tools = llm.bind_tools(tools)


# ── State ─────────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    # Annotated[..., add_messages] is the key design choice here:
    # - Without it: each node's {"messages": [...]} would REPLACE the list
    # - With it:    each node's {"messages": [...]} APPENDS to the list
    # The agent needs to see ALL prior messages (the full conversation history)
    # on every iteration. add_messages makes this automatic.
    #
    # AnyMessage is a union of HumanMessage | AIMessage | ToolMessage | etc.
    # Using AnyMessage (not just str) preserves structured metadata like
    # tool_call_id which the LLM uses to match tool results to requests.


# ── Nodes ─────────────────────────────────────────────────────────────────────
def agent_node(state: AgentState) -> dict:
    """Node: send the full message history to the LLM and get the next action.

    The LLM sees every message so far (human query + all previous tool
    calls and results). It either:
      a) Returns an AIMessage with tool_calls → the agent should use a tool
      b) Returns an AIMessage with content only → this is the final answer

    This node runs on EVERY iteration of the loop, not just the first one.
    """
    print(f"  [agent] thinking... ({len(state['messages'])} messages in history)")
    response = llm_with_tools.invoke(state["messages"])

    if response.tool_calls:
        print(f"  [agent] decided to call: {[c['name'] for c in response.tool_calls]}")
    else:
        print(f"  [agent] final answer ready (no more tool calls)")

    # Returning {"messages": [response]} triggers the add_messages reducer:
    # response is appended to state["messages"], not replacing it.
    return {"messages": [response]}


def tool_node(state: AgentState) -> dict:
    """Node: execute every tool call requested in the last AI message.

    The last message in state is always an AIMessage with tool_calls
    (the router guarantees this node only runs when tool_calls is non-empty).

    One AIMessage can contain multiple tool calls — the LLM might ask for
    several tools at once. We execute all of them and return all results.
    Each result is a ToolMessage whose tool_call_id matches the call that
    triggered it. The LLM needs that ID to pair results with requests.
    """
    last_ai_message = state["messages"][-1]
    tool_results = []

    for call in last_ai_message.tool_calls:
        # call is a dict: {"id": "...", "name": "calculate", "args": {"expression": "3+4"}}
        result = tool_map[call["name"]].invoke(call["args"])
        print(f"  [tool:{call['name']}] args={call['args']} → result={result!r}")

        tool_results.append(
            ToolMessage(
                content=str(result),
                tool_call_id=call["id"],  # links this result to the specific request
            )
        )

    # All ToolMessages are appended to the history via add_messages.
    # On the next agent_node call the LLM will see them and reason about results.
    return {"messages": tool_results}


# ── Router ────────────────────────────────────────────────────────────────────
def should_use_tool(state: AgentState) -> str:
    """Router: inspect the last message to decide if a tool should be called.

    Called after agent_node. Two outcomes:
      - LLM produced tool_calls → route to tool_node (continue the loop)
      - LLM produced text only  → route to END (task is complete)

    This is the sole exit condition for the ReAct loop. The loop runs
    indefinitely until the LLM stops requesting tools.
    In practice, give the LLM a system prompt with a step limit if you
    want a hard cap (or use LangGraph's recursion_limit on compile()).
    """
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tool_node"
    return END


# ── Build the graph ───────────────────────────────────────────────────────────
agent_graph = StateGraph(AgentState)

agent_graph.add_node("agent", agent_node)
agent_graph.add_node("tool_node", tool_node)

# Entry point: always start with the agent (LLM reasons first)
agent_graph.add_edge(START, "agent")

# After agent_node: conditionally go to tool_node or END
agent_graph.add_conditional_edges("agent", should_use_tool)

# After tool_node: always go back to agent (observe → reason again)
# This is the edge that creates the ReAct loop.
agent_graph.add_edge("tool_node", "agent")

# compile() validates the graph (checks for unreachable nodes, missing edges)
# and returns a Pregel-based runnable. After this, no more nodes/edges can be added.
app = agent_graph.compile()
# Optional: set a recursion limit to prevent infinite loops
# app = agent_graph.compile(checkpointer=...) for persistence / resumability


# ── Helper ────────────────────────────────────────────────────────────────────
def run(question: str):
    """Invoke the agent and print the question and final answer."""
    print(f"\n{'='*60}")
    print(f"Q: {question}")
    print(f"{'='*60}")
    result = app.invoke({"messages": [HumanMessage(content=question)]})
    # result["messages"] is the complete conversation history.
    # The last message is always the final AIMessage (no tool_calls).
    print(f"A: {result['messages'][-1].content}")


# ── Run ───────────────────────────────────────────────────────────────────────
print("18 - LangGraph ReAct Agent")
print("Tools: calculate, word_counter, text_upper, list_length\n")

# Single tool call — agent calls calculate once and returns the answer
run("What is 15 multiplied by 23, then add 100?")

# Single tool call — agent calls word_counter
run("How many words are in 'the quick brown fox jumps over the lazy dog'?")

# Multi-step: list_length first, then feed result into calculate
# Demonstrates the loop: agent → tool → agent → tool → agent → END
run("How many items in 'red, green, blue, yellow, purple'? Then calculate that number squared.")

# Single tool call — agent calls text_upper
run("Convert 'hello from langchain' to uppercase.")


# ── Pattern recap ─────────────────────────────────────────────────────────────
print("\n--- ReAct Agent pattern recap ---")
print("State: Annotated[list[AnyMessage], add_messages] → messages accumulate, never overwrite")
print("bind_tools(tools) → LLM knows which tools exist and when to call them")
print("agent_node → LLM sees full history, returns AIMessage with or without tool_calls")
print("tool_node  → executes tool_calls, wraps each result in ToolMessage")
print("should_use_tool → routes to tool_node (loop) or END (done)")
print("tool_node always goes back to agent → the observe-then-reason cycle")
print("Loop exits only when the LLM produces a response with no tool_calls")
