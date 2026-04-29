"""
18 - LangGraph ReAct Agent
===========================
Concept: A ReAct agent loops between:
  1. Reasoning  -> LLM decides what to do
  2. Acting     -> calls a tool
  3. Observing  -> tool result fed back in
  ...until the LLM gives a final answer (no more tool calls).

Tools (no external APIs needed):
  - calculate    -> evaluate math expressions
  - word_counter -> count words in text
  - text_upper   -> uppercase a string
  - list_length  -> length of a comma-separated list
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

@tool
def calculate(expression: str) -> str:
    """Evaluate a safe math expression. Supports +, -, *, /, **, parentheses."""
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

tools = [calculate, word_counter, text_upper, list_length]
tool_map = {t.name: t for t in tools}
llm_with_tools = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def agent_node(state: AgentState) -> dict:
    """LLM reasons and decides whether to call a tool or give a final answer."""
    print(f"  [agent] thinking... ({len(state['messages'])} messages)")
    response = llm_with_tools.invoke(state["messages"])
    if response.tool_calls:
        print(f"  [agent] calling: {[c['name'] for c in response.tool_calls]}")
    else:
        print(f"  [agent] final answer ready")
    return {"messages": [response]}

def tool_node(state: AgentState) -> dict:
    """Execute all tool calls from the last AI message."""
    last = state["messages"][-1]
    results = []
    for call in last.tool_calls:
        result = tool_map[call["name"]].invoke(call["args"])
        print(f"  [tool:{call['name']}] {call['args']} -> {result}")
        results.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
    return {"messages": results}

def should_use_tool(state: AgentState) -> str:
    last = state["messages"][-1]
    return "tool_node" if hasattr(last, "tool_calls") and last.tool_calls else END

agent_graph = StateGraph(AgentState)
agent_graph.add_node("agent", agent_node)
agent_graph.add_node("tool_node", tool_node)
agent_graph.add_edge(START, "agent")
agent_graph.add_conditional_edges("agent", should_use_tool)
agent_graph.add_edge("tool_node", "agent")
app = agent_graph.compile()

def run(question: str):
    print(f"\n{'='*55}\nQ: {question}\n{'='*55}")
    result = app.invoke({"messages": [HumanMessage(content=question)]})
    print(f"A: {result['messages'][-1].content}")

run("What is 15 multiplied by 23, then add 100?")
run("How many words are in 'the quick brown fox jumps over the lazy dog'?")
run("How many items in 'red, green, blue, yellow, purple'? Then calculate that number squared.")
run("Convert 'hello from langchain' to uppercase.")
