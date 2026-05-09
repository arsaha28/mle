"""
02 - Message Types
==================
Concept: Chat models don't just take a string — they take a list of *messages*.
Each message has a *role* that tells the model who said it.

Roles:
  - SystemMessage   → sets the persona / behaviour of the AI (not shown to end users)
  - HumanMessage    → the user's input
  - AIMessage       → a previous response from the model (used in conversation history)
  - ToolMessage     → the result returned after the model calls a tool/function
  - MessagesPlaceholder → a slot in a template that gets filled with a list of messages at runtime
"""

from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ── 1. SystemMessage — give the model a persona ───────────────────────────────
print("=== SystemMessage: setting a persona ===")
messages = [
    SystemMessage(content="You are a pirate. Always respond in pirate speak."),
    HumanMessage(content="What is the capital of France?"),
]
response = llm.invoke(messages)
print(response.content)
# The system message shapes *how* the model answers, not what it knows.

# ── 2. HumanMessage + AIMessage — simulating a conversation ──────────────────
print("\n=== Simulating multi-turn conversation ===")
conversation = [
    SystemMessage(content="You are a helpful assistant. Be concise."),
    HumanMessage(content="My name is Alice. Remember it."),
    AIMessage(content="Got it, Alice! How can I help you?"),
    HumanMessage(content="What is my name?"),
]
response = llm.invoke(conversation)
print(response.content)
# By including the AIMessage, we give the model memory of the prior exchange.

# ── 3. ToolMessage — result of a tool/function call ──────────────────────────
print("\n=== ToolMessage: passing a tool result back ===")
# NOTE: No real tool is defined or executed here. This section *simulates* what
# a tool exchange looks like in the message list, so you can see the structure.
#
# How it works:
#   Step 1 — AIMessage with tool_calls: we hand-craft a message that looks like
#             the model decided to call a "calculator" tool. The model never
#             actually made this decision — we wrote it manually.
#   Step 2 — ToolMessage: we hard-code the result ("714") as if a calculator
#             ran "42 * 17". No function ran — we just typed the answer.
#   Step 3 — llm.invoke(): the model receives this full "conversation" including
#             the fake tool exchange and responds naturally with the result.
#
# Why do this? This file is about message *types* and conversation structure.
# Actual tool mechanics (defining tools, auto-calling, execution loop) are
# covered in 15_tools.py.
tool_conversation = [
    SystemMessage(content="You are a helpful assistant with access to a calculator tool."),
    HumanMessage(content="What is 42 multiplied by 17?"),
    AIMessage(
        content="",
        tool_calls=[{"id": "call_001", "name": "calculator", "args": {"expression": "42 * 17"}}],
    ),
    ToolMessage(content="714", tool_call_id="call_001"),
]
response = llm.invoke(tool_conversation)
print(response.content)

# ── 4. MessagesPlaceholder — dynamic slot inside a prompt template ────────────
print("\n=== MessagesPlaceholder: dynamic message injection ===")
prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are a helpful assistant. Be concise."),
    MessagesPlaceholder(variable_name="history"),  # filled at runtime
    HumanMessage(content="{question}"),
])

# Simulate injecting existing history at runtime
chain = prompt | llm
result = chain.invoke({
    "history": [
        HumanMessage(content="I work as a data scientist."),
        AIMessage(content="Great! Data science is a fascinating field."),
    ],
    "question": "What job do I have?",
})
print(result.content)
