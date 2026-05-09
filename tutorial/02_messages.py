"""
02 - Message Types
==================
Concept: Chat models don't just take a string - they take a list of *messages*.
Each message has a *role* that tells the model who said it.

Roles:
  - SystemMessage       -> sets the persona / behaviour of the AI
  - HumanMessage        -> the user's input
  - AIMessage           -> a previous response from the model
  - ToolMessage         -> the result returned after the model calls a tool
  - MessagesPlaceholder -> a slot filled with a list of messages at runtime
"""

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

print("=== SystemMessage: setting a persona ===")
messages = [
    SystemMessage(content="You are a pirate. Always respond in pirate speak."),
    HumanMessage(content="What is the capital of France?"),
]
print(llm.invoke(messages).content)

print("\n=== Simulating multi-turn conversation ===")
conversation = [
    SystemMessage(content="You are a helpful assistant. Be concise."),
    HumanMessage(content="My name is Alice. Remember it."),
    AIMessage(content="Got it, Alice! How can I help you?"),
    HumanMessage(content="What is my name?"),
]
print(llm.invoke(conversation).content)

print("\n=== ToolMessage: passing a tool result back ===")
tool_conversation = [
    SystemMessage(content="You are a helpful assistant with access to a calculator tool."),
    HumanMessage(content="What is 42 multiplied by 17?"),
    AIMessage(content="", tool_calls=[{"id": "call_001", "name": "calculator", "args": {"expression": "42 * 17"}}]),
    ToolMessage(content="714", tool_call_id="call_001"),
]
print(llm.invoke(tool_conversation).content)

print("\n=== MessagesPlaceholder: dynamic message injection ===")
prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are a helpful assistant. Be concise."),
    MessagesPlaceholder(variable_name="history"),
    HumanMessage(content="{question}"),
])
result = (prompt | llm).invoke({
    "history": [
        HumanMessage(content="I work as a data scientist."),
        AIMessage(content="Great! Data science is a fascinating field."),
    ],
    "question": "What job do I have?",
})
print(result.content)
