"""
14 - Conversation Memory
========================
Concept: LLMs are stateless by default. Memory maintains conversation
history so the model can refer back to earlier turns.

Patterns covered:
  1. InMemoryChatMessageHistory -> store all messages
  2. Sliding window memory      -> keep only last N messages
  3. Summarisation memory       -> compress old history into a summary
"""

from dotenv import load_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, trim_messages
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = StrOutputParser()

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Be concise."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])
chain = prompt | llm | parser

# Pattern 1: Full in-memory history
print("=" * 55 + "\nPattern 1: Full in-memory history\n" + "=" * 55)
store = {}

def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

chat = RunnableWithMessageHistory(chain, get_session_history, input_messages_key="question", history_messages_key="history")
config = {"configurable": {"session_id": "user_alice"}}
for q in ["My name is Alice and I work as a data engineer.", "What technologies do I likely use?", "What did I say my name was?"]:
    print(f"\nUser: {q}")
    print(f"Bot:  {chat.invoke({'question': q}, config=config)}")

# Pattern 2: Sliding window
print("\n" + "=" * 55 + "\nPattern 2: Sliding window (last 4 messages)\n" + "=" * 55)

def make_windowed_chain(max_messages):
    def trim(messages):
        return trim_messages(messages, max_tokens=max_messages, token_counter=len, strategy="last", include_system=True)
    windowed_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are a helpful assistant. Be concise."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])
    return ({"history": lambda x: trim(x["history"]), "question": lambda x: x["question"]} | windowed_prompt | llm | parser)

windowed_chat = RunnableWithMessageHistory(
    make_windowed_chain(4),
    lambda sid: store.setdefault(f"w_{sid}", InMemoryChatMessageHistory()),
    input_messages_key="question", history_messages_key="history",
)
config_w = {"configurable": {"session_id": "windowed"}}
for q in ["I like cats.", "I also like dogs.", "I enjoy hiking.", "What animals did I mention?"]:
    print(f"\nUser: {q}")
    print(f"Bot:  {windowed_chat.invoke({'question': q}, config=config_w)}")

# Pattern 3: Summarisation memory
print("\n" + "=" * 55 + "\nPattern 3: Summarisation memory\n" + "=" * 55)
summarise_chain = ChatPromptTemplate.from_messages([
    ("system", "Summarise this conversation in 2 sentences."),
    MessagesPlaceholder(variable_name="messages"),
]) | llm | parser

class SummarisedHistory:
    def __init__(self, window=4):
        self.messages, self.summary, self.window = [], "", window
    def add(self, human, ai):
        self.messages += [HumanMessage(content=human), AIMessage(content=ai)]
        if len(self.messages) > self.window:
            self.summary = summarise_chain.invoke({"messages": self.messages[:-self.window]})
            self.messages = self.messages[-self.window:]
    def get_context(self):
        msgs = [SystemMessage(content=f"Summary so far: {self.summary}")] if self.summary else []
        return msgs + self.messages

mem = SummarisedHistory(window=4)
direct_chain = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="context"),
    ("human", "{question}"),
]) | llm | parser

for q in ["My favourite colour is blue.", "I have two cats named Luna and Mochi.", "I work in machine learning.", "I live in London.", "What do you know about me?"]:
    print(f"\nUser: {q}")
    answer = direct_chain.invoke({"context": mem.get_context(), "question": q})
    print(f"Bot:  {answer}")
    mem.add(q, answer)
    if mem.summary:
        print(f"[Summary: {mem.summary[:80]}...]")
