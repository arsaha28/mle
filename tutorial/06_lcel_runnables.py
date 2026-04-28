"""
06 - LCEL Runnable Primitives
==============================
Concept: LCEL building blocks for complex data flows.

Covered:
  - RunnablePassthrough  -> pass input unchanged
  - RunnableParallel     -> run multiple chains simultaneously
  - RunnableLambda       -> wrap any Python function as a Runnable
  - RunnableBranch       -> conditional routing
  - itemgetter           -> extract a key from a dict
"""

from operator import itemgetter
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = StrOutputParser()

print("=== 1. RunnablePassthrough ===")
prompt = ChatPromptTemplate.from_messages([("system", "Answer in one sentence."), ("human", "{question}")])
passthrough_chain = RunnableParallel(answer=(prompt | llm | parser), original_question=RunnablePassthrough())
result = passthrough_chain.invoke({"question": "What is Python?"})
print("Answer:", result["answer"])
print("Original input:", result["original_question"])

print("\n=== 2. RunnableParallel ===")
parallel_chain = RunnableParallel(
    pros=(ChatPromptTemplate.from_messages([("human", "List 2 pros of {technology}.")]) | llm | parser),
    cons=(ChatPromptTemplate.from_messages([("human", "List 2 cons of {technology}.")]) | llm | parser),
)
result = parallel_chain.invoke({"technology": "Python"})
print("Pros:\n", result["pros"])
print("Cons:\n", result["cons"])

print("\n=== 3. RunnableLambda ===")
def word_count(text: str) -> str:
    return f"{text}\n\n[Word count: {len(text.split())}]"

summary_chain = (
    ChatPromptTemplate.from_messages([("human", "Summarise {topic} in 3 sentences.")])
    | llm | parser | RunnableLambda(word_count)
)
print(summary_chain.invoke({"topic": "the history of the internet"}))

print("\n=== 4. RunnableBranch ===")
simple_chain = ChatPromptTemplate.from_messages([("system", "Use very simple words."), ("human", "{question}")]) | llm | parser
advanced_chain = ChatPromptTemplate.from_messages([("system", "Use technical language."), ("human", "{question}")]) | llm | parser
branch = RunnableBranch(
    (lambda x: x.get("level") == "simple", simple_chain),
    (lambda x: x.get("level") == "advanced", advanced_chain),
    simple_chain,
)
print("Simple:", branch.invoke({"question": "What is gravity?", "level": "simple"}))
print("\nAdvanced:", branch.invoke({"question": "What is gravity?", "level": "advanced"}))

print("\n=== 5. itemgetter ===")
topic_chain = (
    {"topic": itemgetter("topic")}
    | ChatPromptTemplate.from_messages([("human", "Give one fun fact about {topic}.")] )
    | llm | parser
)
print(topic_chain.invoke({"topic": "black holes", "unused_key": "ignored"}))
