"""
05 - LCEL Basics (LangChain Expression Language)
=================================================
Concept: LCEL composes components using the pipe operator |.
Each component is a Runnable with invoke(), stream(), and batch().

Covered:
  - Basic prompt | llm | parser chain
  - invoke, stream, batch
  - Multi-step chains
  - Inspecting with .get_graph()
"""

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = StrOutputParser()

print("=== 1. Basic chain: prompt | llm | parser ===")
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a concise assistant. Answer in one sentence."),
    ("human", "{question}"),
])
chain = prompt | llm | parser
print(chain.invoke({"question": "What is a neural network?"}))

print("\n=== 2. Streaming the chain ===")
for token in chain.stream({"question": "What is transfer learning?"}):
    print(token, end="", flush=True)
print()

print("\n=== 3. Batch: multiple inputs at once ===")
questions = [{"question": "What is overfitting?"}, {"question": "What is a transformer model?"}, {"question": "What is tokenisation?"}]
for q, r in zip(questions, chain.batch(questions)):
    print(f"Q: {q['question']}\nA: {r}\n")

print("=== 4. Two-step chain ===")
explain_prompt = ChatPromptTemplate.from_messages([("system", "Explain the concept in one sentence."), ("human", "{concept}")])
translate_prompt = ChatPromptTemplate.from_messages([("system", "Translate the English text to French."), ("human", "{text}")])
two_step = (explain_prompt | llm | parser | (lambda text: {"text": text}) | translate_prompt | llm | parser)
print(two_step.invoke({"concept": "machine learning"}))

print("\n=== 5. Chain graph ===")
chain.get_graph().print_ascii()
