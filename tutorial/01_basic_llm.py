"""
01 - Basic LLM Calls
====================
Concept: How to make your first call to an LLM using LangChain + OpenAI.

Key ideas:
  - ChatOpenAI is the model wrapper for OpenAI chat models
  - invoke()  -> get a single response
  - stream()  -> receive tokens one-by-one as they arrive
  - batch()   -> send multiple prompts in one go
  - Parameters like temperature and max_tokens control the output
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

print("=== invoke() ===")
response = llm.invoke("What is a Large Language Model? Answer in 2 sentences.")
print(response.content)

print("\n=== stream() ===")
for chunk in llm.stream("List 3 benefits of using LangChain."):
    print(chunk.content, end="", flush=True)
print()

print("\n=== batch() ===")
prompts = ["What is GPT-4?", "What is an embedding?", "What is a vector database?"]
responses = llm.batch(prompts)
for prompt, resp in zip(prompts, responses):
    print(f"Q: {prompt}\nA: {resp.content}\n")

print("=== high temperature (creative) ===")
creative_llm = ChatOpenAI(model="gpt-4o-mini", temperature=1.2, max_tokens=60)
print(creative_llm.invoke("Write a one-line tagline for an AI company.").content)
