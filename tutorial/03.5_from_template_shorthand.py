"""
03.5 - from_template Shorthand
================================
Concept: from_template() is a convenience class method that creates a template
directly from a string, without needing to specify input_variables manually.
LangChain parses the {variables} out of the string automatically.

When to use:
  - Quick one-liners where you don't need fine-grained control
  - Prototyping and exploration
  - Simple templates with obvious variables

Comparison:
  Verbose (explicit):
      PromptTemplate(
          input_variables=["text", "language"],
          template="Translate '{text}' to {language}."
      )

  Shorthand (from_template):
      PromptTemplate.from_template("Translate '{text}' to {language}.")

  Both are identical in behaviour. from_template just saves you listing the
  variables — LangChain scans the string and finds them for you.

Works on:
  - PromptTemplate.from_template()
  - ChatPromptTemplate.from_template()      ← single human message, no system
  - SystemMessagePromptTemplate.from_template()
  - HumanMessagePromptTemplate.from_template()
"""

from dotenv import load_dotenv
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ── PromptTemplate.from_template ────────────────────────────────────────────────
# LangChain auto-detects {text} and {language} — no input_variables needed.
print("=== PromptTemplate.from_template ===")
quick = PromptTemplate.from_template("Translate '{text}' to {language}.")
print("Auto-detected variables:", quick.input_variables)
# → ['text', 'language']
print(llm.invoke(quick.format(text="Hello, how are you?", language="Spanish")).content)

# ── ChatPromptTemplate.from_template ─────────────────────────────────────────
# Creates a single HumanMessage — no system message.
# Use when you just want a quick chat call with one human turn.
print("\n=== ChatPromptTemplate.from_template (single human message) ===")
single = ChatPromptTemplate.from_template("Summarise this in one sentence: {text}")
chain = single | llm
print(chain.invoke({"text": "Machine learning is a subset of artificial intelligence that enables systems to learn from data."}).content)

# ── SystemMessagePromptTemplate.from_template ─────────────────────────────────
# Shorthand for creating a typed system message with variables.
# Often used when assembling components (as shown in 03.3).
print("\n=== SystemMessagePromptTemplate.from_template ===")
sys_tpl = SystemMessagePromptTemplate.from_template("You are a helpful {style} assistant.")
hum_tpl = HumanMessagePromptTemplate.from_template("What is {topic}?")

chat = ChatPromptTemplate.from_messages([sys_tpl, hum_tpl]) | llm
print(chat.invoke({"style": "concise", "topic": "the water cycle"}).content)

# ── Side-by-side: verbose vs shorthand ────────────────────────────────────────────
print("\n=== Verbose vs shorthand — identical output ===")

verbose = PromptTemplate(
    input_variables=["product"],
    template="Name three benefits of {product}.",
)
shorthand = PromptTemplate.from_template("Name three benefits of {product}.")

# Both produce the same filled string
print("Verbose filled:   ", verbose.format(product="standing desk"))
print("Shorthand filled: ", shorthand.format(product="standing desk"))
# → identical strings
