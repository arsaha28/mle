"""
03.3 - Typed Message Templates
===============================
Concept: SystemMessagePromptTemplate and HumanMessagePromptTemplate are typed
wrappers around PromptTemplate. Instead of using the ("role", "content") tuple
shorthand, you explicitly construct each message template as an object.

When to use:
  - When you want to be explicit about message types for clarity
  - When building reusable system/human message components that get assembled
    into different ChatPromptTemplates
  - When you need to attach metadata or custom config to individual messages

Comparison:
  Shorthand (03.2):  ("system", "You are a {language} tutor.")
  Typed (this file): SystemMessagePromptTemplate.from_template("You are a {language} tutor.")
  Both produce identical output — it's a style / reusability choice.
"""

from dotenv import load_dotenv
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ── Defining typed message templates independently ────────────────────────────
# Each template is its own object — you can create them once and reuse them
# across multiple ChatPromptTemplates.
system_tpl = SystemMessagePromptTemplate.from_template(
    "You are a {language} programming tutor."
)
human_tpl = HumanMessagePromptTemplate.from_template(
    "Show me a minimal example of {concept} in {language}."
)

# Assemble them into a ChatPromptTemplate just like the tuple shorthand.
chat_template = ChatPromptTemplate.from_messages([system_tpl, human_tpl])
chain = chat_template | llm

print("=== Typed message templates ===")
print(chain.invoke({"language": "Python", "concept": "list comprehension"}).content)

# ── The real benefit: reusable components ─────────────────────────────────────
# Because system_tpl and human_tpl are standalone objects, you can mix and
# match them to build different templates without rewriting the messages.
print("\n--- Reusing the same system template with a different human message ---")

explain_tpl = HumanMessagePromptTemplate.from_template(
    "Explain {concept} in simple terms, as if I am a beginner."
)
debug_tpl = HumanMessagePromptTemplate.from_template(
    "What are the most common bugs beginners make with {concept} in {language}?"
)

explain_chain = ChatPromptTemplate.from_messages([system_tpl, explain_tpl]) | llm
debug_chain   = ChatPromptTemplate.from_messages([system_tpl, debug_tpl])   | llm

print("\nExplain chain:")
print(explain_chain.invoke({"language": "Python", "concept": "decorators"}).content)

print("\nDebug chain:")
print(debug_chain.invoke({"language": "Python", "concept": "decorators"}).content)

# ── Inspecting what the typed template produces ───────────────────────────────
print("\n--- Inspecting filled messages ---")
filled = chat_template.format_messages(language="Python", concept="list comprehension")
for msg in filled:
    print(f"[{msg.__class__.__name__}] {msg.content}")
# Identical to the shorthand version in 03.2 — same output, just more explicit code.
