"""
03.1 - PromptTemplate (Plain String Template)
=============================================
Concept: PromptTemplate is the simplest template type. It takes a plain string
with {variables} and fills them in at runtime to produce a single string prompt.

When to use:
  - When you just need to format a string before sending it to the model
  - As a building block inside larger templates
  - When working with older completion-style models (not chat models)

Output type: a single plain string (not a list of messages)
"""

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ── Basic usage ─────────────────────────────────────────────────────────────────
# Define the template once with named {variables}.
# input_variables lists every variable the template expects.
template = PromptTemplate(
    input_variables=["product", "audience"],
    template="Write a one-sentence product description for {product} aimed at {audience}.",
)

# .format() fills in the variables and returns the final string.
# Nothing is sent to the LLM yet — this is just string formatting.
filled = template.format(product="smart water bottle", audience="fitness enthusiasts")
print("Filled prompt:", filled)
# → "Write a one-sentence product description for smart water bottle aimed at fitness enthusiasts."

# Now send the filled string to the LLM.
print("LLM response:", llm.invoke(filled).content)

# ── Key distinction from ChatPromptTemplate ───────────────────────────────────
# PromptTemplate produces a SINGLE STRING. There are no roles (no system/human).
# The model receives the whole thing as one undifferentiated blob of text.
#
# ChatPromptTemplate (covered in 03.2) produces a LIST OF MESSAGES with roles.
# For modern chat models like gpt-4o-mini, ChatPromptTemplate is almost always
# what you want. PromptTemplate is useful for simple string formatting or as a
# building block inside SystemMessagePromptTemplate / HumanMessagePromptTemplate.

# ── Reusability example ────────────────────────────────────────────────────────────
# The same template can be filled with different values — that's the point.
print("\n--- Reusability: same template, different values ---")
audiences = ["teenagers", "senior citizens", "professional athletes"]
for audience in audiences:
    filled = template.format(product="smart water bottle", audience=audience)
    print(f"\nAudience: {audience}")
    print(llm.invoke(filled).content)
