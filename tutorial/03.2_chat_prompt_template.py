"""
03.2 - ChatPromptTemplate (Multi-Message Template)
===================================================
Concept: ChatPromptTemplate defines a list of role-based messages as a template.
Each message has a role (system, human, ai) and can contain {variables}.
At runtime, variables are filled in and the result is a list of typed messages.

When to use:
  - Working with modern chat models (gpt-4o, gpt-4o-mini, Claude, etc.)
  - Whenever you need a system message to set behaviour
  - This is the standard template type — use it almost always

Output type: list of messages → [SystemMessage(...), HumanMessage(...), ...]

Key difference from PromptTemplate:
  PromptTemplate    → one plain string, no roles
  ChatPromptTemplate → structured list of messages, each with a role
"""

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ── Basic usage ─────────────────────────────────────────────────────────────────
# from_messages() takes a list of (role, content) tuples.
# "system" → SystemMessage, "human" → HumanMessage, "ai" → AIMessage
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are an expert {domain} tutor. Explain concepts simply."),
    ("human", "Explain {concept} in one sentence."),
])

# The | operator chains the template directly into the LLM — this is LCEL.
# The template fills in variables and passes the message list to the model.
chain = chat_template | llm
print("=== ChatPromptTemplate basic usage ===")
print(chain.invoke({"domain": "machine learning", "concept": "gradient descent"}).content)

# ── What the template actually produces ───────────────────────────────────────
# Before invoking the LLM, you can inspect what the template generates.
# This is useful for debugging — see exactly what the model receives.
print("\n--- Inspecting the filled messages ---")
filled_messages = chat_template.format_messages(
    domain="machine learning",
    concept="gradient descent"
)
for msg in filled_messages:
    print(f"[{msg.__class__.__name__}] {msg.content}")
# → [SystemMessage] You are an expert machine learning tutor. Explain concepts simply.
# → [HumanMessage]  Explain gradient descent in one sentence.

# ── Reusability: same template, different domains ─────────────────────────────
print("\n--- Same template, different domains ---")
topics = [
    {"domain": "mathematics", "concept": "derivatives"},
    {"domain": "physics",     "concept": "entropy"},
    {"domain": "economics",   "concept": "supply and demand"},
]
for t in topics:
    print(f"\n{t['domain'].title()} — {t['concept']}:")
    print(chain.invoke(t).content)
