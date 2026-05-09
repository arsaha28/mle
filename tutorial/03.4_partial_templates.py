"""
03.4 - Partial Templates
=========================
Concept: A partial template is a template where some variables are pre-filled
(locked in) at definition time, leaving the remaining variables to be supplied
at call time.

Think of it as: "I know the ROLE right now, but the QUESTION will come later."

Why this matters:
  - Write one base template, derive multiple specialisations from it
  - Each specialisation hides the fixed variable from callers — they only see
    what's still open
  - No duplication: the prompt structure is defined once
  - Clean separation: the part of your code that knows the role (e.g. app config)
    is separate from the part that knows the question (e.g. user input handler)

Analogy: partial templates are like partial function application (functools.partial)
         but for prompts.
"""

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ── Base template with two variables ─────────────────────────────────────────────
# Both {role} and {question} are open at this point.
base_template = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}."),
    ("human", "Answer this question: {question}"),
])

# ── Creating specialisations with .partial() ─────────────────────────────────
# .partial() locks in the 'role' variable permanently.
# The returned template only needs 'question' — role is invisible to callers.
doctor_template  = base_template.partial(role="medical professional")
lawyer_template  = base_template.partial(role="legal expert")
teacher_template = base_template.partial(role="primary school teacher")

# Each specialised template becomes its own chain.
doctor_chain  = doctor_template  | llm
lawyer_chain  = lawyer_template  | llm
teacher_chain = teacher_template | llm

# Callers only supply {question} — they don't need to know about {role} at all.
question = "What are the most common causes of fatigue?"

print("=== Partial Templates: same question, three different roles ===")

print("\n--- Medical Professional ---")
print(doctor_chain.invoke({"question": question}).content)

print("\n--- Legal Expert ---")
print(lawyer_chain.invoke({"question": "What happens if I miss a court date?"}).content)

print("\n--- Primary School Teacher ---")
print(teacher_chain.invoke({"question": "Why is the sky blue?"}).content)

# ── Without partial — what you'd have to do instead ──────────────────────────
# Option A: duplicate the template three times (brittle, repetitive)
# Option B: always pass both role and question everywhere (couples callers to internals)
#
# With partial, you define once and fork cleanly. The calling code is simpler
# and the role is set in one place only.

# ── Inspecting what's left open ───────────────────────────────────────────────
print("\n--- Inspecting open variables after partial ---")
print("Base template input variables:   ", base_template.input_variables)
print("Doctor template input variables: ", doctor_template.input_variables)
# base:   ['role', 'question']
# doctor: ['question']   ← role is already filled
