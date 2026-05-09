"""
04.5 - Chain-of-Thought (CoT) Prompting
=========================================
Concept: Chain-of-Thought prompting asks the model to REASON STEP BY STEP before
giving its final answer. Instead of jumping to the answer, it shows its working —
like a student writing out each step of a maths problem.

Why it works:
  - LLMs generate tokens one at a time. When the model writes out reasoning steps,
    those intermediate tokens INFORM the next tokens — effectively letting the model
    "think on paper" before committing to an answer.
  - Without CoT, the model must compress all reasoning into a single output token.
    With CoT, it spreads reasoning across many tokens, which is much easier for it.

The magic phrase: "Let's think step by step."
  - This simple instruction reliably activates step-by-step reasoning
  - Discovered in the paper "Chain-of-Thought Prompting Elicits Reasoning in LLMs" (Wei et al. 2022)

When to use:
  - Maths word problems
  - Multi-step logical reasoning
  - Tasks where the answer depends on several intermediate conclusions
  - Any time the model gives wrong answers on reasoning tasks

When NOT needed:
  - Simple factual lookups ("What is the capital of France?")
  - Creative tasks (writing, summarisation)
  - Direct format conversions
"""

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ── Example 1: maths word problem ───────────────────────────────────────────────
print("=== CoT: maths word problem ===")

# WITHOUT CoT — model may jump to the wrong answer
no_cot = ChatPromptTemplate.from_messages([
    ("system", "You are a maths assistant. Give only the final numeric answer."),
    ("human",
     "A store sells apples for $1.20 each and oranges for $0.80 each. "
     "Sarah buys 5 apples and 3 oranges. How much does she spend in total?"),
])
print("Without CoT:", (no_cot | llm).invoke({}).content)

# WITH CoT — model reasons through each step before the answer
cot_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a logical reasoning assistant."),
    ("human",
     "A store sells apples for $1.20 each and oranges for $0.80 each. "
     "Sarah buys 5 apples and 3 oranges. How much does she spend in total?\n\n"
     "Let's think step by step."),   # ← the CoT trigger
])
print("\nWith CoT:")
print((cot_prompt | llm).invoke({}).content)
# The model will write:
#   Step 1: 5 apples × $1.20 = $6.00
#   Step 2: 3 oranges × $0.80 = $2.40
#   Step 3: $6.00 + $2.40 = $8.40

# ── Example 2: logical reasoning ────────────────────────────────────────────────
print("\n=== CoT: logical reasoning ===")
logic_cot = ChatPromptTemplate.from_messages([
    ("system", "You are a logical reasoning assistant."),
    ("human",
     "All mammals are warm-blooded. Dolphins are mammals. "
     "Snakes are not mammals. Are dolphins warm-blooded? Are snakes warm-blooded?\n\n"
     "Think through each step before answering."),
])
print((logic_cot | llm).invoke({}).content)

# ── Example 3: few-shot CoT (showing reasoning examples) ──────────────────────
# You can combine few-shot with CoT by providing examples that SHOW the reasoning.
# This is especially powerful — the model learns both the format AND how to reason.
print("\n=== Few-shot CoT: showing reasoning in examples ===")
few_shot_cot = ChatPromptTemplate.from_messages([
    ("system", "Solve the problem by showing your reasoning, then give the final answer."),
    ("human",  "I have 3 boxes. Each box has 4 balls. How many balls in total?"),
    ("ai",     "Each box has 4 balls. There are 3 boxes. So total = 3 × 4 = 12 balls. Answer: 12"),
    ("human",  "A train travels 60 km/h for 2.5 hours. How far does it travel?"),
    ("ai",     "Distance = speed × time = 60 × 2.5 = 150 km. Answer: 150 km"),
    ("human",  "{problem}"),
])
chain = few_shot_cot | llm
print(chain.invoke({
    "problem": "A shop has 48 items. It sells 1/3 of them in the morning and 1/4 of the remainder in the afternoon. How many items are left?"
}).content)

# ── Key insight ───────────────────────────────────────────────────────────────────
# CoT is "free" — it costs no extra API calls, just a longer prompt.
# For hard reasoning tasks it can double or triple accuracy.
# The next file (04.6) builds on CoT with Self-Consistency to go even further.
