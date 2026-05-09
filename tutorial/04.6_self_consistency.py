"""
04.6 - Self-Consistency Prompting
===================================
Concept: Self-Consistency runs the SAME Chain-of-Thought prompt multiple times
at a higher temperature (more randomness), then picks the answer that appears
most often across all runs — the majority vote.

Why this works:
  - At temperature > 0, the model takes slightly different reasoning paths each run
  - Most paths lead to the correct answer; only some lead to errors
  - The correct answer tends to win the majority vote even if the model occasionally
    makes a mistake on one run
  - It's like asking 5 people to solve a problem independently — if 4 agree, trust them

Cost vs accuracy trade-off:
  Standard CoT      → 1 LLM call, faster, cheaper
  Self-Consistency  → N LLM calls (typically 3–5), slower, more expensive, more accurate

When to use:
  - High-stakes reasoning tasks where accuracy matters more than cost
  - When CoT alone gives inconsistent answers across runs
  - Maths, logic, multi-step reasoning where errors can compound

When NOT worth it:
  - Simple tasks where CoT already gives 100% accuracy
  - Latency-sensitive applications
  - Tasks with no clear "correct" answer (creative writing, opinions)

Self-Consistency was introduced in "Self-Consistency Improves Chain of Thought
Reasoning in Language Models" (Wang et al. 2022).
"""

import re
from collections import Counter

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

# Use higher temperature so each run takes a different reasoning path.
# temperature=0 would give the same answer every time — no diversity, no voting.
cot_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# ── The classic bat-and-ball problem ───────────────────────────────────────────
# This problem tricks most people (and LLMs at temperature=0) into saying $0.10.
# The correct answer is $0.05.
# Intuitive (wrong): ball = $0.10, bat = $1.10, total = $1.20 ✗
# Correct:           ball = $0.05, bat = $1.05, total = $1.10 ✓
question = (
    "A bat and a ball cost $1.10 in total. "
    "The bat costs $1.00 more than the ball. "
    "How much does the ball cost? "
    "Think step by step, then state your final answer clearly."
)

print("=== Self-Consistency: bat-and-ball problem ===")
print(f"Question: {question}\n")

NUM_RUNS = 3  # in production, use 5–10 for better reliability
answers = []

for i in range(NUM_RUNS):
    response = cot_llm.invoke(question).content
    answers.append(response)
    print(f"--- Run {i + 1} ---")
    print(response)
    print()

# ── Majority vote ───────────────────────────────────────────────────────────────────
# Parse each answer to extract the final dollar amount.
# In production you'd use an LLM or more robust parser; regex works for demos.
def extract_cents(text: str) -> str:
    """Extract a dollar amount like '$0.05' or '5 cents' from the response."""
    match = re.search(r'\$0\.\d+|\d+\s*cents?', text.lower())
    return match.group(0) if match else "unclear"

extracted = [extract_cents(a) for a in answers]
vote_counts = Counter(extracted)
majority = vote_counts.most_common(1)[0][0]

print("=== Majority vote result ===")
for answer, count in vote_counts.items():
    print(f"  '{answer}' → {count}/{NUM_RUNS} runs")
print(f"\nFinal answer (majority): {majority}")
print("Correct answer: $0.05")

# ── A second example: multi-step arithmetic ───────────────────────────────────────
print("\n\n=== Self-Consistency: multi-step arithmetic ===")
arithmetic_q = (
    "A factory produces 240 widgets per day. "
    "It operates 5 days a week. "
    "How many widgets does it produce in 4 weeks? "
    "Think step by step."
)
print(f"Question: {arithmetic_q}\n")

arithmetic_answers = []
for i in range(NUM_RUNS):
    response = cot_llm.invoke(arithmetic_q).content
    arithmetic_answers.append(response)
    # Extract just the final number
    nums = re.findall(r'\b4[,\s]?800\b|\b4800\b', response)
    print(f"Run {i + 1}: {'✓ 4800' if nums else '? ' + response[-100:]}")

# ── Key insight ───────────────────────────────────────────────────────────────────
# Self-Consistency is a meta-strategy that sits on top of CoT.
# You're not changing the prompt — you're running it multiple times and aggregating.
# The more runs, the more reliable the answer, at the cost of more API calls.
#
# In practice, 3 runs gives most of the benefit; 5+ is for critical applications.
