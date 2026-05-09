"""
04.2 - One-Shot Prompting
==========================
Concept: One-shot means providing ONE example of the expected input→output pair
inside the prompt. The model sees the example and mirrors that exact format.

Why one example helps:
  - It shows the model the output FORMAT you want (one word? a sentence? JSON?)
  - It removes ambiguity about what "classify sentiment" means to you
  - It's the minimum needed to lock in consistent output structure

When to use:
  - When zero-shot gives inconsistent formats
  - When the desired output format isn't obvious from the task description alone
  - When you only have one good example

Difference from few-shot (04.3):
  One-shot  → 1 example. Quick, low overhead.
  Few-shot  → 3+ examples. Better for nuanced tasks, teaches patterns.
"""

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ── One-shot: the example is embedded directly in the prompt string ────────────
# The format is: show one complete example, then present the new input.
# The model learns from the example what format to follow.
one_shot = PromptTemplate.from_template(
    "Classify sentiment as Positive, Negative, or Neutral. Reply with just one word.\n\n"
    "Review: Absolutely love this product!\n"   # ← the example input
    "Sentiment: Positive\n\n"                   # ← the example output
    "Review: {review}\n"                        # ← the real input (variable)
    "Sentiment:"                                # ← model completes this
)
chain = one_shot | llm

print("=== One-shot sentiment classification ===")
reviews = [
    "Terrible experience, would not recommend.",
    "Decent product, nothing special.",
    "Exceeded all my expectations — truly outstanding!",
]
for review in reviews:
    result = chain.invoke({"review": review}).content
    print(f"\nReview: {review}")
    print(f"Sentiment: {result}")

# ── Compare: zero-shot vs one-shot output ──────────────────────────────────
print("\n=== Zero-shot vs One-shot — format comparison ===")

zero_shot = PromptTemplate.from_template(
    "Classify the sentiment of this review as Positive, Negative, or Neutral.\n\nReview: {review}"
)
test_review = "Decent product, nothing special."

zero_result = (zero_shot | llm).invoke({"review": test_review}).content
one_result  = chain.invoke({"review": test_review}).content

print(f"Zero-shot output: '{zero_result}'")
print(f"One-shot output:  '{one_result}'")
# Zero-shot may return a full sentence like "The sentiment is Neutral."
# One-shot returns exactly one word: "Neutral" — because that's what the example shows.

# ── Key insight ───────────────────────────────────────────────────────────────────
# The example in a one-shot prompt teaches FORMAT, not just the task.
# If your example output is "Neutral (3/5 stars)", the model will follow that format too.
# The model is pattern-matching against your example.
