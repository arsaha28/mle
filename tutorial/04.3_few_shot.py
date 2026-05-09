"""
04.3 - Few-Shot Prompting (FewShotPromptTemplate)
==================================================
Concept: Few-shot means providing MULTIPLE examples (typically 3–5) so the model
can generalise the pattern across different cases, not just mirror one example.

Why more examples help:
  - The model sees the task from multiple angles — positive, negative, edge cases
  - It learns the PATTERN, not just a single format
  - Handles ambiguous inputs better (e.g. mixed-sentiment reviews)

FewShotPromptTemplate is LangChain's structured way to:
  1. Define a list of examples as dicts
  2. Define how each example should be formatted
  3. Wrap them with a prefix (instructions) and suffix (the new query)

When to use over one-shot:
  - Task has nuanced distinctions (e.g. Neutral vs slightly Negative)
  - You want to cover multiple edge cases
  - Output quality matters more than prompt length

Structure of FewShotPromptTemplate:
  prefix        → instructions shown before all examples
  example_prompt → template applied to EACH example dict
  examples       → the list of example dicts
  suffix        → the new query appended after all examples
  input_variables → variables expected in the suffix
"""

from dotenv import load_dotenv
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ── Step 1: Define examples as a list of dicts ─────────────────────────────────
# Each dict must contain all keys referenced in the example_prompt template.
examples = [
    {"review": "Absolutely love this product!", "sentiment": "Positive"},
    {"review": "Worst purchase I've ever made.", "sentiment": "Negative"},
    {"review": "It arrived on time and works fine.", "sentiment": "Neutral"},
    {"review": "Good quality but a bit overpriced.", "sentiment": "Neutral"},
    {"review": "Broke after two days — total waste of money.", "sentiment": "Negative"},
]

# ── Step 2: Define how each individual example is formatted ───────────────────
example_prompt = PromptTemplate(
    input_variables=["review", "sentiment"],
    template="Review: {review}\nSentiment: {sentiment}",
)

# ── Step 3: Assemble into FewShotPromptTemplate ───────────────────────────────
few_shot = FewShotPromptTemplate(
    examples=examples,           # the list of example dicts
    example_prompt=example_prompt,  # how to format each example
    prefix="Classify the sentiment of each review as Positive, Negative, or Neutral.",
    suffix="Review: {review}\nSentiment:",  # the new query — model completes "Sentiment:"
    input_variables=["review"],  # variables in the suffix
)

chain = few_shot | llm

print("=== Few-shot sentiment classification ===")
test_reviews = [
    "Decent product, nothing special.",
    "Amazing build quality, fast shipping, very happy!",
    "Works as advertised but the instructions were confusing.",
]
for review in test_reviews:
    result = chain.invoke({"review": review}).content
    print(f"\nReview: {review}")
    print(f"Sentiment: {result}")

# ── Inspecting what the template produces before sending to the LLM ───────────
# This shows you the full prompt the model receives — useful for debugging.
print("\n=== Full prompt sent to the LLM ===")
print(few_shot.format(review="Decent product, nothing special."))
# You'll see the prefix, then all 5 formatted examples, then the new review.

# ── Key insight ───────────────────────────────────────────────────────────────────
# The model never sees "the rule" — it infers it from the examples.
# This makes few-shot prompting powerful for tasks that are hard to describe
# explicitly but easy to demonstrate with examples.
