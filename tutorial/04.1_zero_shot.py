"""
04.1 - Zero-Shot Prompting
===========================
Concept: Zero-shot means giving the model ONLY an instruction — no examples.
You describe the task and trust the model to figure out the format and approach
from its training knowledge alone.

When to use:
  - Quick, simple tasks where output format doesn't need to be exact
  - When you don't have examples to provide
  - As a baseline before adding examples to improve quality

Limitation:
  - The model may interpret the task differently than you expect
  - Output format can vary — sometimes verbose, sometimes terse
  - Adding even one example (one-shot, 04.2) often improves consistency

The name "zero-shot" comes from machine learning: the model handles the task
with zero training examples shown at inference time.
"""

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ── Basic zero-shot ─────────────────────────────────────────────────────────────────
# Just a task description. No examples of what the output should look like.
zero_shot = PromptTemplate.from_template(
    "Classify the sentiment of this review as Positive, Negative, or Neutral.\n\nReview: {review}"
)
chain = zero_shot | llm

print("=== Zero-shot sentiment classification ===")
reviews = [
    "The battery life is amazing but the camera is disappointing.",
    "This is the best laptop I have ever owned.",
    "It arrived on time. Does what it says on the box.",
]
for review in reviews:
    result = chain.invoke({"review": review}).content
    print(f"\nReview: {review}")
    print(f"Result: {result}")

# ── Observation: format can be inconsistent ──────────────────────────────────────
# Notice the model sometimes returns just "Negative", sometimes a full sentence.
# That inconsistency is the main drawback of zero-shot.
# One-shot (04.2) fixes this by showing the exact format you want.

# ── Zero-shot with a chat template ─────────────────────────────────────────
# Using ChatPromptTemplate gives you a system message to anchor the persona.
print("\n=== Zero-shot with system message ===")
chat_zero_shot = ChatPromptTemplate.from_messages([
    ("system", "You are a sentiment analysis expert. Reply with exactly one word: Positive, Negative, or Neutral."),
    ("human", "Review: {review}"),
])
chat_chain = chat_zero_shot | llm

for review in reviews:
    result = chat_chain.invoke({"review": review}).content
    print(f"{review[:50]:<52} → {result}")
# Adding a system message with explicit format instructions already
# improves consistency without needing any examples.
