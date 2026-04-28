"""
04 - Prompting Strategies
=========================
Concept: *How* you phrase a prompt dramatically changes output quality.

Strategies covered:
  - Zero-shot        -> no examples, just an instruction
  - One-shot         -> one example to guide the format
  - Few-shot         -> multiple examples (FewShotPromptTemplate)
  - Few-shot chat    -> few-shot with chat messages
  - Chain-of-Thought -> ask the model to reason step-by-step
  - Self-Consistency -> run CoT multiple times, pick majority answer
"""

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate, FewShotPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

print("=== 1. Zero-shot ===")
zero_shot = PromptTemplate.from_template("Classify sentiment as Positive, Negative, or Neutral.\n\nReview: {review}")
print((zero_shot | llm).invoke({"review": "The battery life is amazing but the camera is disappointing."}).content)

print("\n=== 2. One-shot ===")
one_shot = PromptTemplate.from_template(
    "Classify sentiment. Reply with one word.\n\nReview: Absolutely love this!\nSentiment: Positive\n\nReview: {review}\nSentiment:"
)
print((one_shot | llm).invoke({"review": "Terrible experience, would not recommend."}).content)

print("\n=== 3. Few-shot (FewShotPromptTemplate) ===")
examples = [
    {"review": "Absolutely love this product!", "sentiment": "Positive"},
    {"review": "Worst purchase I've ever made.", "sentiment": "Negative"},
    {"review": "It arrived on time and works fine.", "sentiment": "Neutral"},
]
example_template = PromptTemplate(input_variables=["review", "sentiment"], template="Review: {review}\nSentiment: {sentiment}")
few_shot = FewShotPromptTemplate(
    examples=examples, example_prompt=example_template,
    prefix="Classify sentiment as Positive, Negative, or Neutral.",
    suffix="Review: {review}\nSentiment:", input_variables=["review"],
)
print((few_shot | llm).invoke({"review": "Decent product, nothing special."}).content)

print("\n=== 4. Few-shot Chat ===")
chat_examples = [{"input": "2 + 2", "output": "4"}, {"input": "10 - 3", "output": "7"}, {"input": "5 * 6", "output": "30"}]
example_chat_prompt = ChatPromptTemplate.from_messages([("human", "{input}"), ("ai", "{output}")])
few_shot_chat = FewShotChatMessagePromptTemplate(examples=chat_examples, example_prompt=example_chat_prompt)
final_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a calculator. Only reply with the numeric result."),
    few_shot_chat, ("human", "{input}"),
])
print((final_prompt | llm).invoke({"input": "8 * 9"}).content)

print("\n=== 5. Chain-of-Thought ===")
cot_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a logical reasoning assistant."),
    ("human", "A store sells apples for $1.20 and oranges for $0.80. Sarah buys 5 apples and 3 oranges. Total cost?\n\nLet's think step by step."),
])
print((cot_prompt | llm).invoke({}).content)

print("\n=== 6. Self-Consistency ===")
cot_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
question = "A bat and a ball cost $1.10. The bat costs $1.00 more than the ball. How much does the ball cost? Think step by step."
for i in range(3):
    print(f"Run {i+1}:\n{cot_llm.invoke(question).content}\n{'-'*40}")
print("\nSelf-consistency: compare the 3 answers - the majority answer is most reliable.")
