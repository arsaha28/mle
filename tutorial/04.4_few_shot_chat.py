"""
04.4 - Few-Shot Chat Prompting (FewShotChatMessagePromptTemplate)
=================================================================
Concept: FewShotChatMessagePromptTemplate is the chat-model equivalent of
FewShotPromptTemplate. Instead of formatting examples as plain text blocks,
it formats each example as a pair of Human/AI messages.

Why use this instead of FewShotPromptTemplate?
  - Chat models (gpt-4o, Claude, etc.) are trained on message pairs, not raw text
  - Human/AI message pairs align with how the model was fine-tuned
  - Feels more natural for conversational or instruction-following tasks
  - The system message stays separate from the examples — cleaner structure

Structure:
  FewShotChatMessagePromptTemplate expands into a flat list of message pairs:
    HumanMessage("2 + 2")  AIMessage("4")
    HumanMessage("10 - 3") AIMessage("7")
    HumanMessage("5 * 6")  AIMessage("30")
  Then the final ChatPromptTemplate wraps it with a system message and the
  real query at the end.

FewShotPromptTemplate  → plain text blocks  → good for completion-style models
FewShotChatMessagePromptTemplate → message pairs → good for chat models
"""

from dotenv import load_dotenv
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ── Step 1: Define examples as input/output pairs ─────────────────────────────
# 'input' becomes a HumanMessage, 'output' becomes an AIMessage.
examples = [
    {"input": "2 + 2",  "output": "4"},
    {"input": "10 - 3", "output": "7"},
    {"input": "5 * 6",  "output": "30"},
]

# ── Step 2: Define the per-example message template ──────────────────────────
# This template maps each example dict to a Human/AI message pair.
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai",    "{output}"),
])

# ── Step 3: Create the few-shot chat prompt ─────────────────────────────────
few_shot_chat = FewShotChatMessagePromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
)

# ── Step 4: Wrap in a full ChatPromptTemplate ────────────────────────────────
# The few_shot_chat block expands into all the Human/AI message pairs.
# Add a system message before it and the real query after it.
final_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a calculator. Only reply with the numeric result."),
    few_shot_chat,   # ← expands to 3 Human/AI pairs
    ("human", "{input}"),  # ← the real question
])

chain = final_prompt | llm

print("=== Few-shot Chat: calculator ===")
for question in ["8 * 9", "100 / 4", "15 + 27"]:
    result = chain.invoke({"input": question}).content
    print(f"  {question} = {result}")

# ── Inspecting the expanded message list ───────────────────────────────────────
print("\n=== Messages sent to the LLM for '8 * 9' ===")
filled = final_prompt.format_messages(input="8 * 9")
for msg in filled:
    print(f"  [{msg.__class__.__name__:<15}] {msg.content}")
# You'll see:
#   [SystemMessage    ] You are a calculator. Only reply with the numeric result.
#   [HumanMessage     ] 2 + 2
#   [AIMessage        ] 4
#   [HumanMessage     ] 10 - 3
#   [AIMessage        ] 7
#   [HumanMessage     ] 5 * 6
#   [AIMessage        ] 30
#   [HumanMessage     ] 8 * 9   ← the real question

# ── Another example: tone rewriting ──────────────────────────────────────────
print("\n=== Few-shot Chat: tone rewriter ===")
tone_examples = [
    {"input": "Fix this now.",             "output": "Could you please address this when you get a chance?"},
    {"input": "That's wrong.",             "output": "I think there might be a different way to look at this."},
    {"input": "I need this by tomorrow.",  "output": "Would it be possible to have this ready by tomorrow?"},
]
tone_example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai",    "{output}"),
])
tone_few_shot = FewShotChatMessagePromptTemplate(examples=tone_examples, example_prompt=tone_example_prompt)
tone_prompt = ChatPromptTemplate.from_messages([
    ("system", "Rewrite the message in a polite, professional tone."),
    tone_few_shot,
    ("human", "{input}"),
])
tone_chain = tone_prompt | llm
print(tone_chain.invoke({"input": "Stop sending me these reports."}).content)
