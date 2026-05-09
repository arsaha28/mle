"""
03 - Prompt Templates
=====================
Concept: Hard-coding prompts is brittle. Templates let you define a prompt
once with variables ({like_this}) and fill them in at runtime.

Types covered:
  - PromptTemplate
  - ChatPromptTemplate
  - SystemMessagePromptTemplate / HumanMessagePromptTemplate
  - Partial templates
  - from_template shorthand
"""

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, PromptTemplate, SystemMessagePromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

print("=== PromptTemplate ===")
template = PromptTemplate(
    input_variables=["product", "audience"],
    template="Write a one-sentence product description for {product} aimed at {audience}.",
)
filled = template.format(product="smart water bottle", audience="fitness enthusiasts")
print("Filled prompt:", filled)
print("LLM response:", llm.invoke(filled).content)

print("\n=== ChatPromptTemplate ===")
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are an expert {domain} tutor. Explain concepts simply."),
    ("human", "Explain {concept} in one sentence."),
])
print((chat_template | llm).invoke({"domain": "machine learning", "concept": "gradient descent"}).content)

print("\n=== Typed message templates ===")
system_tpl = SystemMessagePromptTemplate.from_template("You are a {language} programming tutor.")
human_tpl = HumanMessagePromptTemplate.from_template("Show me a minimal example of {concept} in {language}.")
chat_template2 = ChatPromptTemplate.from_messages([system_tpl, human_tpl])
print((chat_template2 | llm).invoke({"language": "Python", "concept": "list comprehension"}).content)

print("\n=== Partial template ===")
base_template = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}."),
    ("human", "Answer this question: {question}"),
])
doctor_template = base_template.partial(role="medical professional")
print((doctor_template | llm).invoke({"question": "What are common symptoms of dehydration?"}).content)

print("\n=== from_template shorthand ===")
quick = PromptTemplate.from_template("Translate '{text}' to {language}.")
print(llm.invoke(quick.format(text="Hello, how are you?", language="Spanish")).content)
