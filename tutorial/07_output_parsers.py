"""
07 - Output Parsers
===================
Concept: LLMs return text. Output parsers transform that text into structured
Python objects — strings, lists, or dicts.

Parsers covered:
  - StrOutputParser                → plain string (most common)
  - CommaSeparatedListOutputParser → ["item1", "item2", ...]
  - JsonOutputParser               → dict / list from JSON
"""

from dotenv import load_dotenv
from langchain_core.output_parsers import (
    CommaSeparatedListOutputParser,
    JsonOutputParser,
    StrOutputParser,
)
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ── 1. StrOutputParser ──────────────────────────────────────────────────────────────
print("=== 1. StrOutputParser ===")
chain = (
    ChatPromptTemplate.from_messages([("human", "What is LangChain in one sentence?")])
    | llm
    | StrOutputParser()
)
result = chain.invoke({})
print(type(result), "→", result)


# ── 2. CommaSeparatedListOutputParser ─────────────────────────────────────────────
print("\n=== 2. CommaSeparatedListOutputParser ===")
list_parser = CommaSeparatedListOutputParser()
list_prompt = PromptTemplate(
    template="List 5 programming languages. {format_instructions}",
    input_variables=[],
    partial_variables={"format_instructions": list_parser.get_format_instructions()},
)
list_chain = list_prompt | llm | list_parser
result = list_chain.invoke({})
print(type(result), "→", result)
print("First item:", result[0])


# ── 3. JsonOutputParser ──────────────────────────────────────────────────────────────
print("\n=== 3. JsonOutputParser ===")
json_parser = JsonOutputParser()
json_prompt = ChatPromptTemplate.from_messages([
    ("system", "Always respond with valid JSON only. No markdown, no extra text."),
    ("human",
     "Return a JSON object with keys 'name', 'founded_year', and 'ceo' for the company: {company}"),
])
json_chain = json_prompt | llm | json_parser
result = json_chain.invoke({"company": "OpenAI"})
print(type(result), "→", result)
print("CEO:", result.get("ceo"))
