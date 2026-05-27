"""
07 - Output Parsers
===================
Concept: LLMs return text. Output parsers transform that text into structured
Python objects — strings, lists, dicts, or Pydantic models.

Parsers covered:
  - StrOutputParser              → plain string (most common)
  - CommaSeparatedListOutputParser → ["item1", "item2", ...]
  - JsonOutputParser             → dict / list from JSON
  - PydanticOutputParser         → validated Pydantic model
  - Datetime parsing             → Python datetime object
  - XMLOutputParser              → parsed XML dict
"""

from datetime import datetime

from dotenv import load_dotenv
from langchain_core.output_parsers import (
    CommaSeparatedListOutputParser,
    JsonOutputParser,
    StrOutputParser,
    XMLOutputParser,
)
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

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


# ── 4. PydanticOutputParser ─────────────────────────────────────────────────────────────
print("\n=== 4. PydanticOutputParser ===")
from langchain_core.output_parsers import PydanticOutputParser  # noqa: E402

class MovieReview(BaseModel):
    title: str = Field(description="Title of the movie")
    year: int = Field(description="Release year")
    rating: float = Field(description="Rating out of 10")
    summary: str = Field(description="One-sentence summary")

pydantic_parser = PydanticOutputParser(pydantic_object=MovieReview)
pydantic_prompt = PromptTemplate(
    template="Provide information about the movie '{movie}'.\n{format_instructions}",
    input_variables=["movie"],
    partial_variables={"format_instructions": pydantic_parser.get_format_instructions()},
)
pydantic_chain = pydantic_prompt | llm | pydantic_parser
result = pydantic_chain.invoke({"movie": "Inception"})
print(type(result))
print(f"Title: {result.title}, Year: {result.year}, Rating: {result.rating}")
print(f"Summary: {result.summary}")


# ── 5. Datetime parsing ─────────────────────────────────────────────────────────────
# DatetimeOutputParser was removed from langchain.output_parsers in newer versions.
# We implement the same behaviour directly: ask the LLM for a fixed format string,
# then parse it with strptime — this is all DatetimeOutputParser did internally.
print("\n=== 5. Datetime parsing (manual) ===")

DATE_FORMAT = "%Y-%m-%dT%H:%M:%S.%f"
datetime_prompt = PromptTemplate(
    template=(
        "What date was the first iPhone announced? "
        f"Respond with ONLY a datetime string in this exact format: {DATE_FORMAT}. "
        "No other text."
    ),
    input_variables=[],
)
datetime_chain = datetime_prompt | llm | StrOutputParser()
raw = datetime_chain.invoke({})
result = datetime.strptime(raw.strip(), DATE_FORMAT)
print(type(result), "→", result)
print("Year:", result.year)


# ── 6. XMLOutputParser ──────────────────────────────────────────────────────────────
print("\n=== 6. XMLOutputParser ===")
xml_parser = XMLOutputParser(tags=["person", "name", "age", "city"])
xml_prompt = ChatPromptTemplate.from_messages([
    ("system", "Respond only with valid XML. No markdown code fences."),
    ("human",
     "Create an XML document with a <person> tag containing <name>, <age>, and <city> "
     "for a fictional character named {name}."),
])
xml_chain = xml_prompt | llm | xml_parser
result = xml_chain.invoke({"name": "Elena"})
print(type(result), "→", result)
