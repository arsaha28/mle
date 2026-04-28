"""
07 - Output Parsers
===================
Concept: Transform LLM text output into structured Python objects.

Parsers covered:
  - StrOutputParser
  - CommaSeparatedListOutputParser
  - JsonOutputParser
  - PydanticOutputParser
  - DatetimeOutputParser
  - XMLOutputParser
"""

from dotenv import load_dotenv
from langchain_core.output_parsers import CommaSeparatedListOutputParser, JsonOutputParser, StrOutputParser, XMLOutputParser, PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.output_parsers import DatetimeOutputParser
from pydantic import BaseModel, Field

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

print("=== 1. StrOutputParser ===")
result = (ChatPromptTemplate.from_messages([("human", "What is LangChain in one sentence?")]) | llm | StrOutputParser()).invoke({})
print(type(result), "->", result)

print("\n=== 2. CommaSeparatedListOutputParser ===")
list_parser = CommaSeparatedListOutputParser()
list_prompt = PromptTemplate(template="List 5 programming languages. {format_instructions}", input_variables=[], partial_variables={"format_instructions": list_parser.get_format_instructions()})
result = (list_prompt | llm | list_parser).invoke({})
print(type(result), "->", result)

print("\n=== 3. JsonOutputParser ===")
json_parser = JsonOutputParser()
json_prompt = ChatPromptTemplate.from_messages([
    ("system", "Always respond with valid JSON only."),
    ("human", "Return JSON with keys 'name', 'founded_year', 'ceo' for: {company}"),
])
result = (json_prompt | llm | json_parser).invoke({"company": "OpenAI"})
print(type(result), "->", result)

print("\n=== 4. PydanticOutputParser ===")
class MovieReview(BaseModel):
    title: str = Field(description="Title of the movie")
    year: int = Field(description="Release year")
    rating: float = Field(description="Rating out of 10")
    summary: str = Field(description="One-sentence summary")

pydantic_parser = PydanticOutputParser(pydantic_object=MovieReview)
pydantic_prompt = PromptTemplate(template="Provide info about '{movie}'.\n{format_instructions}", input_variables=["movie"], partial_variables={"format_instructions": pydantic_parser.get_format_instructions()})
result = (pydantic_prompt | llm | pydantic_parser).invoke({"movie": "Inception"})
print(f"Title: {result.title}, Year: {result.year}, Rating: {result.rating}")

print("\n=== 5. DatetimeOutputParser ===")
datetime_parser = DatetimeOutputParser()
datetime_prompt = PromptTemplate(template="When was the first iPhone announced? {format_instructions}", input_variables=[], partial_variables={"format_instructions": datetime_parser.get_format_instructions()})
result = (datetime_prompt | llm | datetime_parser).invoke({})
print(type(result), "->", result)

print("\n=== 6. XMLOutputParser ===")
xml_parser = XMLOutputParser(tags=["person", "name", "age", "city"])
xml_prompt = ChatPromptTemplate.from_messages([
    ("system", "Respond only with valid XML. No markdown."),
    ("human", "Create <person> with <name>, <age>, <city> for a character named {name}."),
])
result = (xml_prompt | llm | xml_parser).invoke({"name": "Elena"})
print(type(result), "->", result)
