"""
15 - Tools
==========
Concept: Tools let an LLM take actions. The model decides when to call
a tool and what arguments to pass.

Patterns covered:
  1. @tool decorator
  2. StructuredTool
  3. BaseTool subclass
  4. Pydantic args schema
  5. Binding tools to an LLM
  6. Manually executing tool calls
"""

import re
from datetime import date
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool, tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 1. @tool decorator
print("=== 1. @tool decorator ===")

@tool
def calculate(expression: str) -> str:
    """Evaluate a safe math expression. Supports +, -, *, /, **, parentheses."""
    clean = re.sub(r"[^\d\s\+\-\*\/\(\\.]", "", expression)
    try:
        return str(round(eval(clean), 6))
    except Exception as e:
        return f"Error: {e}"

@tool
def get_today() -> str:
    """Return today's date in YYYY-MM-DD format."""
    return date.today().isoformat()

print(f"Tool name: {calculate.name}")
print(f"calculate('10 * 7') = {calculate.invoke('10 * 7')}")
print(f"get_today() = {get_today.invoke({})}")

# 2. StructuredTool
print("\n=== 2. StructuredTool ===")

def word_stats(text: str, include_unique: bool = False) -> dict:
    words = text.lower().split()
    result = {"word_count": len(words), "char_count": len(text)}
    if include_unique:
        result["unique_words"] = len(set(words))
    return result

word_stats_tool = StructuredTool.from_function(
    func=word_stats, name="word_stats",
    description="Count words and characters in text.",
)
print(word_stats_tool.invoke({"text": "the cat sat on the mat", "include_unique": True}))

# 3. Pydantic args schema
print("\n=== 3. Pydantic args schema ===")

class UnitConversionInput(BaseModel):
    value: float = Field(description="The numeric value to convert")
    from_unit: str = Field(description="Source unit, e.g. 'km', 'celsius'")
    to_unit: str = Field(description="Target unit, e.g. 'miles', 'fahrenheit'")

@tool(args_schema=UnitConversionInput)
def convert_unit(value: float, from_unit: str, to_unit: str) -> str:
    """Convert a value between common units of measurement."""
    conversions = {
        ("km", "miles"): lambda v: v * 0.621371,
        ("miles", "km"): lambda v: v / 0.621371,
        ("celsius", "fahrenheit"): lambda v: v * 9 / 5 + 32,
        ("fahrenheit", "celsius"): lambda v: (v - 32) * 5 / 9,
    }
    key = (from_unit.lower(), to_unit.lower())
    if key not in conversions:
        return f"Conversion from {from_unit} to {to_unit} not supported."
    return f"{value} {from_unit} = {conversions[key](value):.4f} {to_unit}"

print(convert_unit.invoke({"value": 100, "from_unit": "km", "to_unit": "miles"}))
print(convert_unit.invoke({"value": 37, "from_unit": "celsius", "to_unit": "fahrenheit"}))

# 4. BaseTool subclass
print("\n=== 4. BaseTool subclass ===")

class TextReverserTool(BaseTool):
    name: str = "text_reverser"
    description: str = "Reverse the characters in a string."
    def _run(self, text: str) -> str:
        return text[::-1]
    async def _arun(self, text: str) -> str:
        return self._run(text)

print(TextReverserTool().invoke("Hello, World!"))

# 5. Binding tools to an LLM
print("\n=== 5. Binding tools to an LLM ===")
tools = [calculate, get_today, convert_unit, word_stats_tool]
llm_with_tools = llm.bind_tools(tools)
response = llm_with_tools.invoke([HumanMessage("What is 15% of 240, and what is today's date?")])
print("Tool calls requested:", response.tool_calls)

# 6. Manually executing tool calls
print("\n=== 6. Executing tool calls ===")
tool_map = {t.name: t for t in tools}
messages = [HumanMessage("Convert 5 km to miles and tell me today's date.")]
ai_msg = llm_with_tools.invoke(messages)
messages.append(ai_msg)
for call in ai_msg.tool_calls:
    result = tool_map[call["name"]].invoke(call["args"])
    messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
    print(f"  Tool '{call['name']}' -> {result}")
print(f"\nFinal answer: {llm_with_tools.invoke(messages).content}")
