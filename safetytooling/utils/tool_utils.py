import json
import logging
from typing import Any, Dict, List, Optional, Union

from langchain.tools import BaseTool


def make_tools_hashable(tools: list[BaseTool]) -> Optional[tuple[str, ...]]:
    if not tools:
        return None

    tool_signatures = []
    for tool in tools:
        # Get the key identifying information
        signature = {
            "name": tool.name,
            "description": tool.description,
            "args_schema": tool.args_schema.schema() if tool.args_schema else None,
        }
        tool_signatures.append(json.dumps(signature, sort_keys=True))

    return tuple(tool_signatures)


def convert_tool_to_anthropic(tool: BaseTool) -> Dict[str, Any]:
    """Convert a single LangChain tool to Anthropic tool format."""

    # Get the input schema
    input_schema = {"type": "object", "properties": {}, "required": []}

    if tool.args_schema:
        try:
            schema = tool.args_schema.schema()
            input_schema = {
                "type": "object",
                "properties": schema.get("properties", {}),
                "required": schema.get("required", []),
            }
        except Exception as e:
            logging.warning(f"Failed to get schema for tool {tool.name}: {e}")

    return {"name": tool.name, "description": tool.description, "input_schema": input_schema}


def convert_tools_to_anthropic(tools: List[BaseTool]) -> List[Dict[str, Any]]:
    """Convert a list of LangChain tools to Anthropic tool format."""
    if not tools:
        return []

    anthropic_tools = []
    for tool in tools:
        try:
            anthropic_tool = convert_tool_to_anthropic(tool)
            anthropic_tools.append(anthropic_tool)
        except Exception as e:
            logging.warning(f"Failed to convert tool {getattr(tool, 'name', 'unknown')}: {e}")
            continue

    return anthropic_tools


def convert_tool_to_openai(tool: BaseTool) -> Dict[str, Any]:
    """Convert a single LangChain tool to OpenAI tool format."""

    # Get the parameters schema
    parameters = {"type": "object", "properties": {}, "required": []}

    if tool.args_schema:
        try:
            schema = tool.args_schema.schema()
            parameters = {
                "type": "object",
                "properties": schema.get("properties", {}),
                "required": schema.get("required", []),
            }
        except Exception as e:
            logging.warning(f"Failed to get schema for tool {tool.name}: {e}")

    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": parameters,
        },
    }


def convert_tools_to_openai(tools: List[BaseTool]) -> List[Dict[str, Any]]:
    """Convert a list of LangChain tools to OpenAI tool format."""
    if not tools:
        return []

    openai_tools = []
    for tool in tools:
        try:
            openai_tool = convert_tool_to_openai(tool)
            openai_tools.append(openai_tool)
        except Exception as e:
            logging.warning(f"Failed to convert tool {getattr(tool, 'name', 'unknown')}: {e}")
            continue

    return openai_tools