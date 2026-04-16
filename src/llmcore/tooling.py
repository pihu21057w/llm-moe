from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Callable, Any


@dataclass(slots=True)
class ToolSpec:
    name: str
    description: str
    schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], Any] | None = None


@dataclass(slots=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, tool: ToolSpec) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def call(self, name: str, arguments: dict[str, Any]) -> Any:
        tool = self._tools[name]
        if tool.handler is None:
            raise ValueError(f"Tool {name} has no handler")
        return tool.handler(arguments)

    def as_prompt_block(self) -> str:
        lines = ["Available tools:"]
        for tool in self._tools.values():
            lines.append(f"- {tool.name}: {tool.description}")
            lines.append(json.dumps(tool.schema, sort_keys=True))
        return "\n".join(lines)


def parse_tool_call(text: str) -> ToolCall | None:
    start = text.find("<tool_call>")
    end = text.find("</tool_call>")
    if start == -1 or end == -1 or end <= start:
        return None
    payload = text[start + len("<tool_call>") : end].strip()
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None
    name = data.get("name")
    arguments = data.get("arguments", {})
    if not isinstance(name, str) or not isinstance(arguments, dict):
        return None
    return ToolCall(name=name, arguments=arguments)


def format_tool_call(name: str, arguments: dict[str, Any]) -> str:
    payload = {"name": name, "arguments": arguments}
    return f"<tool_call>{json.dumps(payload, sort_keys=True)}</tool_call>"
