import os

from coderag.tool_registry import register_tool


@register_tool("read_file")
def read_file_tool(filepath: str) -> str:
    """Read a code file from local repo."""
    if not os.path.exists(filepath):
        return f"File not found: {filepath}"
    if os.path.isdir(filepath):
        return f"'{filepath}' is a directory."
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        return f"Content of {filepath}:\n\n{content[:2000]}..."
    except Exception as e:
        return f"Error reading file: {e}"
