from coderag.search import search_code
from coderag.tool_registry import register_tool

@register_tool("local_search")
def local_search_tool(query: str) -> str:
    """Search local codebase using FAISS."""
    results = search_code(query)
    return "\n".join([r["content"] for r in results]) if results else "No local matches found."
