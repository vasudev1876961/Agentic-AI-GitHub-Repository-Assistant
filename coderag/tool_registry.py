"""
Tool Registry for Agentic CodeRAG
---------------------------------
This module registers all available tools (local + external)
that the agent can call when it needs to perform specific actions.
"""

import logging

logger = logging.getLogger(__name__)

# Tool registry dictionary
TOOL_REGISTRY = {}


def register_tool(name):
    """Decorator to register a new tool."""
    def wrapper(func):
        TOOL_REGISTRY[name] = func
        logger.info(f" Registered tool: {name}")
        return func
    return wrapper


def get_tool(name):
    """Retrieve a registered tool by name."""
    return TOOL_REGISTRY.get(name, None)


def list_tools():
    """List all registered tools."""
    return list(TOOL_REGISTRY.keys())

import importlib, pkgutil
import coderag.tools

# Auto-import all modules under coderag.tools
for _, modname, _ in pkgutil.iter_modules(coderag.tools.__path__):
    importlib.import_module(f"coderag.tools.{modname}")
