"""
Agent Controller — orchestrates tool usage + reasoning.
Now with retry + OpenAI fallback for robust execution.
"""

import logging

from google import genai
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from coderag.config import (
    GEMINI_API_KEY,
    GEMINI_CHAT_MODEL,
    OPENAI_API_KEY,
    OPENAI_CHAT_MODEL,
)
from coderag.planner import decide_tool
from coderag.tool_registry import get_tool

logger = logging.getLogger(__name__)

# Initialize Clients
try:
    if not GEMINI_API_KEY:
        raise ValueError(" Gemini API key not found in environment variables.")
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    logger.info(
        f" Gemini client initialized successfully with model: {GEMINI_CHAT_MODEL}"
    )
except Exception as e:
    logger.error(f" Failed to initialize Gemini client: {e}")
    gemini_client = None

# OpenAI fallback
try:
    openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
    if openai_client:
        logger.info(f" OpenAI fallback initialized ({OPENAI_CHAT_MODEL})")
except Exception as e:
    logger.warning(f" Could not initialize OpenAI fallback: {e}")
    openai_client = None


import logging

# Retry Wrapper for Gemini Calls
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, max=12),
    reraise=True,
)
def call_gemini_with_retry(prompt: str) -> str:
    """
    Safely call Gemini model with automatic retries for transient errors (503, timeouts, etc.).
    Includes structured logging and flexible response parsing.
    """
    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_CHAT_MODEL,
            contents=prompt,
        )

        text_output = None
        if hasattr(response, "text"):
            text_output = response.text
        elif hasattr(response, "candidates") and response.candidates:
            try:
                text_output = response.candidates[0].content.parts[0].text
            except Exception:
                text_output = None

        if not text_output or not text_output.strip():
            raise RuntimeError("Empty or malformed response from Gemini.")

        logger.info(" Gemini response retrieved successfully.")
        return text_output.strip()

    except Exception as e:
        if "503" in str(e) or "UNAVAILABLE" in str(e):
            logger.warning(f" Gemini service overloaded or unavailable: {e}")
            raise RuntimeError(
                "Gemini service is currently overloaded. Retrying automatically..."
            )
        logger.error(f" Gemini API call failed: {e}")
        raise


#  Main Agent Reasoning Function
def run_agent(user_query: str) -> str:
    logger.info(f" Received query: {user_query}")

    tool_name = decide_tool(user_query)
    tool = get_tool(tool_name)
    if not tool:
        logger.warning(f" No tool registered with name '{tool_name}'")
        return f" No suitable tool found for: {user_query}"

    logger.info(f" Using tool: {tool_name}")
    try:
        tool_output = tool(user_query)
    except Exception as e:
        logger.error(f" Tool '{tool_name}' execution failed: {e}")
        return f"Error executing tool '{tool_name}': {e}"

    logger.info(
        f" Tool '{tool_name}' output length: {len(tool_output) if tool_output else 0}"
    )

    if tool_name == "web_fetch" and tool_output and len(tool_output) > 150:
        logger.info(" Returning web_fetch output directly (no LLM call).")
        return tool_output

    prompt = f"""
You are an intelligent AI coding assistant.
The user asked: "{user_query}"

Tool used: {tool_name}
Tool output (truncated if long):
{tool_output}

Provide a clear, helpful, and concise answer.
    """

    try:
        if not gemini_client:
            raise RuntimeError("Gemini client not initialized.")
        text_output = call_gemini_with_retry(prompt)
        if not text_output or not text_output.strip():
            raise ValueError("Empty response from Gemini")
        logger.info(" Successfully generated response using Gemini.")
        return text_output.strip()

    except Exception as e:
        logger.warning(f" Gemini failed after retries: {e}")
        if openai_client:
            ...
        return f" Gemini failed and no fallback available: {e}"
