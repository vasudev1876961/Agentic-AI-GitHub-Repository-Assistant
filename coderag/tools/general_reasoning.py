"""
General Reasoning Tool
----------------------
Fallback tool for out-of-context or general queries.
Uses Gemini with retry and robust error handling.
"""

import logging

from google import genai
from tenacity import retry, stop_after_attempt, wait_exponential

from coderag.config import GEMINI_API_KEY, GEMINI_CHAT_MODEL
from coderag.tool_registry import register_tool

logger = logging.getLogger(__name__)

# Initialize Gemini Client
try:
    if not GEMINI_API_KEY:
        raise ValueError(" Missing GEMINI_API_KEY in environment.")
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    logger.info(f" Gemini client initialized for reasoning tool ({GEMINI_CHAT_MODEL})")
except Exception as e:
    logger.error(f" Gemini client init failed: {e}")
    gemini_client = None


# Retry Wrapper for Gemini Calls
@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, max=10))
def call_gemini_with_retry(prompt: str) -> str:
    """Call Gemini with retry logic for overloads (503, unavailable, etc.)."""
    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_CHAT_MODEL,
            contents=prompt,
        )
        if hasattr(response, "text"):
            return response.text
        elif hasattr(response, "candidates"):
            return response.candidates[0].content.parts[0].text
        else:
            raise RuntimeError("Empty response from Gemini")

    except Exception as e:
        if "503" in str(e) or "UNAVAILABLE" in str(e):
            logger.warning(f" Gemini overloaded, retrying... ({e})")
            raise
        raise


# Register the General Reasoning Tool
@register_tool("general_reasoning")
def general_reasoning_tool(query: str) -> str:
    """Fallback tool for general knowledge or reasoning queries."""
    prompt = f"Answer this clearly, concisely, and accurately:\n\n{query}"

    try:
        if not gemini_client:
            raise RuntimeError("Gemini client not initialized.")
        text_output = call_gemini_with_retry(prompt)

        if not text_output or not text_output.strip():
            raise ValueError("Empty response from Gemini.")
        logger.info(" General reasoning response from Gemini.")
        return text_output.strip()

    except Exception as e:
        logger.error(f" General reasoning tool failed: {e}")
        return f" Gemini reasoning error: {e}"
