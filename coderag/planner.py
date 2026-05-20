"""
🧭 Agentic Planner — Smart Tool Selection Logic (Improved)
----------------------------------------------------------
Fixes:
- Repo-intent queries ALWAYS use local_search
- Prevents accidental web_fetch on queries like “explain this repo”
"""

import logging

from coderag.search import search_code

logger = logging.getLogger(__name__)


def decide_tool(query: str) -> str:
    q = query.lower()
    logger.info(f"🧭 Planner analyzing raw user query: {query}")

    # 0️⃣ Detect repo-related intent → ALWAYS local_search
    REPO_KEYWORDS = [
        "repo",
        "repository",
        "project",
        "folder",
        "codebase",
        "module",
        "function",
        "class",
        "explain this repo",
        "explain repository",
        "what does this file do",
        "read this code",
        "how does this work",
        "flow",
        "architecture",
        "embedding",
        "vector",
        "faiss",
        "index",
    ]
    if any(k in q for k in REPO_KEYWORDS):
        logger.info("📘 Repo-intent detected → forcing 'local_search'")
        return "local_search"

    # 1️⃣ General factual questions → web_fetch
    GENERAL_QUESTION = [
        "who is",
        "what is",
        "latest",
        "news",
        "define",
        "meaning",
        "release",
        "update",
        "information about",
        "history of",
        "founder of",
    ]
    if any(k in q for k in GENERAL_QUESTION):
        logger.info("🌐 General knowledge question → using 'web_fetch'")
        return "web_fetch"

    # 2️⃣ File reading intent
    if any(
        k in q for k in ["read file", "show file", "open file", ".py", ".txt", ".md"]
    ):
        logger.info("📄 File intent → using 'read_file'")
        return "read_file"

    # 3️⃣ Try FAISS retrieval
    try:
        results = search_code(query)
        if results:
            best = results[0]["distance"]
            if best > 0.35:
                logger.info(f"💾 Good similarity ({best}) → using 'local_search'")
                return "local_search"
            else:
                logger.info(f"⚠️ Low similarity ({best}) → ignoring FAISS")
    except Exception as e:
        logger.warning(f"⚠️ FAISS error: {e}")

    # 4️⃣ Final fallback → general reasoning
    logger.info("🧠 Default fallback → using 'general_reasoning'")
    return "general_reasoning"
