import logging
import re
from urllib.parse import unquote, urljoin

import requests
from bs4 import BeautifulSoup

from coderag.tool_registry import register_tool

logger = logging.getLogger(__name__)


@register_tool("web_fetch")
def web_fetch_tool(query: str) -> str:
    """
    🌐 Smart Web Fetch (Clean & Summarized)
    ---------------------------------------
    ✔ Searches DuckDuckGo
    ✔ Fetches Wikipedia or high-quality pages
    ✔ Extracts ONLY readable paragraphs
    ✔ Produces:
        - 1–2 short paragraphs (summary)
        - 3 bullet points (key facts)
    """
    try:
        logger.info(f"🌐 web_fetch started for query: {query}")

        # 1️⃣ DuckDuckGo search
        search_url = f"https://duckduckgo.com/html/?q={query}"
        res = requests.get(
            search_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10
        )
        soup = BeautifulSoup(res.text, "html.parser")

        links = soup.find_all("a", {"class": "result__a"}, limit=5)
        if not links:
            return f"⚠️ No results found for '{query}'."

        for tag in links:
            raw = tag.get("href")
            if not raw:
                continue

            # Decode DuckDuckGo redirect
            if "uddg=" in raw:
                m = re.search(r"uddg=([^&]+)", raw)
                if m:
                    raw = unquote(m.group(1))

            # Prefer Wikipedia
            if "wikipedia.org" not in raw:
                continue

            # 2️⃣ Fetch the page
            page = requests.get(raw, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            soup_page = BeautifulSoup(page.text, "html.parser")

            paragraphs = soup_page.find_all("p")
            clean_paras = []

            for p in paragraphs:
                text = p.get_text(" ", strip=True)
                # Filter junk
                if len(text) > 80 and "may refer to" not in text:
                    clean_paras.append(text)

            if not clean_paras:
                continue

            # 3️⃣ Choose first meaningful paragraphs (1–2)
            summary_paragraphs = clean_paras[:2]
            summary_text = "\n\n".join(summary_paragraphs)

            # 4️⃣ Create 3 bullet points
            bullet_points = []
            for para in clean_paras:
                sentences = re.split(r"(?<=[.!?]) +", para)
                for s in sentences:
                    if len(s) > 40 and len(bullet_points) < 3:
                        bullet_points.append(s.strip())
                if len(bullet_points) >= 3:
                    break

            # 5️⃣ Build final clean output
            title_tag = soup_page.find("h1")
            title = title_tag.get_text(strip=True) if title_tag else query

            bullets = "\n- ".join(bullet_points)

            return (
                f"Fetched from {raw}:\n\n"
                f"### {title}\n\n"
                f"{summary_text}\n\n"
                f"**Key Points:**\n"
                f"- {bullets}"
            )

        return f"⚠️ No suitable detailed page found for '{query}'."

    except Exception as e:
        logger.error(f"❌ Web fetch failed: {e}")
        return f"⚠️ Web fetch failed: {e}"
