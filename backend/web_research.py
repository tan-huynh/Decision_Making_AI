from __future__ import annotations

import re
from html import unescape
from typing import Any
from urllib.parse import quote_plus

import httpx


async def search_web(query: str, enabled: bool) -> dict[str, Any]:
    if not enabled or not query.strip():
        return {"enabled": enabled, "results": [], "summary": ""}
    url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"
    try:
        async with httpx.AsyncClient(timeout=8, follow_redirects=True) as client:
            response = await client.get(url, headers={"User-Agent": "DecisionMakingAI/0.1"})
            response.raise_for_status()
    except Exception as exc:
        return {"enabled": enabled, "results": [], "summary": f"Web research unavailable: {exc}"}

    html = response.text
    matches = re.findall(r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', html, flags=re.S)
    snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', html, flags=re.S)
    results = []
    for index, (href, title) in enumerate(matches[:5]):
        clean_title = re.sub("<.*?>", "", title)
        clean_snippet = re.sub("<.*?>", "", snippets[index] if index < len(snippets) else "")
        results.append(
            {
                "title": unescape(clean_title).strip(),
                "url": unescape(href).strip(),
                "snippet": unescape(clean_snippet).strip(),
            }
        )
    summary = "\n".join(f"- {item['title']}: {item['snippet']}" for item in results)
    return {"enabled": enabled, "results": results, "summary": summary}
