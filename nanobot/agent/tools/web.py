# -*- coding: utf-8 -*-
"""Web 工具模块：网页搜索和网页抓取"""

from __future__ import annotations

import html
import json
import os
import re
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import httpx

from nanobot.agent.tools.base import Tool

if TYPE_CHECKING:
    from nanobot.config.schema import WebSearchConfig


USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_7_2) AppleWebKit/537.36"
MAX_REDIRECTS = 5


def _strip_tags(text: str) -> str:
    """去除 HTML 标签并解码 HTML 实体"""
    text = re.sub(r'<script[\s\S]*?</script>', '', text, flags=re.I)
    text = re.sub(r'<style[\s\S]*?</style>', '', text, flags=re.I)
    text = re.sub(r'<[^>]+>', '', text)
    return html.unescape(text).strip()


def _normalize(text: str) -> str:
    """规范化空白字符"""
    text = re.sub(r'[ \t]+', ' ', text)
    return re.sub(r'\n{3,}', '\n\n', text).strip()


def _validate_url(url: str) -> tuple[bool, str]:
    """验证 URL 是否合法"""
    try:
        p = urlparse(url)
        if p.scheme not in ('http', 'https'):
            return False, "only http/https allowed"
        if not p.netloc:
            return False, "missing domain"
        return True, ""
    except Exception as e:
        return False, str(e)


def _format_results(query: str, items: list[dict[str, Any]], n: int) -> str:
    """格式化搜索结果"""
    if not items:
        return f'No results found for "{query}".'
    lines = [f'Search results for "{query}":\n']
    for i, item in enumerate(items[:n], 1):
        title = _normalize(_strip_tags(item.get("title", "")))
        snippet = _normalize(_strip_tags(item.get("content", "")))
        lines.append(f"{i}. {title}\n   {item.get('url', '')}")
        if snippet:
            lines.append(f"   {snippet}")
    return "\n".join(lines)


class WebSearchTool(Tool):
    """网页搜索工具，支持 Brave 和 Tavily"""

    name = "web_search"
    description = "Search the web and return titles, links and summaries."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query."},
            "count": {"type": "integer", "description": "Number of results (1-10).", "minimum": 1, "maximum": 10},
        },
        "required": ["query"],
    }

    def __init__(self, config: WebSearchConfig | None = None, proxy: str | None = None):
        from nanobot.config.schema import WebSearchConfig
        self.config = config if config is not None else WebSearchConfig()
        self.proxy = proxy

    async def execute(self, query: str, count: int | None = None, **kwargs: Any) -> str:
        provider = self.config.provider.strip().lower() or "brave"
        n = min(max(count or self.config.max_results, 1), 10)

        if provider == "tavily":
            return await self._search_tavily(query, n)
        else:
            return await self._search_brave(query, n)

    async def _search_brave(self, query: str, n: int) -> str:
        """使用 Brave Search API"""
        api_key = self.config.api_key or os.environ.get("BRAVE_API_KEY", "")
        if not api_key:
            return "Error: BRAVE_API_KEY not set. Please configure it in config file or environment variable."

        try:
            async with httpx.AsyncClient(proxy=self.proxy) as client:
                r = await client.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    params={"q": query, "count": n},
                    headers={"Accept": "application/json", "X-Subscription-Token": api_key},
                    timeout=10.0,
                )
                r.raise_for_status()
            items = [
                {"title": x.get("title", ""), "url": x.get("url", ""), "content": x.get("description", "")}
                for x in r.json().get("web", {}).get("results", [])
            ]
            return _format_results(query, items, n)
        except Exception as e:
            return f"Error: Brave search failed: {e}"

    async def _search_tavily(self, query: str, n: int) -> str:
        """使用 Tavily API"""
        api_key = self.config.api_key or os.environ.get("TAVILY_API_KEY", "")
        if not api_key:
            return "Error: TAVILY_API_KEY not set. Please configure it in config file or environment variable."

        try:
            async with httpx.AsyncClient(proxy=self.proxy) as client:
                r = await client.post(
                    "https://api.tavily.com/search",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={"query": query, "max_results": n},
                    timeout=15.0,
                )
                r.raise_for_status()
            return _format_results(query, r.json().get("results", []), n)
        except Exception as e:
            return f"Error: Tavily search failed: {e}"


class WebFetchTool(Tool):
    """网页抓取工具"""

    name = "web_fetch"
    description = "Fetch a web page and extract its main content."
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "URL to fetch."},
            "extractMode": {"type": "string", "enum": ["markdown", "text"], "default": "markdown"},
            "maxChars": {"type": "integer", "minimum": 100, "description": "Max characters to return."},
        },
        "required": ["url"],
    }

    def __init__(self, max_chars: int = 50000, proxy: str | None = None):
        self.max_chars = max_chars
        self.proxy = proxy

    async def execute(self, url: str, extractMode: str = "markdown", maxChars: int | None = None, **kwargs: Any) -> str:
        max_chars = maxChars or self.max_chars
        is_valid, error_msg = _validate_url(url)
        if not is_valid:
            return f"Error: Invalid URL ({error_msg})."

        return await self._fetch(url, extractMode, max_chars)

    async def _fetch(self, url: str, extract_mode: str, max_chars: int) -> str:
        """抓取网页内容"""
        try:
            async with httpx.AsyncClient(
                follow_redirects=True,
                max_redirects=MAX_REDIRECTS,
                timeout=30.0,
                proxy=self.proxy,
            ) as client:
                r = await client.get(url, headers={"User-Agent": USER_AGENT})
                r.raise_for_status()

            ctype = r.headers.get("content-type", "")

            if "application/json" in ctype:
                text = json.dumps(r.json(), indent=2, ensure_ascii=False)
            elif "text/html" in ctype or r.text[:256].lower().startswith(("<!doctype", "<html")):
                text = self._extract_html_content(r.text, extract_mode)
            else:
                text = r.text

            if len(text) > max_chars:
                text = text[:max_chars] + "\n... (truncated)"

            return f"Content from {str(r.url)}:\n\n{text}"
        except httpx.HTTPStatusError as e:
            return f"Error: HTTP {e.response.status_code}."
        except Exception as e:
            return f"Error: Failed to fetch page ({e})."

    def _extract_html_content(self, html_text: str, extract_mode: str) -> str:
        """从 HTML 中提取正文"""
        title = ""

        try:
            from readability import Document

            doc = Document(html_text)
            title = _normalize(_strip_tags(doc.title() or ""))
            body_html = doc.summary()
        except Exception:
            title = self._extract_title_from_html(html_text)
            body_match = re.search(r"<body[^>]*>([\s\S]*?)</body>", html_text, flags=re.I)
            body_html = body_match.group(1) if body_match else html_text

        if extract_mode == "markdown":
            content = self._to_markdown(body_html)
        else:
            content = _normalize(_strip_tags(body_html))

        if title and content:
            return f"# {title}\n\n{content}"
        return title or content

    def _extract_title_from_html(self, html_text: str) -> str:
        """从 HTML 中提取标题"""
        match = re.search(r"<title[^>]*>([\s\S]*?)</title>", html_text, flags=re.I)
        if not match:
            return ""
        return _normalize(_strip_tags(match.group(1)))

    def _to_markdown(self, html_content: str) -> str:
        """将 HTML 转换为 Markdown"""
        text = re.sub(r'<a\s+[^>]*href=["\']([^"\']+)["\'][^>]*>([\s\S]*?)</a>',
                      lambda m: f'[{_strip_tags(m[2])}]({m[1]})', html_content, flags=re.I)
        text = re.sub(r'<h([1-6])[^>]*>([\s\S]*?)</h\1>',
                      lambda m: f'\n{"#" * int(m[1])} {_strip_tags(m[2])}\n', text, flags=re.I)
        text = re.sub(r'<li[^>]*>([\s\S]*?)</li>', lambda m: f'\n- {_strip_tags(m[1])}', text, flags=re.I)
        text = re.sub(r'</(p|div|section|article)>', '\n\n', text, flags=re.I)
        text = re.sub(r'<(br|hr)\s*/?>', '\n', text, flags=re.I)
        return _normalize(_strip_tags(text))
