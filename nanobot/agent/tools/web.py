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
    # 先删除 script 标签及其内容
    text = re.sub(r'<script[\s\S]*?</script>', '', text, flags=re.I)
    # 再删除 style 标签及其内容
    text = re.sub(r'<style[\s\S]*?</style>', '', text, flags=re.I)
    # 删除所有 HTML 标签
    text = re.sub(r'<[^>]+>', '', text)
    # 解码 HTML 实体（如 &lt; 转为 <），并去除首尾空白
    return html.unescape(text).strip()


def _normalize(text: str) -> str:
    """规范化空白字符：合并连续空格，保留最多两个换行"""
    # 将多个空格/制表符合并为一个空格
    text = re.sub(r'[ \t]+', ' ', text)
    # 将三个及以上换行合并为两个换行
    return re.sub(r'\n{3,}', '\n\n', text).strip()


def _validate_url(url: str) -> tuple[bool, str]:
    """验证 URL 是否合法（只允许 http/https）"""
    try:
        # 解析 URL 结构
        p = urlparse(url)
        # 检查协议是否为 http 或 https
        if p.scheme not in ('http', 'https'):
            return False, "only http/https allowed"
        # 检查是否包含域名
        if not p.netloc:
            return False, "missing domain"
        return True, ""  # 验证通过
    except Exception as e:
        return False, str(e)


def _format_results(query: str, items: list[dict[str, Any]], n: int) -> str:
    """格式化搜索结果为多行文本"""
    # 无结果时返回提示
    if not items:
        return f'No results found for "{query}".'
    # 构建结果列表
    lines = [f'Search results for "{query}":\n']
    for i, item in enumerate(items[:n], 1):
        # 清理标题和摘要
        title = _normalize(_strip_tags(item.get("title", "")))
        snippet = _normalize(_strip_tags(item.get("content", "")))
        # 添加序号、标题和 URL
        lines.append(f"{i}. {title}\n   {item.get('url', '')}")
        # 有摘要时添加
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
        # 使用传入配置或默认配置
        self.config = config if config is not None else WebSearchConfig()
        # HTTP 代理地址
        self.proxy = proxy

    async def execute(self, query: str, count: int | None = None, **kwargs: Any) -> str:
        # 确定搜索提供商（默认 brave）
        provider = self.config.provider.strip().lower() or "brave"
        # 限制结果数量在 1-10 之间
        n = min(max(count or self.config.max_results, 1), 10)

        # 根据提供商分发到对应方法
        if provider == "tavily":
            return await self._search_tavily(query, n)
        else:
            return await self._search_brave(query, n)

    async def _search_brave(self, query: str, n: int) -> str:
        """使用 Brave Search API"""
        # 从配置或环境变量获取 API 密钥
        api_key = self.config.api_key or os.environ.get("BRAVE_API_KEY", "")
        if not api_key:
            return "Error: BRAVE_API_KEY not set. Please configure it in config file or environment variable."

        try:
            # 使用异步 HTTP 客户端发送搜索请求
            async with httpx.AsyncClient(proxy=self.proxy) as client:
                r = await client.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    params={"q": query, "count": n},
                    headers={"Accept": "application/json", "X-Subscription-Token": api_key},
                    timeout=10.0,
                )
                # 检查 HTTP 状态码，失败会抛出异常
                r.raise_for_status()
            # 提取搜索结果列表
            items = [
                {"title": x.get("title", ""), "url": x.get("url", ""), "content": x.get("description", "")}
                for x in r.json().get("web", {}).get("results", [])
            ]
            return _format_results(query, items, n)
        except Exception as e:
            return f"Error: Brave search failed: {e}"

    async def _search_tavily(self, query: str, n: int) -> str:
        """使用 Tavily API"""
        # 从配置或环境变量获取 API 密钥
        api_key = self.config.api_key or os.environ.get("TAVILY_API_KEY", "")
        if not api_key:
            return "Error: TAVILY_API_KEY not set. Please configure it in config file or environment variable."

        try:
            # 使用异步 HTTP 客户端发送 POST 搜索请求
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
        # 最大返回字符数
        self.max_chars = max_chars
        # HTTP 代理地址
        self.proxy = proxy

    async def execute(self, url: str, extractMode: str = "markdown", maxChars: int | None = None, **kwargs: Any) -> str:
        # 使用参数或默认值
        max_chars = maxChars or self.max_chars
        # 验证 URL 合法性
        is_valid, error_msg = _validate_url(url)
        if not is_valid:
            return f"Error: Invalid URL ({error_msg})."

        # 执行抓取
        return await self._fetch(url, extractMode, max_chars)

    async def _fetch(self, url: str, extract_mode: str, max_chars: int) -> str:
        """抓取网页内容"""
        try:
            # 创建 HTTP 客户端，配置跟随重定向、超时、代理
            async with httpx.AsyncClient(
                follow_redirects=True,  # 自动跟随重定向
                max_redirects=MAX_REDIRECTS,  # 最多重定向次数
                timeout=30.0,  # 超时时间
                proxy=self.proxy,  # 代理设置
            ) as client:
                # 发送 GET 请求
                r = await client.get(url, headers={"User-Agent": USER_AGENT})
                r.raise_for_status()  # 检查 HTTP 状态

            # 获取 Content-Type
            ctype = r.headers.get("content-type", "")

            # 根据内容类型处理响应
            if "application/json" in ctype:
                # JSON 格式美化输出
                text = json.dumps(r.json(), indent=2, ensure_ascii=False)
            elif "text/html" in ctype or r.text[:256].lower().startswith(("<!doctype", "<html")):
                # HTML 格式提取正文
                text = self._extract_html_content(r.text, extract_mode)
            else:
                # 其他类型直接返回原文
                text = r.text

            # 截断超长内容
            if len(text) > max_chars:
                text = text[:max_chars] + "\n... (truncated)"

            return f"Content from {str(r.url)}:\n\n{text}"
        except httpx.HTTPStatusError as e:
            return f"Error: HTTP {e.response.status_code}."
        except Exception as e:
            return f"Error: Failed to fetch page ({e})."

    def _extract_html_content(self, html_text: str, extract_mode: str) -> str:
        """从 HTML 中提取正文"""
        title = ""  # 页面标题

        try:
            # 尝试使用 readability 库提取正文
            from readability import Document

            doc = Document(html_text)
            title = _normalize(_strip_tags(doc.title() or ""))  # 提取标题
            body_html = doc.summary()  # 提取正文 HTML
        except Exception:
            # readability 失败则手动提取
            title = self._extract_title_from_html(html_text)
            # 提取 body 标签内容
            body_match = re.search(r"<body[^>]*>([\s\S]*?)</body>", html_text, flags=re.I)
            body_html = body_match.group(1) if body_match else html_text

        # 根据提取模式处理
        if extract_mode == "markdown":
            content = self._to_markdown(body_html)  # 转为 Markdown
        else:
            content = _normalize(_strip_tags(body_html))  # 纯文本

        # 组合标题和内容
        if title and content:
            return f"# {title}\n\n{content}"
        return title or content

    def _extract_title_from_html(self, html_text: str) -> str:
        """从 HTML 中提取标题。

        通过正则匹配 <title> 标签提取标题文本，
        然后去除 HTML 标签并规范化空白字符后返回。
        """
        # 用正则匹配 <title> 标签中的内容（忽略大小写，支持换行）
        match = re.search(r"<title[^>]*>([\s\S]*?)</title>", html_text, flags=re.I)
        # 如果找不到 title 标签，返回空字符串
        if not match:
            return ""
        # 提取匹配到的文本，去除标签并规范化空白
        return _normalize(_strip_tags(match.group(1)))

    def _to_markdown(self, html_content: str) -> str:
        """将 HTML 转换为 Markdown。

        转换规则如下：
        1. 超链接 <a href="url">text</a> → [text](url)
        2. 标题 <h1>~<h6> → # ~ ######
        3. 列表项 <li> → - 项
        4. 段落/区块结束 </p></div> → 双换行
        5. 换行/分割线 <br><hr> → 单换行
        """
        # 第1步：转换超链接 <a href="url">text</a> → [text](url)
        text = re.sub(
            r'<a\s+[^>]*href=["\']([^"\']+)["\'][^>]*>([\s\S]*?)</a>',
            # 匹配到超链接时，提取 href 和标签内容，转为 Markdown 链接格式
            lambda m: f'[{_strip_tags(m[2])}]({m[1]})', html_content, flags=re.I)
        # 第2步：转换标题 <h1>~<h6> → 对应数量的 # + 标题文本
        text = re.sub(
            r'<h([1-6])[^>]*>([\s\S]*?)</h\1>',
            # m[1] 是标题级别（1-6），m[2] 是标题内容
            lambda m: f'\n{"#" * int(m[1])} {_strip_tags(m[2])}\n', text, flags=re.I)
        # 第3步：转换列表项 <li> → - 项
        text = re.sub(
            r'<li[^>]*>([\s\S]*?)</li>',
            # 将每个列表项转为减号前缀的 Markdown 列表项
            lambda m: f'\n- {_strip_tags(m[1])}', text, flags=re.I)
        # 第4步：段落/区块闭合标签 → 双换行（分段）
        text = re.sub(r'</(p|div|section|article)>', '\n\n', text, flags=re.I)
        # 第5步：换行符和水平线 → 单换行
        text = re.sub(r'<(br|hr)\s*/?>', '\n', text, flags=re.I)
        # 最后：去除剩余 HTML 标签，规范化空白并返回
        return _normalize(_strip_tags(text))
