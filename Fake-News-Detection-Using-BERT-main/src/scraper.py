"""
News Article and Web Page Scraper for Fake News Detection.
Extracts title, body, author, and publication metadata from news URLs.
"""
import re
import logging
from typing import Dict, Any, Optional
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


def extract_article_from_url(url: str, timeout: int = 10) -> Dict[str, Any]:
    """
    Fetches and extracts headline, body text, domain, and metadata from a news URL.
    """
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid URL format: '{url}'. Must include http:// or https://")

    domain = parsed.netloc.replace("www.", "")

    try:
        response = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch article from {url}: {e}")
        raise RuntimeError(f"Unable to access URL: {e}")

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove non-content elements
    for element in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "form"]):
        element.decompose()

    # 1. Extract Title / Headline
    title = ""
    og_title = soup.find("meta", property="og:title") or soup.find("meta", attrs={"name": "twitter:title"})
    if og_title and og_title.get("content"):
        title = og_title["content"].strip()
    elif soup.title and soup.title.string:
        title = soup.title.string.strip()
    elif soup.find("h1"):
        title = soup.find("h1").get_text(strip=True)

    # 2. Extract Author
    author = "Unknown"
    author_meta = (
        soup.find("meta", attrs={"name": "author"})
        or soup.find("meta", property="article:author")
        or soup.find("span", class_=re.compile(r"author|byline", re.I))
    )
    if author_meta:
        if hasattr(author_meta, "get") and author_meta.get("content"):
            author = author_meta["content"].strip()
        else:
            author = author_meta.get_text(strip=True)

    # 3. Extract Main Body Text
    paragraphs = []
    # Prefer <article> tag if available
    article_tag = soup.find("article") or soup.find("main") or soup.find("div", class_=re.compile(r"article-body|content|entry-content", re.I))
    search_context = article_tag if article_tag else soup

    for p in search_context.find_all("p"):
        p_text = p.get_text(strip=True)
        # Filter out short menu or disclaimer paragraphs
        if len(p_text.split()) > 6:
            paragraphs.append(p_text)

    body_text = "\n\n".join(paragraphs)

    # Fallback if few paragraphs found
    if not body_text:
        body_text = soup.get_text(separator=" ", strip=True)
        # Collapse excessive whitespace
        body_text = re.sub(r"\s+", " ", body_text)[:5000]

    return {
        "url": url,
        "domain": domain,
        "title": title or "Untitled Article",
        "author": author,
        "text": body_text,
        "word_count": len(body_text.split()),
        "summary": body_text[:400] + ("..." if len(body_text) > 400 else "")
    }
