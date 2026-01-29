import re
from typing import Dict, List
from bs4 import BeautifulSoup
from urllib.parse import urljoin

YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")

def absolute_url(base: str, href: str) -> str:
    return urljoin(base, href)

def extract_links(base_url: str, html: str) -> List[str]:
    soup = BeautifulSoup(html, "lxml")
    urls: List[str] = []
    for a in soup.select("a[href]"):
        href = (a.get("href") or "").strip()
        if not href or href.startswith("#"):
            continue
        urls.append(absolute_url(base_url, href))
    return urls

def _txt(el) -> str:
    return el.get_text(" ", strip=True) if el else ""

def _meta_content(soup: BeautifulSoup, key: str) -> str:
    tag = soup.find("meta", attrs={"name": key}) or soup.find("meta", attrs={"property": key})
    if not tag:
        return ""
    return (tag.get("content") or "").strip()

def parse_publication_page(url: str, html: str) -> Dict:
    soup = BeautifulSoup(html, "lxml")

    title = _txt(soup.find("h1")) or _meta_content(soup, "citation_title") or _meta_content(soup, "og:title") or _txt(soup.find("title"))

    page_text = soup.get_text(" ", strip=True)
    year = ""
    m = YEAR_RE.search(page_text)
    if m:
        year = m.group(0)
    else:
        meta_date = _meta_content(soup, "citation_publication_date") or _meta_content(soup, "citation_date")
        m2 = YEAR_RE.search(meta_date)
        if m2:
            year = m2.group(0)

    authors = []
    author_urls = []
    for a in soup.select('a[href*="/en/persons/"]'):
        name = _txt(a)
        href = (a.get("href") or "").strip()
        if name and name not in authors:
            authors.append(name)
        if href:
            au = absolute_url(url, href)
            if au not in author_urls:
                author_urls.append(au)

    if not authors:
        for tag in soup.find_all("meta", attrs={"name": "citation_author"}):
            name = (tag.get("content") or "").strip()
            if name and name not in authors:
                authors.append(name)

    abstract = ""
    h = soup.find(lambda tag: tag.name in ("h2","h3","strong") and "abstract" in tag.get_text(" ", strip=True).lower())
    if h:
        nxt = h.find_next(["p","div"])
        abstract = _txt(nxt)
    if not abstract:
        abstract = _meta_content(soup, "citation_abstract") or _meta_content(soup, "description")

    return {
        "publication_url": url,
        "title": title,
        "year": year,
        "authors": authors,
        "author_urls": author_urls,
        "abstract": abstract,
        "organisations": [],
    }

def parse_list_page_for_publications(base_url: str, html: str) -> List[Dict]:
    soup = BeautifulSoup(html, "lxml")
    pubs = []
    seen = set()

    for a in soup.select('a[href*="/en/publications/"]'):
        href = (a.get("href") or "").strip()
        if not href:
            continue
        absu = absolute_url(base_url, href)
        if "/en/publications/" not in absu or absu in seen:
            continue

        title = _txt(a)
        if not title:
            title = (a.get("title") or "").strip()
        if not title:
            title = (a.get("aria-label") or "").strip()

        if len(title) < 4:
            continue

        pubs.append({"title": title, "publication_url": absu})
        seen.add(absu)

    return pubs
