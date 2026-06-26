import re
import json
import os
import time
import requests
import xml.etree.ElementTree as ET
from src.config import WIKI_TOPICS, ARXIV_QUERIES, DOCS_CACHE, CHUNK_SIZE_CHARS

HEADERS = {"User-Agent": "MarketingAnalyticsRAG/2.0 (aneelaveldi09@gmail.com)"}


# ── Wikipedia ─────────────────────────────────────────────────────────────────

def fetch_wikipedia(title: str) -> dict | None:
    params = {
        "action": "query", "prop": "extracts",
        "explaintext": "1", "titles": title,
        "format": "json", "redirects": "1",
    }
    try:
        r = requests.get("https://en.wikipedia.org/w/api.php",
                         params=params, headers=HEADERS, timeout=12)
        page = next(iter(r.json()["query"]["pages"].values()))
        text = (page.get("extract") or "").strip()
        return {"title": page["title"], "text": text, "source_type": "Wikipedia"} if text else None
    except Exception:
        return None


# ── arXiv ─────────────────────────────────────────────────────────────────────

def fetch_arxiv(query: str, max_results: int = 8) -> list[dict]:
    url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{query}",
        "max_results": max_results,
        "sortBy": "relevance",
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        root = ET.fromstring(r.text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        docs = []
        for entry in root.findall("atom:entry", ns):
            title   = (entry.findtext("atom:title", namespaces=ns) or "").strip().replace("\n", " ")
            summary = (entry.findtext("atom:summary", namespaces=ns) or "").strip()
            authors = [a.findtext("atom:name", namespaces=ns) or ""
                       for a in entry.findall("atom:author", ns)]
            published = (entry.findtext("atom:published", namespaces=ns) or "")[:7]
            if title and summary and len(summary) > 100:
                text = f"Title: {title}\nAuthors: {', '.join(authors[:3])}\nPublished: {published}\n\nAbstract:\n{summary}"
                docs.append({"title": title, "text": text, "source_type": "arXiv Research Paper"})
        return docs
    except Exception:
        return []


# ── Chunker ───────────────────────────────────────────────────────────────────

def chunk_text(doc: dict) -> list[dict]:
    text  = doc["text"]
    title = doc["title"]
    stype = doc.get("source_type", "Article")
    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks, current = [], ""
    for para in paras:
        if len(current) + len(para) < CHUNK_SIZE_CHARS:
            current = (current + "\n\n" + para).strip()
        else:
            if current:
                chunks.append({"text": current, "source": title, "source_type": stype})
            current = para
    if current:
        chunks.append({"text": current, "source": title, "source_type": stype})
    return chunks


# ── Main loader ───────────────────────────────────────────────────────────────

def load_corpus(use_cache: bool = True) -> list[dict]:
    if use_cache and os.path.exists(DOCS_CACHE):
        with open(DOCS_CACHE) as f:
            return json.load(f)

    os.makedirs("data", exist_ok=True)
    all_chunks: list[dict] = []

    print("Fetching Wikipedia articles...")
    for topic in WIKI_TOPICS:
        doc = fetch_wikipedia(topic)
        if doc:
            c = chunk_text(doc)
            all_chunks.extend(c)
            print(f"  {topic}: {len(c)} chunks")
        else:
            print(f"  {topic}: not found")

    print("\nFetching arXiv research papers...")
    seen_titles: set[str] = set()
    for query in ARXIV_QUERIES:
        papers = fetch_arxiv(query, max_results=6)
        for paper in papers:
            if paper["title"] not in seen_titles:
                seen_titles.add(paper["title"])
                c = chunk_text(paper)
                all_chunks.extend(c)
        print(f"  '{query}': {len(papers)} papers")
        time.sleep(0.4)

    with open(DOCS_CACHE, "w") as f:
        json.dump(all_chunks, f)

    n_wiki  = sum(1 for c in all_chunks if c["source_type"] == "Wikipedia")
    n_arxiv = sum(1 for c in all_chunks if c["source_type"] == "arXiv Research Paper")
    print(f"\nTotal: {len(all_chunks)} chunks ({n_wiki} Wikipedia, {n_arxiv} arXiv papers)")
    return all_chunks
