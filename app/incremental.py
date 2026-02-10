import json
import os
import re
import time
import hashlib
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional
from urllib.parse import urljoin, urlparse
import xml.etree.ElementTree as ET

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.config import settings
from app.text_utils import clean_text, chunk_text
from app.tools import refresh_vectorstore


BASE_URL = os.getenv("BIL_BASE_URL", "https://www.bil.bt").rstrip("/")
STATE_PATH = Path("data/index_state.json")
FORMS_JSON = Path("data/forms.json")
VS_DIR = Path("data/faiss_index")
RAW_DIR = Path("data/raw")

DOC_EXTS = (".pdf", ".docx", ".doc")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".ico")
SKIP_EXTS = IMAGE_EXTS + (".css", ".js", ".mp4", ".mp3", ".zip", ".rar", ".7z", ".woff", ".woff2", ".ttf", ".eot")

SITEMAP_CANDIDATES = [
    "/sitemap.xml",
    "/sitemap_index.xml",
    "/sitemap-index.xml",
    "/wp-sitemap.xml",
]

DEFAULT_SEEDS = [
    "/",
    "/loans",
    "/insurance",
    "/announcements",
    "/about-us",
    "/e-services",
    "/help-and-support",
    "/publications",
    "/gmc/insurance",
]

S = requests.Session()
S.headers.update({"User-Agent": "BILChatbotIndexer/Incremental"})


def _normalize_url(u: str) -> str:
    u = (u or "").strip()
    u = u.split("#")[0]
    if u.endswith("/") and len(u) > len("https://x.y/"):
        u = u[:-1]
    return u


def _same_domain(u: str) -> bool:
    try:
        return urlparse(u).netloc == urlparse(BASE_URL).netloc
    except Exception:
        return False


def _is_doc(url: str) -> bool:
    ul = url.lower().split("?")[0]
    return any(ul.endswith(ext) for ext in DOC_EXTS)


def _is_skip_asset(url: str) -> bool:
    ul = url.lower().split("?")[0]
    return any(ul.endswith(ext) for ext in SKIP_EXTS)


def _fetch(url: str) -> requests.Response:
    r = S.get(url, timeout=25)
    r.raise_for_status()
    return r


def _head(url: str) -> Optional[requests.Response]:
    try:
        r = S.head(url, allow_redirects=True, timeout=20)
        if r.status_code >= 400:
            return None
        return r
    except Exception:
        return None


def _html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(" ")
    return re.sub(r"\s+", " ", text).strip()


def _extract_links(html: str, base_url: str) -> List[str]:
    soup = BeautifulSoup(html, "lxml")
    urls = set()
    for a in soup.select("a[href]"):
        href = a.get("href")
        if href:
            urls.add(urljoin(base_url, href))
    for m in re.findall(r"""https?://[^\s"'<>]+""", html):
        urls.add(m)
    cleaned = []
    for u in urls:
        u = _normalize_url(u)
        if u:
            cleaned.append(u)
    return cleaned


def _try_fetch_sitemap_urls() -> List[str]:
    urls = []
    for cand in SITEMAP_CANDIDATES:
        sm_url = BASE_URL + cand
        try:
            r = _fetch(sm_url)
            txt = r.text
            if "<urlset" not in txt and "<sitemapindex" not in txt:
                continue
            root = ET.fromstring(txt)
            locs = root.findall(".//{*}loc")
            for loc in locs:
                if loc.text:
                    u = _normalize_url(loc.text.strip())
                    if _same_domain(u):
                        urls.append(u)
        except Exception:
            continue
    return list(dict.fromkeys(urls))


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_state() -> Dict[str, Any]:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"urls": {}}


def _save_state(state: Dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _is_forms_url(url: str) -> bool:
    ul = url.lower()
    return ("/document/forms/" in ul) and ul.split("?")[0].endswith(".pdf")


def _title_from_filename(url: str) -> str:
    name = url.split("?")[0].split("/")[-1]
    name = re.sub(r"\.(pdf|docx|doc)$", "", name, flags=re.IGNORECASE)
    name = name.replace("-", " ").replace("_", " ").strip()
    name = re.sub(r"\s+", " ", name)
    return " ".join(w.capitalize() if len(w) > 2 else w for w in name.split())


def _download_file(url: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    name = os.path.basename(urlparse(url).path) or "file"
    name = re.sub(r"[^a-zA-Z0-9._-]", "_", name)
    path = RAW_DIR / name
    r = _fetch(url)
    path.write_bytes(r.content)
    return path


def _extract_pdf_text_and_title(path: Path) -> tuple[str, str]:
    reader = PdfReader(str(path))
    meta_title = ""
    try:
        if reader.metadata and getattr(reader.metadata, "title", None):
            meta_title = str(reader.metadata.title or "").strip()
    except Exception:
        meta_title = ""

    parts = []
    first_page_text = ""
    for i, page in enumerate(reader.pages):
        t = page.extract_text() or ""
        t = re.sub(r"\s+", " ", t).strip()
        if i == 0:
            first_page_text = t
        if t:
            parts.append(t)

    full_text = re.sub(r"\s+", " ", " ".join(parts)).strip()
    derived_title = first_page_text[:120].strip() if first_page_text else ""
    derived_title = re.sub(r"[\|\•]+", " ", derived_title).strip()
    title = meta_title or derived_title or path.name
    return full_text, title[:140].strip()


def incremental_update(max_pages: int = 200, crawl_delay_s: float = 0.2) -> int:
    state = _load_state()
    state_urls = state.get("urls", {})

    queue = []
    sitemap_urls = _try_fetch_sitemap_urls()
    if sitemap_urls:
        queue.extend(sitemap_urls)
    else:
        queue.extend([_normalize_url(BASE_URL + s) for s in DEFAULT_SEEDS])
    queue.append(_normalize_url(BASE_URL + "/"))
    queue = list(dict.fromkeys(queue))
    seen = set()

    new_docs: List[Document] = []
    forms_catalog = {}

    processed = 0
    idx = 0
    while idx < len(queue) and processed < max_pages:
        url = queue[idx]
        idx += 1
        url = _normalize_url(url)
        if not url or url in seen or not _same_domain(url):
            continue
        seen.add(url)
        if _is_skip_asset(url):
            continue
        processed += 1

        try:
            if _is_doc(url):
                if url.lower().split("?")[0].endswith(".pdf"):
                    state_entry = state_urls.get(url, {})
                    head = _head(url)
                    etag = (head.headers.get("ETag") if head else None) or ""
                    last_mod = (head.headers.get("Last-Modified") if head else None) or ""

                    if etag and state_entry.get("etag") == etag:
                        state_entry["last_crawled"] = int(time.time())
                        state_urls[url] = state_entry
                        continue
                    if (not etag) and last_mod and state_entry.get("last_modified") == last_mod:
                        state_entry["last_crawled"] = int(time.time())
                        state_urls[url] = state_entry
                        continue

                    path = _download_file(url)
                    text, title = _extract_pdf_text_and_title(path)
                    if not text:
                        continue
                    h = _hash_text(text)
                    if state_entry.get("hash") == h:
                        state_entry.update(
                            {"hash": h, "last_crawled": int(time.time()), "etag": etag, "last_modified": last_mod}
                        )
                        state_urls[url] = state_entry
                        continue

                    state_urls[url] = {
                        "hash": h,
                        "last_crawled": int(time.time()),
                        "etag": etag,
                        "last_modified": last_mod,
                    }
                    chunks = chunk_text(text)
                    for idx2, ck in enumerate(chunks):
                        ck = clean_text(ck)
                        if not ck:
                            continue
                        new_docs.append(Document(
                            page_content=ck,
                            metadata={
                                "source": url,
                                "title": title,
                                "type": "document",
                                "section_type": "Other",
                                "chunk_index": idx2,
                                "content_hash": h,
                                "updated_at": int(time.time()),
                            },
                        ))
                    if _is_forms_url(url):
                        clean_title = _title_from_filename(url) or title
                        forms_catalog[url] = {"title": clean_title, "url": url, "category": "form"}
                continue

            r = _fetch(url)
            ctype = (r.headers.get("Content-Type") or "").lower()
            if ("text/html" not in ctype) and ("application/xhtml" not in ctype):
                continue
            html = r.text
            text = _html_to_text(html)
            if not text:
                continue
            h = _hash_text(text)
            if state_urls.get(url, {}).get("hash") == h:
                continue
            soup = BeautifulSoup(html, "lxml")
            title = soup.title.text.strip() if soup.title else url
            state_urls[url] = {"hash": h, "last_crawled": int(time.time())}
            chunks = chunk_text(text)
            for idx, ck in enumerate(chunks):
                ck = clean_text(ck)
                if not ck:
                    continue
                new_docs.append(Document(
                    page_content=ck,
                    metadata={
                        "source": url,
                        "title": title,
                        "type": "page",
                        "section_type": "Other",
                        "chunk_index": idx,
                        "content_hash": h,
                        "updated_at": int(time.time()),
                    },
                ))

            # small link discovery (single hop)
            for u in _extract_links(html, url):
                if _same_domain(u) and not _is_skip_asset(u):
                    queue.append(u)

            time.sleep(crawl_delay_s)
        except Exception:
            continue

    if not new_docs:
        _save_state({"urls": state_urls})
        return 0

    embeddings = OpenAIEmbeddings(
        model=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
        api_key=os.getenv("OPENAI_API_KEY", ""),
    )

    if VS_DIR.exists():
        vectorstore = FAISS.load_local(str(VS_DIR), embeddings, allow_dangerous_deserialization=True)
        vectorstore.add_documents(new_docs)
    else:
        VS_DIR.mkdir(parents=True, exist_ok=True)
        vectorstore = FAISS.from_documents(new_docs, embeddings)

    vectorstore.save_local(str(VS_DIR))
    refresh_vectorstore()

    # Update forms catalog
    existing = {}
    if FORMS_JSON.exists():
        try:
            existing = {f["url"]: f for f in json.loads(FORMS_JSON.read_text(encoding="utf-8"))}
        except Exception:
            existing = {}
    existing.update(forms_catalog)
    forms_list = list(existing.values())
    forms_list.sort(key=lambda x: (x.get("title", "").lower(), x.get("url", "")))
    FORMS_JSON.write_text(json.dumps(forms_list, ensure_ascii=False, indent=2), encoding="utf-8")

    _save_state({"urls": state_urls})
    return len(new_docs)


def start_incremental_loop() -> None:
    enabled = os.getenv("ENABLE_INCREMENTAL_INDEX", "1") == "1"
    if not enabled:
        return

    interval_hours = int(os.getenv("CRAWL_INTERVAL_HOURS", "6"))
    max_pages = int(os.getenv("CRAWL_MAX_PAGES", "200"))
    delay_s = float(os.getenv("CRAWL_DELAY_S", "0.2"))

    def _loop():
        while True:
            try:
                incremental_update(max_pages=max_pages, crawl_delay_s=delay_s)
            except Exception:
                pass
            time.sleep(max(1, interval_hours) * 3600)

    t = threading.Thread(target=_loop, daemon=True)
    t.start()
