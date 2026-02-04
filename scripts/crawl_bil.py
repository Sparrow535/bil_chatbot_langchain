import os
import re
import json
import time
from collections import Counter, defaultdict
from urllib.parse import urljoin, urlparse
from pathlib import Path
import xml.etree.ElementTree as ET

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from pypdf import PdfReader

# Optional JS-render fallback
USE_PLAYWRIGHT = True
try:
    from playwright.sync_api import sync_playwright
except Exception:
    USE_PLAYWRIGHT = False


BASE_URL = os.getenv("BIL_BASE_URL", "https://www.bil.bt").rstrip("/")
OUT_JSONL = Path("data/raw.jsonl")
RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

FORMS_JSON = Path("data/forms.json")
COVERAGE_JSON = Path("data/coverage_report.json")

S = requests.Session()
S.headers.update({"User-Agent": "BILChatbotIndexer/3.0"})


# --- URL / Asset filtering ---
DOC_EXTS = (".pdf", ".docx", ".doc")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".ico")
SKIP_EXTS = IMAGE_EXTS + (".css", ".js", ".mp4", ".mp3", ".zip", ".rar", ".7z", ".woff", ".woff2", ".ttf", ".eot")


SITEMAP_CANDIDATES = [
    "/sitemap.xml",
    "/sitemap_index.xml",
    "/sitemap-index.xml",
    "/wp-sitemap.xml",
]

# Strong seeds help hub→detail discovery even when home doesn’t link everything
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

def title_from_filename(url: str) -> str:
    name = url.split("?")[0].split("/")[-1]
    name = re.sub(r"\.(pdf|docx|doc)$", "", name, flags=re.IGNORECASE)
    name = name.replace("-", " ").replace("_", " ").strip()
    name = re.sub(r"\s+", " ", name)
    # Title-case (simple)
    return " ".join(w.capitalize() if len(w) > 2 else w for w in name.split())


def normalize_url(u: str) -> str:
    u = (u or "").strip()
    u = u.split("#")[0]
    # Remove trailing slash except domain root
    if u.endswith("/") and len(u) > len("https://x.y/"):
        u = u[:-1]
    return u

def same_domain(u: str) -> bool:
    try:
        return urlparse(u).netloc == urlparse(BASE_URL).netloc
    except Exception:
        return False

def is_doc(url: str) -> bool:
    ul = url.lower().split("?")[0]
    return any(ul.endswith(ext) for ext in DOC_EXTS)

def is_skip_asset(url: str) -> bool:
    ul = url.lower().split("?")[0]
    return any(ul.endswith(ext) for ext in SKIP_EXTS)

def fetch(url: str) -> requests.Response:
    r = S.get(url, timeout=25)
    r.raise_for_status()
    return r

def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(" ")
    return re.sub(r"\s+", " ", text).strip()

def extract_links(html: str, base_url: str):
    soup = BeautifulSoup(html, "lxml")
    urls = set()

    # Normal href links
    for a in soup.select("a[href]"):
        href = a.get("href")
        if href:
            urls.add(urljoin(base_url, href))

    # Some sites put URLs in data-* attributes
    for attr in ["data-url", "data-href", "data-link"]:
        for el in soup.select(f"[{attr}]"):
            v = el.get(attr)
            if v:
                urls.add(urljoin(base_url, v))

    # Pagination: rel=next
    for a in soup.select('a[rel="next"][href]'):
        urls.add(urljoin(base_url, a.get("href")))

    # Also scan raw html for hard-coded links
    for m in re.findall(r"""https?://[^\s"'<>]+""", html):
        urls.add(m)

    cleaned = []
    for u in urls:
        u = normalize_url(u)
        if u:
            cleaned.append(u)
    return cleaned

def should_fallback_to_js(text: str) -> bool:
    if len(text) < 300:
        return True
    if "loading" in text.lower() and len(text) < 900:
        return True
    return False

def render_with_playwright(url: str) -> str:
    if not USE_PLAYWRIGHT:
        return ""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until="networkidle", timeout=45000)
        html = page.content()
        browser.close()
        return html

def safe_filename_from_url(url: str) -> str:
    name = os.path.basename(urlparse(url).path) or "file"
    name = re.sub(r"[^a-zA-Z0-9._-]", "_", name)
    return name

def download_file(url: str) -> Path:
    path = RAW_DIR / safe_filename_from_url(url)
    if path.exists():
        return path
    r = fetch(url)
    path.write_bytes(r.content)
    return path

def extract_pdf_text_and_title(path: Path) -> tuple[str, str]:
    """
    Returns (text, title) where title is derived from:
      1) PDF metadata Title if present
      2) First meaningful line from page 1 text
      3) Filename fallback
    """
    reader = PdfReader(str(path))

    # 1) metadata title
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

    # 2) first page first line-ish
    derived_title = ""
    if first_page_text:
        # take first ~120 chars up to sentence break / separator
        derived_title = first_page_text[:120].strip()
        # reduce noise
        derived_title = re.sub(r"[\|\•]+", " ", derived_title).strip()

    title = meta_title or derived_title or path.name
    # normalize title length
    title = title.strip()
    if len(title) > 140:
        title = title[:140].rstrip()

    return full_text, title

def try_fetch_sitemap_urls() -> list[str]:
    urls = []
    for cand in SITEMAP_CANDIDATES:
        sm_url = BASE_URL + cand
        try:
            r = fetch(sm_url)
            txt = r.text
            if "<urlset" not in txt and "<sitemapindex" not in txt:
                continue

            root = ET.fromstring(txt)
            locs = root.findall(".//{*}loc")
            for loc in locs:
                if loc.text:
                    u = normalize_url(loc.text.strip())
                    if same_domain(u):
                        urls.append(u)
        except Exception:
            continue
    # unique preserve order
    return list(dict.fromkeys(urls))

def is_forms_url(url: str) -> bool:
    ul = url.lower()
    return ("/document/forms/" in ul) and ul.split("?")[0].endswith(".pdf")


def crawl(max_pages: int = 1500, crawl_delay_s: float = 0.15):
    # Reset outputs
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if OUT_JSONL.exists():
        OUT_JSONL.unlink()
    if FORMS_JSON.exists():
        FORMS_JSON.unlink()

    seen = set()
    queue = []

    # Seeds: sitemap if available else strong defaults
    sitemap_urls = try_fetch_sitemap_urls()
    if sitemap_urls:
        queue.extend(sitemap_urls)
    else:
        queue.extend([normalize_url(BASE_URL + s) for s in DEFAULT_SEEDS])

    # Always include homepage
    queue.append(normalize_url(BASE_URL + "/"))
    # unique
    queue = list(dict.fromkeys(queue))

    # Coverage tracking
    stats = Counter()
    top_paths = Counter()
    failed_urls = []
    skipped_assets = []
    discovered_all = set(queue)  # will grow
    forms_catalog = {}  # url -> {title,url,category}

    with OUT_JSONL.open("a", encoding="utf-8") as f:
        for _ in tqdm(range(max_pages)):
            if not queue:
                break

            url = normalize_url(queue.pop(0))
            if not url or url in seen:
                continue
            if not same_domain(url):
                continue

            discovered_all.add(url)

            # Skip obvious assets early
            if is_skip_asset(url):
                stats["skipped_assets_ext"] += 1
                skipped_assets.append(url)
                seen.add(url)
                continue

            seen.add(url)

            try:
                # Document handling
                if is_doc(url):
                    # but skip images already; docs are pdf/docx/doc
                    if url.lower().split("?")[0].endswith(".pdf"):
                        path = download_file(url)

                        # content-type guard (avoid jpg served with pdf-like url)
                        # if server sends image/*, skip
                        # (We re-request headers quickly)
                        try:
                            head = S.head(url, timeout=15)
                            ctype = (head.headers.get("Content-Type") or "").lower()
                            if ctype.startswith("image/"):
                                stats["skipped_assets_ctype"] += 1
                                continue
                        except Exception:
                            pass

                        text, title = extract_pdf_text_and_title(path)
                        if text:
                            rec = {
                                "type": "document",
                                "title": title or path.name,
                                "source": url,
                                "text": text,
                            }
                            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            stats["docs_pdf_indexed"] += 1

                        # Forms catalog: specifically /document/forms/*.pdf
                        if is_forms_url(url):
                            # Prefer filename title always (clean + stable)
                            clean_title = title_from_filename(url)
                            # If filename somehow empty, fallback to extracted title
                            if not clean_title:
                                clean_title = title or path.name
                            forms_catalog[url] = {
                                "title": clean_title,
                                "url": url,
                                "category": "form"
                            }
                            stats["forms_detected"] += 1

                    else:
                        # doc/docx support can be added similarly, but bil.bt forms are mostly pdf
                        stats["docs_other_skipped"] += 1

                    # Count path bucket
                    top_paths[urlparse(url).path.split("/")[1] if len(urlparse(url).path.split("/")) > 1 else "root"] += 1
                    continue

                # HTML page handling
                r = fetch(url)
                ctype = (r.headers.get("Content-Type") or "").lower()

                # Skip binary / image content
                if ctype.startswith("image/"):
                    stats["skipped_assets_ctype"] += 1
                    skipped_assets.append(url)
                    continue

                # If not html-ish, skip
                if ("text/html" not in ctype) and ("application/xhtml" not in ctype):
                    stats["skipped_non_html"] += 1
                    continue

                html = r.text
                text = html_to_text(html)

                # JS fallback only if needed
                if should_fallback_to_js(text):
                    js_html = render_with_playwright(url)
                    if js_html:
                        html = js_html
                        text = html_to_text(html)

                # If still empty-ish, keep but mark
                soup = BeautifulSoup(html, "lxml")
                title = soup.title.text.strip() if soup.title else url

                if text:
                    f.write(json.dumps({
                        "type": "page",
                        "title": title,
                        "source": url,
                        "text": text
                    }, ensure_ascii=False) + "\n")
                    stats["pages_indexed"] += 1
                else:
                    stats["pages_empty"] += 1

                # Discover more links
                discovered = extract_links(html, url)
                for u in discovered:
                    if not u or not same_domain(u):
                        continue
                    ul = u.lower()
                    if any(x in ul for x in ["mailto:", "tel:", "javascript:"]):
                        continue
                    # Skip assets by ext early
                    if is_skip_asset(u):
                        discovered_all.add(u)
                        continue
                    discovered_all.add(u)
                    if u not in seen:
                        queue.append(u)

                # Path bucket
                top_paths[urlparse(url).path.split("/")[1] if len(urlparse(url).path.split("/")) > 1 else "root"] += 1

                stats["visited_total"] += 1
                time.sleep(crawl_delay_s)

            except Exception as e:
                stats["failed"] += 1
                failed_urls.append(url)
                continue

    # Save clean forms catalog
    forms_list = list(forms_catalog.values())
    # De-dup by URL (already keyed) and sort by title
    forms_list.sort(key=lambda x: (x.get("title", "").lower(), x.get("url", "")))
    FORMS_JSON.write_text(json.dumps(forms_list, ensure_ascii=False, indent=2), encoding="utf-8")

    # Coverage report
    # "missed" here means discovered but not crawled due to max_pages limit or being left in queue
    remaining_in_queue = list(dict.fromkeys(queue))
    missed_urls = [u for u in remaining_in_queue if u not in seen]

    report = {
        "base_url": BASE_URL,
        "max_pages_config": max_pages,
        "crawl_delay_s": crawl_delay_s,
        "stats": dict(stats),
        "unique_seen": len(seen),
        "unique_discovered": len(discovered_all),
        "top_path_buckets": dict(top_paths.most_common(25)),
        "forms_count": len(forms_list),
        "failed_urls_count": len(failed_urls),
        "failed_urls_sample": failed_urls[:50],
        "skipped_assets_count": len(skipped_assets),
        "skipped_assets_sample": skipped_assets[:50],
        "missed_urls_count": len(missed_urls),
        "missed_urls_sample": missed_urls[:100],
        "notes": [
            "missed_urls are URLs discovered but not crawled (often due to max_pages limit).",
            "skipped assets are images/css/js/media excluded intentionally.",
            "forms.json includes PDFs under /document/forms/ with extracted PDF titles."
        ]
    }
    COVERAGE_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n✅ Crawl complete")
    print(" - raw.jsonl:", OUT_JSONL)
    print(" - forms.json:", FORMS_JSON)
    print(" - coverage_report.json:", COVERAGE_JSON)


if __name__ == "__main__":
    crawl(max_pages=1500, crawl_delay_s=0.15)
