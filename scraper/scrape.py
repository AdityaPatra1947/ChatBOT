"""Polite crawler for PUBLIC VIT Mumbai pages. No logins, respects robots.txt.

Output: data/raw/pages.json  -> [{"url", "title", "text", "type"}]
"""
import io
import json
import re
import time
from collections import deque
from pathlib import Path
from urllib.parse import urljoin, urldefrag, urlparse
from urllib.robotparser import RobotFileParser

import numpy as np
import pypdfium2 as pdfium
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
from rapidocr_onnxruntime import RapidOCR

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "raw" / "pages.json"

ALLOWED_HOSTS = {"vit.edu.in", "www.vit.edu.in", "old.vit.edu.in"}
SEEDS = [
    "https://vit.edu.in/",
    "https://vit.edu.in/admissions/",
    "https://vit.edu.in/admission-engineering-ug/",
    "https://vit.edu.in/admission-engineering-lateral-entry/",
    "https://vit.edu.in/admission-engineering-pg/",
    "https://vit.edu.in/admission-mms/",
    "https://vit.edu.in/admission-phd/",
    "https://vit.edu.in/contact/",
    "https://vit.edu.in/placement/",
    "https://vit.edu.in/scholarship/",
    "https://vit.edu.in/wp-content/uploads/2026/07/VIT-Admission-Brochure-2026.pdf",
    "https://vit.edu.in/wp-content/uploads/2026/08/FRA-Fees.pdf",
    "https://vit.edu.in/wp-content/uploads/2026/09/Rules-of-Refund-of-Fees-after-cancellation-of-admission-by-Candidate.pdf",
    "https://www.vit.edu.in/images/Admissions 2024-25/Frequently Asked Questions Related to Admissions.pdf",
]
SKIP = re.compile(
    r"(wp-login|wp-admin|/feed|/tag/|/author/|\?replytocom|login|erp|lms|"
    r"\.(jpg|jpeg|png|gif|svg|webp|zip|mp4|docx?|xlsx?|pptx?)$)",
    re.I,
)
# Documents listing individual students (names, marks) - not indexed for privacy.
PERSONAL = re.compile(r"(merit|selected|students?(%20|[-_ ])+placed|placed(%20|[-_ ])*list)", re.I)
MAX_PAGES = 350
DELAY = 1.0
HEADERS = {"User-Agent": "VIT-Enquiry-Chatbot-Student-Project/1.0"}

robots = RobotFileParser()
try:
    robots.set_url("https://vit.edu.in/robots.txt")
    robots.read()
except Exception:
    robots = None


def allowed(url: str) -> bool:
    if urlparse(url).netloc not in ALLOWED_HOSTS or SKIP.search(url) or PERSONAL.search(url):
        return False
    return robots.can_fetch(HEADERS["User-Agent"], url) if robots else True


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def parse_html(url, html):
    soup = BeautifulSoup(html, "lxml")
    for t in soup(["script", "style", "nav", "footer", "header", "noscript", "form", "iframe"]):
        t.decompose()
    title = clean(soup.title.get_text()) if soup.title else url
    main = soup.find("main") or soup.find("article") or soup.body or soup
    text = clean(main.get_text(" "))
    links = [urldefrag(urljoin(url, a["href"]))[0] for a in soup.find_all("a", href=True)]
    return title, text, links


_ocr = None


def ocr_pdf(content, max_pages=6):
    """OCR for scanned PDFs (most admission/fee notices are images)."""
    global _ocr
    _ocr = _ocr or RapidOCR()
    lines = []
    for i, page in enumerate(pdfium.PdfDocument(content)):
        if i >= max_pages:
            break
        img = np.array(page.render(scale=2.5).to_pil().convert("RGB"))
        result, _ = _ocr(img)
        rows = {}
        for box, text, _conf in result or []:
            y = round((box[0][1] + box[2][1]) / 2 / 25)
            rows.setdefault(y, []).append((box[0][0], text))
        lines += [" | ".join(t for _, t in sorted(rows[y])) for y in sorted(rows)]
    return clean(" . ".join(lines))


def parse_pdf(content):
    reader = PdfReader(io.BytesIO(content))
    text = clean(" ".join((p.extract_text() or "") for p in reader.pages))
    if len(text) < 80:
        text = ocr_pdf(content)
    return text


def main():
    queue, seen, pages = deque(SEEDS), set(), []
    session = requests.Session()
    session.headers.update(HEADERS)
    while queue and len(pages) < MAX_PAGES:
        url = queue.popleft()
        if url in seen or not allowed(url):
            continue
        seen.add(url)
        try:
            r = session.get(url, timeout=25)
            if r.status_code != 200:
                continue
            ctype = r.headers.get("content-type", "")
            if "pdf" in ctype or url.lower().endswith(".pdf"):
                text = parse_pdf(r.content)
                title, kind, links = url.rsplit("/", 1)[-1], "pdf", []
            elif "html" in ctype:
                title, text, links = parse_html(url, r.text)
                kind = "html"
            else:
                continue
        except Exception as e:
            print("FAIL", url, e)
            continue
        if len(text) > 80:
            pages.append({"url": url, "title": title, "text": text, "type": kind})
            print(f"[{len(pages)}] {kind} {url} ({len(text)} chars)")
        new = [l for l in links if l not in seen]
        if url in SEEDS:
            # Documents linked from key pages (fees, schedules, guidelines) go first.
            queue.extendleft(reversed([l for l in new if l.lower().endswith(".pdf")]))
            queue.extend(l for l in new if not l.lower().endswith(".pdf"))
        else:
            queue.extend(new)
        time.sleep(DELAY)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(pages, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Saved {len(pages)} pages -> {OUT}")


if __name__ == "__main__":
    main()
