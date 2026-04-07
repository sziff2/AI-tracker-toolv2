"""
Document Processing / Parsing Service (§7)

Responsibilities:
  - Classify document type via LLM
  - Extract text from PDFs / DOCX
  - Detect and extract tables
  - Split into sections
  - Persist parsed results as JSON and DocumentSection rows
"""

import json
import logging
import uuid
from pathlib import Path

import fitz  # pymupdf
import pdfplumber

from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.models import Document, DocumentSection
from configs.settings import settings
from prompts import DOCUMENT_CLASSIFIER
from schemas import ClassifiedDocument
from services.llm_client import call_llm_json

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# Text extraction
# ─────────────────────────────────────────────────────────────────

def extract_text_pymupdf(file_path: str) -> list[dict]:
    """Return a list of {page, text} dicts using PyMuPDF."""
    doc = fitz.open(file_path)
    pages = []
    for i, page in enumerate(doc):
        pages.append({"page": i + 1, "text": page.get_text("text")})
    doc.close()
    return pages


def extract_tables_pdfplumber(file_path: str) -> list[dict]:
    """Return a list of {page, tables} using pdfplumber."""
    results = []
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            tables = page.extract_tables()
            if tables:
                results.append({"page": i + 1, "tables": tables})
    return results


def extract_text_docx(file_path: str) -> list[dict]:
    """Extract text from a .docx file."""
    from docx import Document as DocxDoc
    doc = DocxDoc(file_path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return [{"page": 1, "text": "\n".join(paragraphs)}]


def extract_text_html(file_path: str) -> tuple[list[dict], list[dict]]:
    """
    Extract text and tables from an HTML file (e.g. SEC EDGAR .htm filings).
    Returns (pages, tables) where pages are chunked by ~3000 chars for consistency
    with PDF page-based processing.
    """
    from bs4 import BeautifulSoup
    import re

    raw = Path(file_path).read_text(errors="replace")
    soup = BeautifulSoup(raw, "lxml")

    # Remove script, style, and hidden elements
    for tag in soup.find_all(["script", "style", "meta", "link", "head"]):
        tag.decompose()

    # Extract tables separately for structured data
    tables = []
    for i, table in enumerate(soup.find_all("table")):
        rows = []
        for tr in table.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
            if any(c for c in cells):
                rows.append(cells)
        if rows and len(rows) > 1:
            tables.append({"page": i + 1, "tables": [rows]})

    # Get clean text
    text = soup.get_text(separator="\n")
    # Clean up whitespace: collapse multiple blank lines, strip lines
    lines = [line.strip() for line in text.splitlines()]
    # Remove empty lines but keep paragraph breaks
    cleaned = []
    prev_empty = False
    for line in lines:
        if not line:
            if not prev_empty:
                cleaned.append("")
            prev_empty = True
        else:
            cleaned.append(line)
            prev_empty = False
    full_text = "\n".join(cleaned).strip()

    # Remove common SEC filing boilerplate
    full_text = re.sub(r'(?i)^\s*UNITED STATES\s*\n\s*SECURITIES AND EXCHANGE COMMISSION.*?FORM\s+\d+-[A-Z]+\s*\n', '', full_text, flags=re.DOTALL)

    # Split into page-like chunks (~3000 chars each for consistency with PDF processing)
    CHUNK_SIZE = 3000
    pages = []
    for i in range(0, len(full_text), CHUNK_SIZE):
        chunk = full_text[i:i + CHUNK_SIZE]
        # Try to break at a paragraph boundary
        if i + CHUNK_SIZE < len(full_text):
            last_break = chunk.rfind("\n\n")
            if last_break > CHUNK_SIZE * 0.5:
                chunk = full_text[i:i + last_break]
        pages.append({"page": len(pages) + 1, "text": chunk.strip()})

    if not pages:
        pages = [{"page": 1, "text": full_text}]

    logger.info("HTML extraction: %d chars, %d pages, %d tables from %s",
                len(full_text), len(pages), len(tables), file_path)
    return pages, tables


# ─────────────────────────────────────────────────────────────────
# Document classification (LLM)
# ─────────────────────────────────────────────────────────────────

def classify_document(text_preview: str, ticker: str = None) -> ClassifiedDocument:
    """Classify a document using the first ~2000 chars."""
    prompt = DOCUMENT_CLASSIFIER.format(text=text_preview[:2000])
    data = call_llm_json(prompt, feature="classification", model="claude-haiku-4-5-20251001", ticker=ticker)
    return ClassifiedDocument(**data)


# ─────────────────────────────────────────────────────────────────
# Full processing pipeline
# ─────────────────────────────────────────────────────────────────

async def process_document(db: AsyncSession, document: Document, ticker: str = "UNKNOWN") -> dict:
    """
    End-to-end processing of a single document:
      1. Extract text + tables
      2. Classify
      3. Split into sections and persist
      4. Write processed JSON files
    Returns a summary dict.
    """
    file_path = document.file_path
    ext = Path(file_path).suffix.lower()

    # Restore file if missing on disk (e.g. after Railway redeploy wipes ephemeral storage)
    if not Path(file_path).exists():
        # Try re-downloading from source_url
        source_url = getattr(document, 'source_url', None)
        if source_url:
            try:
                import httpx
                logger.info("File missing, re-downloading from %s", source_url[:80])
                headers = {"User-Agent": "Oldfield Partners research-bot@oldfieldpartners.com"}
                resp = httpx.get(source_url, headers=headers, timeout=60.0, follow_redirects=True)
                resp.raise_for_status()
                Path(file_path).parent.mkdir(parents=True, exist_ok=True)
                Path(file_path).write_bytes(resp.content)
                logger.info("Re-downloaded %d bytes to %s", len(resp.content), file_path)
            except Exception as dl_err:
                logger.warning("Re-download failed for %s: %s", source_url[:60], dl_err)

    if not Path(file_path).exists():
        raise FileNotFoundError(f"File not found and re-download failed: {file_path}")

    # 1. Extract
    if ext == ".pdf":
        pages = extract_text_pymupdf(file_path)
        tables = extract_tables_pdfplumber(file_path)
    elif ext in (".docx", ".doc"):
        pages = extract_text_docx(file_path)
        tables = []
    elif ext in (".htm", ".html", ".xhtml"):
        pages, tables = extract_text_html(file_path)
    else:
        # Plain text fallback
        raw_text = Path(file_path).read_text(errors="replace")
        # Chunk into pages for consistency
        CHUNK = 3000
        pages = [{"page": i + 1, "text": raw_text[i:i+CHUNK].strip()}
                 for i in range(0, len(raw_text), CHUNK)] or [{"page": 1, "text": raw_text}]
        tables = []

    full_text = "\n\n".join(p["text"] for p in pages)

    # Store parsed pages on document object so metric_extractor can use
    # real page boundaries for the pre-filter (avoids 3000-char fallback)
    document._parsed_pages = pages

    # 2. Classify
    classification = classify_document(full_text, ticker=ticker)

    # 3. Persist sections (strip null bytes — PostgreSQL rejects \x00 in text)
    for p in pages:
        clean_text = p["text"].replace("\x00", "") if p.get("text") else ""
        p["text"] = clean_text  # also clean for JSON export
        section = DocumentSection(
            id=uuid.uuid4(),
            document_id=document.id,
            section_type="page",
            section_title=f"Page {p['page']}",
            page_number=p["page"],
            text_content=clean_text,
        )
        db.add(section)

    # 4. Write processed JSON files
    proc_dir = Path(settings.storage_base_path) / "processed" / ticker / (document.period_label or "misc")
    proc_dir.mkdir(parents=True, exist_ok=True)

    (proc_dir / "parsed_text.json").write_text(json.dumps(pages, indent=2))
    (proc_dir / "tables.json").write_text(json.dumps(tables, indent=2))
    (proc_dir / "metadata.json").write_text(classification.model_dump_json(indent=2))

    # 5. Update status
    document.parsing_status = "completed"
    if classification.document_type and not document.document_type:
        document.document_type = classification.document_type
    await db.commit()

    logger.info("Processed document %s → %s", document.id, classification.document_type)
    return {
        "document_id": str(document.id),
        "pages": len(pages),
        "tables_found": sum(len(t.get("tables", [])) for t in tables),
        "classification": classification.model_dump(),
        "full_text": full_text,  # Return directly for parallel processing
        "tables_data": tables,
    }
