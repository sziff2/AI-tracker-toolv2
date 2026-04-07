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


def _strip_xbrl(html: str) -> str:
    """Strip inline XBRL wrapper tags (ix:...) that bloat SEC filings.
    These wrap every number/text in verbose XML but contain the same content."""
    import re
    # Remove ix: opening tags but keep content: <ix:nonFraction ...>123</ix:nonFraction> → 123
    html = re.sub(r'<ix:[^>]*>', '', html)
    html = re.sub(r'</ix:[^>]*>', '', html)
    # Remove xbrli/link/context blocks (XBRL metadata, not visible content)
    html = re.sub(r'<xbrli:context[^>]*>.*?</xbrli:context>', '', html, flags=re.DOTALL)
    html = re.sub(r'<link:[^>]*>.*?</link:[^>]*>', '', html, flags=re.DOTALL)
    html = re.sub(r'<xbrli:[^>]*>.*?</xbrli:[^>]*>', '', html, flags=re.DOTALL)
    # Remove XML namespace declarations that bloat the file
    html = re.sub(r'\s+xmlns:[a-z\-]+="[^"]*"', '', html)
    return html


def _extract_sec_sections(html: str) -> dict:
    """Split SEC filing HTML into known sections for targeted extraction.
    Returns dict with keys like 'financial_statements', 'mdna', 'risk_factors', etc."""
    import re
    sections = {}
    # SEC filings use Part I/II and Item N headings
    # Find financial statements section (Item 1 in 10-Q, Item 8 in 10-K)
    for pattern in [
        r'(?i)(?:Item\s+1[.\s]*Financial\s+Statements)(.*?)(?=Item\s+[2-9]|Part\s+II|$)',
        r'(?i)(?:Item\s+8[.\s]*Financial\s+Statements)(.*?)(?=Item\s+9|Part\s+[IVX]|$)',
        r'(?i)(?:CONSOLIDATED\s+(?:STATEMENTS?|BALANCE))(.*?)(?=Item\s+\d|Part\s+[IVX]|SIGNATURES|$)',
    ]:
        m = re.search(pattern, html, re.DOTALL)
        if m and len(m.group(1)) > 1000:
            sections['financial_statements'] = m.group(1)
            break
    # MD&A section
    for pattern in [
        r'(?i)(?:Item\s+2[.\s]*Management.s\s+Discussion)(.*?)(?=Item\s+[3-9]|Part\s+II|$)',
        r'(?i)(?:Item\s+7[.\s]*Management.s\s+Discussion)(.*?)(?=Item\s+[89]|Part\s+[IVX]|$)',
    ]:
        m = re.search(pattern, html, re.DOTALL)
        if m and len(m.group(1)) > 500:
            sections['mdna'] = m.group(1)
            break
    return sections


def extract_text_html(file_path: str) -> tuple[list[dict], list[dict]]:
    """
    Extract text and tables from an HTML file (e.g. SEC EDGAR .htm filings).
    Optimised for large SEC filings (10-K can be 8MB+):
    1. Strip XBRL tags first (reduces size 50-80%)
    2. Extract financial sections only for very large files
    3. Parse tables via regex for speed, BeautifulSoup for text
    """
    from bs4 import BeautifulSoup
    import re

    raw = Path(file_path).read_text(errors="replace")
    file_size = len(raw)
    logger.info("HTML extraction starting: %d bytes from %s", file_size, file_path)

    # Step 1: Strip XBRL if present (reduces 8MB → ~2MB typically)
    if 'xmlns:ix=' in raw[:5000] or '<ix:' in raw[:5000]:
        raw = _strip_xbrl(raw)
        logger.info("Stripped XBRL: %d → %d bytes (%.0f%% reduction)",
                     file_size, len(raw), (1 - len(raw)/file_size) * 100)

    # Step 2: For very large files, extract only financial sections
    working_html = raw
    if len(raw) > 2_000_000:  # > 2MB after XBRL strip
        sections = _extract_sec_sections(raw)
        if sections:
            # Combine financial statements + MD&A for extraction
            working_html = sections.get('financial_statements', '')
            if 'mdna' in sections:
                working_html += '\n\n' + sections['mdna']
            logger.info("Large file: extracted %d chars from %d sections (financial + MD&A)",
                         len(working_html), len(sections))
        if len(working_html) < 1000:
            # Section extraction failed — fall back to full but truncated
            working_html = raw[:2_000_000]
            logger.warning("Section extraction found too little content, using first 2MB")

    # Step 3: Extract tables via regex (faster than BeautifulSoup for large files)
    tables = []
    table_matches = list(re.finditer(r'<table[^>]*>(.*?)</table>', working_html, re.DOTALL | re.IGNORECASE))
    for i, m in enumerate(table_matches):
        table_html = m.group(0)
        rows = []
        for tr_match in re.finditer(r'<tr[^>]*>(.*?)</tr>', table_html, re.DOTALL | re.IGNORECASE):
            cells = re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>', tr_match.group(1), re.DOTALL | re.IGNORECASE)
            # Strip HTML from cell content
            cleaned_cells = [re.sub(r'<[^>]+>', '', c).strip() for c in cells]
            if any(c for c in cleaned_cells):
                rows.append(cleaned_cells)
        if rows and len(rows) > 1:
            tables.append({"page": i + 1, "tables": [rows]})

    # Step 4: Extract text — try BeautifulSoup, fall back to regex strip
    try:
        soup = BeautifulSoup(working_html, "html.parser")  # html.parser handles partial HTML better than lxml
        for tag in soup.find_all(["script", "style", "meta", "link", "head"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
    except Exception as e:
        logger.warning("BeautifulSoup failed, using regex text extraction: %s", str(e)[:100])
        text = ""

    # Fallback: if BeautifulSoup produced nothing, strip HTML tags via regex
    if len(text.strip()) < 100:
        logger.info("BeautifulSoup produced %d chars, falling back to regex strip", len(text.strip()))
        text = re.sub(r'<[^>]+>', ' ', working_html)
        text = re.sub(r'&nbsp;|&amp;|&lt;|&gt;|&#\d+;|&#x[0-9a-fA-F]+;', ' ', text)
        text = re.sub(r'\s+', ' ', text)

    lines = [line.strip() for line in text.splitlines()]
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

    # Remove SEC boilerplate
    full_text = re.sub(r'(?i)^\s*UNITED STATES\s*\n\s*SECURITIES AND EXCHANGE COMMISSION.*?FORM\s+\d+-[A-Z]+\s*\n', '', full_text, flags=re.DOTALL)

    # Split into page-like chunks
    CHUNK_SIZE = 3000
    pages = []
    pos = 0
    while pos < len(full_text):
        chunk = full_text[pos:pos + CHUNK_SIZE]
        if pos + CHUNK_SIZE < len(full_text):
            last_break = chunk.rfind("\n\n")
            if last_break > CHUNK_SIZE * 0.5:
                chunk = full_text[pos:pos + last_break]
        pages.append({"page": len(pages) + 1, "text": chunk.strip()})
        pos += len(chunk)

    if not pages:
        pages = [{"page": 1, "text": full_text}]

    logger.info("HTML extraction: %d chars, %d pages, %d tables from %s (original %d bytes)",
                len(full_text), len(pages), len(tables), file_path, file_size)
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
