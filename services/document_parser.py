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


# ─────────────────────────────────────────────────────────────────
# Document classification (LLM)
# ─────────────────────────────────────────────────────────────────

def classify_document(text_preview: str) -> ClassifiedDocument:
    """Classify a document using the first ~2000 chars."""
    prompt = DOCUMENT_CLASSIFIER.format(text=text_preview[:2000])
    data = call_llm_json(prompt)
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

    # Restore file from DB if missing on disk (e.g. after Railway redeploy)
    if not Path(file_path).exists() and document.file_content:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        Path(file_path).write_bytes(document.file_content)
        logger.info("Restored file from DB: %s", file_path)

    if not Path(file_path).exists():
        raise FileNotFoundError(f"File not found and no DB backup: {file_path}")

    # 1. Extract
    if ext == ".pdf":
        pages = extract_text_pymupdf(file_path)
        tables = extract_tables_pdfplumber(file_path)
    elif ext in (".docx", ".doc"):
        pages = extract_text_docx(file_path)
        tables = []
    else:
        pages = [{"page": 1, "text": Path(file_path).read_text(errors="replace")}]
        tables = []

    full_text = "\n\n".join(p["text"] for p in pages)

    # 2. Classify
    classification = classify_document(full_text)

    # 3. Persist sections
    for p in pages:
        section = DocumentSection(
            id=uuid.uuid4(),
            document_id=document.id,
            section_type="page",
            section_title=f"Page {p['page']}",
            page_number=p["page"],
            text_content=p["text"],
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
    }
