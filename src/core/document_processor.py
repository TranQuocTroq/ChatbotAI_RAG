"""
DocumentProcessor - Extracts text and metadata from PDF, DOCX, and TXT files using column-aware extraction.

Column-Aware PDF Extraction technique:
- Uses PyMuPDF (fitz) bounding box analysis to detect multi-column layout (academic papers, reports).
- Reads in correct flow order: full-width header -> left column -> right column -> full-width footer.
- Fallback: standard pdfplumber when fitz is unavailable.
"""
import re
import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def clean_pdf_text_artifacts(text: str) -> str:
    """Strips repeated character encoding artifacts commonly found in PDF text extraction."""
    if not text:
        return ""

    def fix_word(w: str) -> str:
        if len(w) >= 4 and len(w) % 2 == 0:
            if all(w[i] == w[i+1] for i in range(0, len(w), 2)):
                return "".join(w[i] for i in range(0, len(w), 2))
        return w

    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        words = line.split()
        cleaned_words = [fix_word(w) for w in words]
        cleaned_lines.append(" ".join(cleaned_words))

    return "\n".join(cleaned_lines)


def filter_pdf_header_footer(text: str, min_len: int = 20) -> str:
    """Filters out overly short repeated header/footer lines (such as standalone page numbers)."""
    lines = text.split("\n")
    filtered = []
    for l in lines:
        l = l.strip()
        if len(l) < min_len and re.match(r"^\d+$", l):
            continue
        filtered.append(l)
    return "\n".join(filtered)


def _extract_pdf_column_aware_fitz(path: Path, file_name: str) -> List[Dict[str, Any]]:
    """
    Column-Aware PDF Extraction using PyMuPDF (fitz) + pdfplumber for Tables.
    """
    import fitz  # PyMuPDF
    import pdfplumber

    doc = fitz.open(str(path))
    
    pdf_plumb = None
    try:
        pdf_plumb = pdfplumber.open(str(path))
    except Exception as e:
        logger.warning("[PDF Extraction] pdfplumber failed to open for table extraction: %s", e)

    pages_content = []

    for page_idx, page in enumerate(doc):
        rect = page.rect
        page_width = rect.width
        page_height = rect.height
        mid_x = page_width / 2.0

        blocks = page.get_text("blocks")
        text_blocks = [b for b in blocks if len(b) >= 6 and b[6] == 0 and b[4].strip()]

        full_width_top = []
        left_column = []
        right_column = []
        full_width_bottom = []

        for b in text_blocks:
            x0, y0, x1, y1, text = b[0], b[1], b[2], b[3], b[4]
            block_width = x1 - x0
            center_x = (x0 + x1) / 2.0

            if block_width > 0.60 * page_width:
                if y0 < page_height * 0.40:
                    full_width_top.append((y0, text))
                else:
                    full_width_bottom.append((y0, text))
            elif center_x < mid_x:
                left_column.append((y0, text))
            else:
                right_column.append((y0, text))

        full_width_top.sort(key=lambda x: x[0])
        left_column.sort(key=lambda x: x[0])
        right_column.sort(key=lambda x: x[0])
        full_width_bottom.sort(key=lambda x: x[0])

        ordered_texts = []
        for _, t in full_width_top:
            ordered_texts.append(t.strip())
        for _, t in left_column:
            ordered_texts.append(t.strip())
        for _, t in right_column:
            ordered_texts.append(t.strip())
        for _, t in full_width_bottom:
            ordered_texts.append(t.strip())

        # Markdown Table Extraction with pdfplumber
        md_tables_str = ""
        if pdf_plumb and page_idx < len(pdf_plumb.pages):
            try:
                plumb_page = pdf_plumb.pages[page_idx]
                tables = plumb_page.extract_tables()
                if tables:
                    md_tables = []
                    for table in tables:
                        if not table or not table[0]: continue
                        cleaned_table = []
                        for row in table:
                            # Clean cell text, replace newlines and pipes
                            cleaned_row = [str(cell).replace("\n", " ").replace("|", "").strip() if cell else "" for cell in row]
                            # Only add row if it's not completely empty
                            if any(cleaned_row):
                                cleaned_table.append(cleaned_row)
                                
                        if not cleaned_table: continue
                        
                        header = cleaned_table[0]
                        md_table = "| " + " | ".join(header) + " |\n"
                        md_table += "| " + " | ".join(["---"] * len(header)) + " |\n"
                        for row in cleaned_table[1:]:
                            if len(row) < len(header):
                                row.extend([""] * (len(header) - len(row)))
                            elif len(row) > len(header):
                                row = row[:len(header)]
                            md_table += "| " + " | ".join(row) + " |\n"
                        md_tables.append(md_table)
                    if md_tables:
                        md_tables_str = "\n\n[BẢNG SỐ LIỆU ĐƯỢC TRÍCH XUẤT]:\n" + "\n\n".join(md_tables)
            except Exception as e:
                logger.warning("[PDF Extraction] pdfplumber table extraction failed on page %d: %s", page_idx, e)

        page_text = "\n\n".join([t for t in ordered_texts if t]) + md_tables_str
        page_text = clean_pdf_text_artifacts(page_text)

        if page_text.strip():
            pages_content.append({
                "page": page_idx + 1,
                "text": page_text.strip(),
                "source": file_name,
                "layout": "column-aware+tables" if md_tables_str else "column-aware"
            })

    doc.close()
    if pdf_plumb:
        pdf_plumb.close()
    return pages_content


class DocumentProcessor:
    """Extracts text content and metadata from PDF, DOCX, and TXT files."""

    def __init__(self):
        pass

    def extract_text_with_pages(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Reads a document file and returns a list of page dicts with text and metadata.
        Output format: [{"page": 1, "text": "...", "source": "filename.pdf"}]
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        file_type = path.suffix.lower()
        file_name = path.name

        if file_type == ".pdf":
            return self._extract_pdf(path, file_name)
        elif file_type == ".docx":
            return self._extract_docx(path, file_name)
        elif file_type == ".txt":
            return self._extract_txt(path, file_name)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

    def _extract_pdf(self, path: Path, file_name: str) -> List[Dict[str, Any]]:
        """
        Extracts PDF text using Column-Aware layout analysis.
        Tries PyMuPDF (fitz) first, falling back to pdfplumber on failure.
        """
        try:
            pages = _extract_pdf_column_aware_fitz(path, file_name)
            if pages:
                logger.info("[PDF Extraction] Column-aware PyMuPDF succeeded for '%s' (%d pages)", file_name, len(pages))
                return pages
        except ImportError:
            logger.warning("[PDF Extraction] PyMuPDF (fitz) unavailable. Falling back to pdfplumber.")
        except Exception as e:
            logger.warning("[PDF Extraction] PyMuPDF failed: %s. Falling back to pdfplumber.", e)

        try:
            import pdfplumber
            pages_content = []
            with pdfplumber.open(path) as pdf:
                for idx, page in enumerate(pdf.pages):
                    raw_text = page.extract_text(layout=True) or page.extract_text() or ""
                    cleaned = clean_pdf_text_artifacts(raw_text.strip())
                    if cleaned:
                        pages_content.append({
                            "page": idx + 1,
                            "text": cleaned,
                            "source": file_name,
                            "layout": "pdfplumber"
                        })
            if pages_content:
                logger.info("[PDF Extraction] pdfplumber fallback succeeded for '%s'", file_name)
                return pages_content
        except Exception as e:
            logger.error("[PDF Extraction] pdfplumber failed: %s", e)

        raise RuntimeError(f"Failed to extract text from PDF: {file_name}")

    def _extract_docx(self, path: Path, file_name: str) -> List[Dict[str, Any]]:
        try:
            import docx
            doc = docx.Document(path)
            full_text = []
            for p in doc.paragraphs:
                if p.text.strip():
                    full_text.append(clean_pdf_text_artifacts(p.text.strip()))

            combined_text = "\n".join(full_text)
            return [{
                "page": 1,
                "text": combined_text,
                "source": file_name
            }]
        except Exception as e:
            raise RuntimeError(f"Failed to read DOCX file {file_name}: {e}")

    def _extract_txt(self, path: Path, file_name: str) -> List[Dict[str, Any]]:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read().strip()
            return [{
                "page": 1,
                "text": clean_pdf_text_artifacts(content),
                "source": file_name
            }]
        except Exception as e:
            raise RuntimeError(f"Failed to read TXT file {file_name}: {e}")
