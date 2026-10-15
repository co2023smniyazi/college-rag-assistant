from io import BytesIO
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from models import DocumentPage


class DocumentProcessingError(ValueError):
    """Raised when a PDF cannot be used as a text source."""


def extract_pdf_pages(pdf_bytes: bytes, filename: str) -> list[DocumentPage]:
    if Path(filename).suffix.lower() != ".pdf":
        raise DocumentProcessingError("Please upload a PDF file.")
    if not pdf_bytes:
        raise DocumentProcessingError("The uploaded PDF is empty.")

    try:
        reader = PdfReader(BytesIO(pdf_bytes), strict=False)
        if reader.is_encrypted:
            raise DocumentProcessingError(
                "This PDF is password-protected. Upload an unlocked PDF."
            )
        if not reader.pages:
            raise DocumentProcessingError("The PDF contains no pages.")

        pages = [
            DocumentPage(
                source=Path(filename).name,
                page_number=page_number,
                text=text,
            )
            for page_number, page in enumerate(reader.pages, start=1)
            if (text := (page.extract_text() or "").strip())
        ]
    except DocumentProcessingError:
        raise
    except (PdfReadError, OSError, ValueError) as error:
        raise DocumentProcessingError(
            "The PDF could not be read. Check that it is a valid PDF file."
        ) from error

    if not pages:
        raise DocumentProcessingError(
            "No selectable text was found. Scanned or image-only PDFs need OCR, "
            "which this application does not provide."
        )
    return pages
