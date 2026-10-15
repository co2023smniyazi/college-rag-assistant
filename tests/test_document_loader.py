import pytest

from document_loader import (
    MAX_PDF_SIZE_BYTES,
    DocumentProcessingError,
    extract_pdf_pages,
)


def test_extract_pdf_pages_preserves_filename_and_page(make_text_pdf) -> None:
    pdf = make_text_pdf(["attendance is 75%", "", "registration closes soon"])

    pages = extract_pdf_pages(pdf, "college_rules.pdf")

    assert [(page.source, page.page_number) for page in pages] == [
        ("college_rules.pdf", 1),
        ("college_rules.pdf", 3),
    ]
    assert "75%" in pages[0].text


@pytest.mark.parametrize(
    ("pdf_bytes", "filename", "message"),
    [
        (b"not a pdf", "rules.pdf", "could not be read"),
        (b"", "rules.pdf", "empty"),
        (b"data", "rules.txt", "PDF file"),
    ],
)
def test_extract_pdf_pages_reports_invalid_uploads(
    pdf_bytes: bytes, filename: str, message: str
) -> None:
    with pytest.raises(DocumentProcessingError, match=message):
        extract_pdf_pages(pdf_bytes, filename)


def test_extract_pdf_pages_rejects_image_only_pdf(make_text_pdf) -> None:
    pdf = make_text_pdf([""])

    with pytest.raises(DocumentProcessingError, match="OCR"):
        extract_pdf_pages(pdf, "scan.pdf")


def test_extract_pdf_pages_rejects_oversized_upload() -> None:
    with pytest.raises(DocumentProcessingError, match="20 MB"):
        extract_pdf_pages(b"x" * (MAX_PDF_SIZE_BYTES + 1), "large.pdf")
