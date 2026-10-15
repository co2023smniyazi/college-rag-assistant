import pytest

from chunker import chunk_pages, chunk_text
from models import DocumentPage


def test_chunk_text_uses_overlap_and_covers_text() -> None:
    text = "0123456789" * 3
    chunks = chunk_text(text, chunk_size=12, overlap=3)

    assert chunks == [text[:12], text[9:21], text[18:]]
    assert all(len(chunk) <= 12 for chunk in chunks)


def test_chunk_text_handles_empty_and_short_text() -> None:
    assert chunk_text("  ") == []
    assert chunk_text("short") == ["short"]


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [(0, 0), (10, -1), (10, 10)],
)
def test_chunk_text_rejects_invalid_sizes(chunk_size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size, overlap)


def test_chunk_pages_keeps_source_and_page_number() -> None:
    page = DocumentPage(source="rules.pdf", page_number=3, text="a" * 20)

    chunks = chunk_pages([page], chunk_size=10, overlap=2)

    assert [chunk.page_number for chunk in chunks] == [3, 3, 3]
    assert all(chunk.source == "rules.pdf" for chunk in chunks)
    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2]
