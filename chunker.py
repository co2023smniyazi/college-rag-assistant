from models import DocumentChunk, DocumentPage


DEFAULT_CHUNK_SIZE = 1500
DEFAULT_OVERLAP = 200


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size.")

    normalized = text.strip()
    if not normalized:
        return []

    chunks: list[str] = []
    step = chunk_size - overlap
    start = 0
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(normalized):
            break
        start += step
    return chunks


def chunk_pages(
    pages: list[DocumentPage],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    for page in pages:
        for chunk_index, text in enumerate(
            chunk_text(page.text, chunk_size=chunk_size, overlap=overlap)
        ):
            chunks.append(
                DocumentChunk(
                    source=page.source,
                    page_number=page.page_number,
                    chunk_index=chunk_index,
                    text=text,
                )
            )
    return chunks
