from dataclasses import dataclass


@dataclass(frozen=True)
class DocumentPage:
    source: str
    page_number: int
    text: str


@dataclass(frozen=True)
class DocumentChunk:
    source: str
    page_number: int
    chunk_index: int
    text: str


@dataclass(frozen=True)
class RetrievedChunk:
    source: str
    page_number: int
    text: str
    distance: float
