import pytest

from chunker import chunk_pages
from models import DocumentPage
from vector_store import ChromaVectorStore, document_id_for


def test_chroma_persists_chunks_and_source_metadata(tmp_path) -> None:
    store = ChromaVectorStore(path=tmp_path / "chroma")
    source_bytes = b"fictional handbook bytes"
    document_id = document_id_for(source_bytes)
    chunks = chunk_pages(
        [DocumentPage("handbook.pdf", 4, "attendance policy")],
    )

    store.add_chunks(document_id, chunks, [[1.0, 0.0, 0.0]])

    assert store.has_document(document_id)
    assert store.list_documents() == [
        {"document_id": document_id, "source": "handbook.pdf"}
    ]
    result = store.search([1.0, 0.0, 0.0])
    assert len(result) == 1
    assert result[0].source == "handbook.pdf"
    assert result[0].page_number == 4
    assert result[0].text == "attendance policy"

    reopened = ChromaVectorStore(path=tmp_path / "chroma")
    persisted = reopened.search([1.0, 0.0, 0.0])
    assert persisted[0].source == "handbook.pdf"
    assert persisted[0].page_number == 4


def test_chroma_removes_all_chunks_for_a_document(tmp_path) -> None:
    store = ChromaVectorStore(path=tmp_path / "chroma")
    document_id = document_id_for(b"one pdf")
    chunks = chunk_pages(
        [DocumentPage("remove.pdf", 1, "text " * 500)],
        chunk_size=100,
        overlap=10,
    )
    store.add_chunks(document_id, chunks, [[1.0, 0.0]] * len(chunks))

    store.delete_document(document_id)

    assert not store.has_document(document_id)
    assert store.search([1.0, 0.0]) == []


def test_identical_text_on_different_pages_keeps_both_sources(tmp_path) -> None:
    store = ChromaVectorStore(path=tmp_path / "chroma")
    document_id = document_id_for(b"same page contents in test PDF")
    chunks = chunk_pages(
        [
            DocumentPage("handbook.pdf", 1, "same paragraph"),
            DocumentPage("handbook.pdf", 2, "same paragraph"),
        ]
    )

    store.add_chunks(document_id, chunks, [[1.0, 0.0]] * len(chunks))

    assert {item.page_number for item in store.search([1.0, 0.0])} == {1, 2}


def test_document_id_is_stable_and_content_based() -> None:
    assert document_id_for(b"same") == document_id_for(b"same")
    assert document_id_for(b"same") != document_id_for(b"different")


def test_reindexing_same_document_does_not_duplicate_chunks(tmp_path) -> None:
    store = ChromaVectorStore(path=tmp_path / "chroma")
    document_id = document_id_for(b"same PDF")
    chunks = chunk_pages([DocumentPage("same.pdf", 1, "repeated content")])

    store.add_chunks(document_id, chunks, [[1.0, 0.0]])
    store.add_chunks(document_id, chunks, [[1.0, 0.0]])

    assert store.collection.count() == 1


def test_store_rejects_inconsistent_embedding_dimensions(tmp_path) -> None:
    store = ChromaVectorStore(path=tmp_path / "chroma")
    chunks = chunk_pages(
        [
            DocumentPage("dimensions.pdf", 1, "first chunk"),
            DocumentPage("dimensions.pdf", 2, "second chunk"),
        ]
    )

    with pytest.raises(ValueError, match="same dimension"):
        store.add_chunks(document_id_for(b"dimensions"), chunks, [[1.0], [1.0, 2.0]])
