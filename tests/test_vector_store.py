import pytest
import chromadb
from chromadb.config import Settings

from chunker import chunk_pages
from embeddings import EMBEDDING_MODEL
from models import DocumentPage
from vector_store import (
    EMBEDDING_MODEL_KEY,
    ChromaVectorStore,
    VectorStoreError,
    document_id_for,
)


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
    reopened.ensure_embedding_model(
        lambda texts: (_ for _ in ()).throw(
            AssertionError("Verified vectors should not be re-embedded.")
        )
    )
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


def test_legacy_collection_is_reembedded_before_it_can_be_searched() -> None:
    client = chromadb.EphemeralClient(settings=Settings(allow_reset=True))
    client.reset()
    legacy = client.get_or_create_collection(
        name="college_documents",
        metadata={"hnsw:space": "cosine"},
    )
    legacy.add(
        ids=["legacy-id"],
        documents=["attendance requirement"],
        metadatas=[
            {
                "source": "legacy.pdf",
                "page_number": 1,
                "chunk_index": 0,
                "document_id": "legacy-document",
            }
        ],
        embeddings=[[1.0, 0.0]],
    )
    store = ChromaVectorStore(client=client)
    embedded_texts = []

    def reembed(texts):
        embedded_texts.extend(texts)
        return [[0.0, 1.0] for _ in texts]

    store.ensure_embedding_model(reembed)

    assert embedded_texts == ["attendance requirement"]
    stored = store.collection.get(include=["metadatas"])
    assert stored["metadatas"][0][EMBEDDING_MODEL_KEY] == EMBEDDING_MODEL
    assert store.search([0.0, 1.0])[0].distance == pytest.approx(0.0)
    assert store.collection.count() == 1


def test_empty_collection_is_tagged_without_requesting_embeddings() -> None:
    client = chromadb.EphemeralClient(settings=Settings(allow_reset=True))
    client.reset()
    store = ChromaVectorStore(client=client)

    def unexpected_embed(texts):
        raise AssertionError("An empty collection has no vectors to rebuild.")

    store.ensure_embedding_model(unexpected_embed)

    assert store.collection.count() == 0


def test_dimension_mismatch_keeps_legacy_collection_unsearchable() -> None:
    client = chromadb.EphemeralClient(settings=Settings(allow_reset=True))
    client.reset()
    legacy = client.get_or_create_collection(
        name="college_documents",
        metadata={"hnsw:space": "cosine"},
    )
    legacy.add(
        ids=["legacy-id"],
        documents=["legacy text"],
        metadatas=[
            {
                "source": "legacy.pdf",
                "page_number": 1,
                "chunk_index": 0,
                "document_id": "legacy-document",
            }
        ],
        embeddings=[[1.0, 0.0]],
    )
    store = ChromaVectorStore(client=client)

    with pytest.raises(VectorStoreError, match="incompatible embedding dimension"):
        store.ensure_embedding_model(lambda texts: [[1.0, 0.0, 0.0] for _ in texts])

    with pytest.raises(VectorStoreError, match="unverified or incompatible"):
        store.search([1.0, 0.0])
