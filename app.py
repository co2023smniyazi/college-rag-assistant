from pathlib import Path

import streamlit as st

from app_services import build_rag_service
from chunker import chunk_pages
from document_loader import DocumentProcessingError, extract_pdf_pages
from embeddings import EmbeddingServiceError, OllamaEmbedder
from llm_service import LLMServiceError
from rag_service import RAGService
from vector_store import ChromaVectorStore, VectorStoreError, document_id_for


PROJECT_DIR = Path(__file__).resolve().parent
CHROMA_PATH = PROJECT_DIR / "chroma_db"


@st.cache_resource
def get_services() -> tuple[ChromaVectorStore, OllamaEmbedder, RAGService]:
    return build_rag_service(CHROMA_PATH)


st.set_page_config(page_title="College Document Q&A Assistant", page_icon="📚")
st.title("College Document Q&A Assistant")
st.caption(
    "Ask questions about your PDFs. Answers use only retrieved document passages; "
    "sources include the document name and page."
)

try:
    store, embedder, rag_service = get_services()
except VectorStoreError as error:
    st.error(str(error))
    st.stop()

with st.sidebar:
    st.header("Add college documents")
    uploads = st.file_uploader(
        "Choose text-based PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Scanned/image-only PDFs are not supported because OCR is not enabled.",
    )
    if st.button("Index uploaded PDFs", disabled=not uploads):
        for uploaded_file in uploads or []:
            raw_bytes = uploaded_file.getvalue()
            identifier = document_id_for(raw_bytes)
            try:
                if store.has_document(identifier):
                    st.info(f"{uploaded_file.name} is already indexed.")
                    continue
                with st.spinner(f"Reading and indexing {uploaded_file.name}..."):
                    pages = extract_pdf_pages(raw_bytes, uploaded_file.name)
                    chunks = chunk_pages(pages)
                    vectors = embedder.embed_texts([chunk.text for chunk in chunks])
                    store.add_chunks(identifier, chunks, vectors)
            except (
                DocumentProcessingError,
                EmbeddingServiceError,
                VectorStoreError,
                ValueError,
            ) as error:
                st.error(f"{uploaded_file.name}: {error}")
                continue
            st.success(
                f"Indexed {uploaded_file.name}: {len(pages)} readable page(s), "
                f"{len(chunks)} chunk(s)."
            )

    st.divider()
    st.header("Indexed documents")
    try:
        indexed_documents = store.list_documents()
    except VectorStoreError as error:
        st.error(str(error))
        indexed_documents = None
    if indexed_documents == []:
        st.caption("No documents indexed yet.")
    elif indexed_documents is not None:
        for document in indexed_documents:
            st.write(f"📄 {document['source']}")
        document_labels = {
            f"{item['source']} · {item['document_id'][:8]}": item["document_id"]
            for item in indexed_documents
        }
        selected_label = st.selectbox(
            "Choose a document to remove",
            options=list(document_labels),
            label_visibility="collapsed",
        )
        if st.button("Remove selected document", type="secondary"):
            try:
                store.delete_document(document_labels[selected_label])
            except VectorStoreError as error:
                st.error(str(error))
            else:
                st.success(f"Removed {selected_label.split(' · ')[0]}.")
                st.rerun()

st.subheader("Ask a question")
with st.form("question_form"):
    question = st.text_input(
        "Your question",
        placeholder="For example: What does the document say about attendance?",
    )
    submitted = st.form_submit_button("Ask")

if submitted:
    if not question.strip():
        st.warning("Enter a question before asking.")
    else:
        try:
            with st.spinner("Searching your documents and preparing an answer..."):
                answer, sources = rag_service.ask(question)
        except (
            EmbeddingServiceError,
            LLMServiceError,
            VectorStoreError,
            ValueError,
        ) as error:
            st.error(str(error))
        else:
            st.subheader("Answer")
            if not sources:
                st.warning(answer)
                st.caption(
                    "No sufficiently relevant indexed passage was found. "
                    "Try a different question or check the uploaded documents."
                )
            else:
                st.write(answer)
                st.subheader("Retrieved sources")
                for source in sources:
                    st.markdown(
                        f"- **{source.source}**, page {source.page_number} "
                        f"(retrieval distance: {source.distance:.3f})"
                    )
                    with st.expander("View supporting passage"):
                        st.write(source.text)
