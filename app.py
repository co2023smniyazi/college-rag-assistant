import html
import logging
from pathlib import Path

import streamlit as st

from app_services import (
    build_rag_service,
    get_ollama_status,
    model_is_available,
    resolve_ollama_host,
)
from chunker import chunk_pages
from document_loader import (
    DocumentProcessingError,
    extract_pdf_pages,
)
from embeddings import EMBEDDING_MODEL, EmbeddingServiceError, OllamaEmbedder
from llm_service import CHAT_MODEL, LLMServiceError, UNKNOWN_ANSWER
from rag_service import RAGService
from retriever import DEFAULT_RESULT_COUNT
from vector_store import ChromaVectorStore, VectorStoreError, document_id_for


PROJECT_DIR = Path(__file__).resolve().parent
CHROMA_PATH = PROJECT_DIR / "chroma_db"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def get_configured_ollama_host() -> str | None:
    try:
        return resolve_ollama_host(st.secrets)
    except FileNotFoundError:
        return resolve_ollama_host()


@st.cache_resource
def get_services(
    ollama_host: str | None,
) -> tuple[ChromaVectorStore, OllamaEmbedder, RAGService]:
    return build_rag_service(CHROMA_PATH, ollama_host=ollama_host)


@st.cache_data(ttl=30)
def get_cached_ollama_status(ollama_host: str | None) -> tuple[bool, set[str], str]:
    return get_ollama_status(ollama_host=ollama_host)


def render_status_card(
    label: str, value: str, note: str, ready: bool
) -> None:
    card_class = "ok" if ready else "warn"
    st.markdown(
        f"""
        <div class="status-card {card_class}">
          <div class="status-top"><span class="status-dot"></span>{html.escape(label)}</div>
          <div class="status-value">{html.escape(value)}</div>
          <div class="status-note">{html.escape(note)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(page_title="College Document Q&A Assistant", page_icon="📚")
st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(ellipse at 8% 0%, rgba(99, 102, 241, .08), transparent 32rem),
            #f6f8fc;
        color: #192342;
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stMainBlockContainer"] {
        max-width: 1120px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f0f2ff 0%, #f8faff 100%);
        border-right: 1px solid #e4e8f3;
    }
    [data-testid="stSidebar"] h2 {
        color: #242d55;
        letter-spacing: -.03em;
    }
    .hero {
        position: relative;
        overflow: hidden;
        padding: 2rem 2.2rem;
        margin-bottom: 1.45rem;
        border-radius: 24px;
        color: white;
        background: linear-gradient(118deg, #252b67 0%, #3942a4 53%, #167b83 120%);
        box-shadow: 0 18px 44px rgba(41, 53, 126, .18);
    }
    .hero:after {
        content: "";
        position: absolute;
        width: 260px;
        height: 260px;
        right: -64px;
        top: -128px;
        border: 1px solid rgba(255,255,255,.18);
        border-radius: 50%;
        box-shadow: 0 0 0 28px rgba(255,255,255,.04), 0 0 0 58px rgba(255,255,255,.035);
    }
    .hero-kicker {
        display: inline-block;
        margin-bottom: .7rem;
        padding: .34rem .7rem;
        border: 1px solid rgba(255,255,255,.24);
        border-radius: 999px;
        background: rgba(255,255,255,.1);
        color: #d9f7f1;
        font-size: .7rem;
        font-weight: 750;
        letter-spacing: .12em;
    }
    .hero h1 {
        margin: 0;
        color: #fff;
        font-size: clamp(1.9rem, 4vw, 2.65rem);
        letter-spacing: -.045em;
        line-height: 1.12;
    }
    .hero p {
        max-width: 650px;
        margin: .75rem 0 0;
        color: rgba(255,255,255,.82);
        font-size: 1rem;
        line-height: 1.65;
    }
    .section-heading {
        margin: 1.45rem 0 .75rem;
        color: #263151;
        font-size: 1.17rem;
        font-weight: 750;
        letter-spacing: -.025em;
    }
    .section-hint {
        margin: -.4rem 0 1rem;
        color: #68738d;
        font-size: .91rem;
    }
    .status-card {
        min-height: 105px;
        padding: 1rem 1.05rem;
        border: 1px solid #e5eaf2;
        border-radius: 16px;
        background: rgba(255,255,255,.9);
        box-shadow: 0 5px 18px rgba(37, 48, 86, .045);
    }
    .status-top {
        display: flex;
        align-items: center;
        gap: .55rem;
        margin-bottom: .5rem;
        color: #65708a;
        font-size: .78rem;
        font-weight: 650;
    }
    .status-dot {
        width: 9px;
        height: 9px;
        border-radius: 50%;
        background: #10a878;
        box-shadow: 0 0 0 4px rgba(16,168,120,.11);
    }
    .status-card.warn .status-dot {
        background: #e5a129;
        box-shadow: 0 0 0 4px rgba(229,161,41,.13);
    }
    .status-value {
        color: #202a49;
        font-size: .98rem;
        font-weight: 750;
        line-height: 1.35;
    }
    .status-note {
        margin-top: .3rem;
        color: #79839a;
        font-size: .76rem;
    }
    [data-testid="stForm"] {
        padding: 1.25rem 1.35rem 1rem;
        border: 1px solid #e5eaf2;
        border-radius: 18px;
        background: rgba(255,255,255,.92);
        box-shadow: 0 8px 26px rgba(37, 48, 86, .055);
    }
    [data-testid="stTextInput"] input {
        min-height: 3rem;
        border-color: #dce2ee;
        border-radius: 11px;
        background: #fbfcff;
    }
    [data-testid="stTextInput"] input:focus {
        border-color: #7771ee;
        box-shadow: 0 0 0 3px rgba(79,70,229,.12);
    }
    .stButton > button, [data-testid="stFormSubmitButton"] button {
        min-height: 2.75rem;
        border-radius: 10px;
        font-weight: 700;
        transition: transform .15s ease, box-shadow .15s ease;
    }
    [data-testid="stFormSubmitButton"] button[kind="primary"] {
        border: 0;
        background: linear-gradient(105deg, #5148df, #4262d9);
        box-shadow: 0 6px 14px rgba(79,70,229,.2);
    }
    .stButton > button:hover, [data-testid="stFormSubmitButton"] button:hover {
        transform: translateY(-1px);
    }
    .document-row {
        margin: .35rem 0;
        padding: .65rem .75rem;
        overflow-wrap: anywhere;
        border: 1px solid #e7eaf3;
        border-radius: 10px;
        background: rgba(255,255,255,.78);
        color: #34405e;
        font-size: .86rem;
    }
    [data-testid="stVerticalBlockBorderWrapper"]:has(.answer-label) {
        border: 1px solid #e3e8f2;
        border-left: 4px solid #4f46e5;
        border-radius: 16px;
        background: #fff;
        box-shadow: 0 8px 24px rgba(37, 48, 86, .055);
    }
    [data-testid="stVerticalBlockBorderWrapper"]:has(.answer-label) > div {
        padding: .5rem .7rem;
    }
    .answer-label {
        margin-bottom: .55rem;
        color: #5149c9;
        font-size: .76rem;
        font-weight: 800;
        letter-spacing: .1em;
        text-transform: uppercase;
    }
    .source-chip {
        display: inline-block;
        margin: .2rem .3rem .2rem 0;
        padding: .4rem .65rem;
        border: 1px solid #dfe5f3;
        border-radius: 999px;
        background: #f4f6ff;
        color: #38436b;
        font-size: .8rem;
        font-weight: 650;
    }
    [data-testid="stExpander"] {
        border-color: #e5eaf2;
        border-radius: 12px;
        background: rgba(255,255,255,.72);
    }
    @media (max-width: 700px) {
        [data-testid="stMainBlockContainer"] { padding: 1rem 1rem 2rem; }
        .hero { padding: 1.5rem; border-radius: 18px; }
        .status-card { min-height: 90px; padding: .8rem; }
    }
    @media (prefers-reduced-motion: reduce) {
        *, *:before, *:after { transition-duration: .01ms !important; }
    }
    </style>
    <section class="hero">
      <div class="hero-kicker">DOCUMENT-GROUNDED · OLLAMA-POWERED</div>
      <h1>College Knowledge, made easier.</h1>
      <p>Ask your college PDFs a question. Get a clear answer grounded in the
      documents, with the source page right there to check.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

ollama_host = get_configured_ollama_host()
try:
    store, embedder, rag_service = get_services(ollama_host)
except VectorStoreError as error:
    st.error(str(error))
    st.stop()

ollama_connected, installed_models, ollama_message = get_cached_ollama_status(
    ollama_host
)
chat_model_available = model_is_available(CHAT_MODEL, installed_models)
embedding_model_available = model_is_available(EMBEDDING_MODEL, installed_models)
try:
    indexed_count = len(store.list_documents())
    chroma_ready = True
    chroma_note = f"{indexed_count} indexed document(s)"
except VectorStoreError as error:
    chroma_ready = False
    chroma_note = str(error)

search_ready = embedding_model_available and chroma_ready
search_note = (
    f"{chroma_note} · {EMBEDDING_MODEL}"
    if search_ready
    else (
        f"Install with: ollama pull {EMBEDDING_MODEL}"
        if not embedding_model_available
        else chroma_note
    )
)
status_columns = st.columns(3, gap="medium")
with status_columns[0]:
    render_status_card(
        "Ollama service",
        "Connected" if ollama_connected else "Unavailable",
        (
            "Configured Ollama endpoint"
            if ollama_connected and ollama_host
            else "Local Ollama runtime"
            if ollama_connected
            else ollama_message
        ),
        ollama_connected,
    )
with status_columns[1]:
    render_status_card(
        "Language model",
        "Ready" if chat_model_available else "Model missing",
        CHAT_MODEL if chat_model_available else f"Install with: ollama pull {CHAT_MODEL}",
        chat_model_available,
    )
with status_columns[2]:
    render_status_card(
        "Document search",
        "Ready" if search_ready else "Setup needed",
        search_note,
        search_ready,
    )

with st.sidebar:
    st.header("📚 Your documents")
    st.caption("Build your private college knowledge base.")
    uploads = st.file_uploader(
        "Choose text-based PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Scanned/image-only PDFs are not supported because OCR is not enabled.",
    )
    st.caption("PDF only · up to 20 MB each · scanned PDFs need OCR")
    retrieval_limit = st.slider(
        "Retrieved passages per question",
        min_value=1,
        max_value=10,
        value=DEFAULT_RESULT_COUNT,
    )
    if st.button(
        "Index uploaded PDFs",
        disabled=not uploads or not ollama_connected or not embedding_model_available,
    ):
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
                f"Indexed {uploaded_file.name}: {len(pages)} page(s) with text, "
                f"{len(chunks)} chunk(s)."
            )

    st.divider()
    st.header("Indexed library")
    try:
        indexed_documents = store.list_documents()
    except VectorStoreError as error:
        st.error(str(error))
        indexed_documents = None
    if indexed_documents == []:
        st.info("Your library is empty. Upload a PDF above to get started.")
    elif indexed_documents is not None:
        for document in indexed_documents:
            st.markdown(
                f'<div class="document-row">📄 {html.escape(document["source"])}</div>',
                unsafe_allow_html=True,
            )
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

st.markdown(
    '<div class="section-heading">Ask your documents</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="section-hint">Ask about a policy, deadline, requirement, or '
    'service mentioned in your PDFs.</div>',
    unsafe_allow_html=True,
)
if st.sidebar.button("Clear current answer"):
    st.session_state.pop("answer", None)
    st.session_state.pop("sources", None)
    st.rerun()

with st.form("question_form"):
    question = st.text_input(
        "Question",
        placeholder="e.g. What attendance requirement is listed?",
    )
    submitted = st.form_submit_button("✨ Ask a question", type="primary")

if submitted:
    st.session_state.pop("answer", None)
    st.session_state.pop("sources", None)
    if not question.strip():
        st.warning("Enter a question before asking.")
    else:
        try:
            with st.spinner("Searching your documents and preparing an answer..."):
                answer, sources = rag_service.ask(question, limit=retrieval_limit)
        except (
            EmbeddingServiceError,
            LLMServiceError,
            VectorStoreError,
            ValueError,
        ) as error:
            st.error(str(error))
        else:
            st.session_state["answer"] = answer
            st.session_state["sources"] = sources

if "answer" in st.session_state:
    answer = st.session_state["answer"]
    sources = st.session_state["sources"]
    with st.container(border=True):
        st.markdown(
            '<div class="answer-label">Your answer</div>',
            unsafe_allow_html=True,
        )
        if answer == UNKNOWN_ANSWER:
            st.warning(
                f"{answer} Check the indexed documents or try rephrasing your question."
            )
        else:
            st.markdown(answer)
    if sources:
        st.markdown(
            '<div class="section-heading">Evidence from your documents</div>',
            unsafe_allow_html=True,
        )
        for source in sources:
            st.markdown(
                f'<span class="source-chip">📄 {html.escape(source.source)}'
                f' · Page {source.page_number}</span>',
                unsafe_allow_html=True,
            )
            with st.expander("View supporting passage"):
                st.write(source.text)

st.markdown(
    '<div style="margin-top:2.5rem;color:#8a93a8;font-size:.78rem;text-align:center">'
    "Uploaded PDFs are processed in memory; extracted text and vectors are stored "
    "in this app's ChromaDB · Answers are based on retrieved passages"
    "</div>",
    unsafe_allow_html=True,
)
