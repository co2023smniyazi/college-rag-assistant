from chunker import chunk_pages
from document_loader import extract_pdf_pages
from embeddings import EMBEDDING_MODEL, OllamaEmbedder
from llm_service import UNKNOWN_ANSWER
from rag_service import RAGService
from retriever import Retriever
from vector_store import ChromaVectorStore, document_id_for


class KeywordEmbeddingClient:
    def __init__(self):
        self.models = []

    def embed(self, model, input):
        self.models.append(model)
        return {
            "embeddings": [
                [
                    float("attendance" in text.casefold()),
                    float("exam" in text.casefold() or "test" in text.casefold()),
                    float("library" in text.casefold()),
                    float("scholarship" in text.casefold()),
                ]
                for text in input
            ]
        }


class ContextEchoAnswerGenerator:
    def __init__(self):
        self.questions = []
        self.contexts = []

    def generate(self, question, chunks):
        self.questions.append(question)
        self.contexts.append(chunks)
        return chunks[0].text


def test_pdf_to_answer_pipeline_direct_paraphrased_irrelevant_and_absent(
    tmp_path, make_text_pdf
) -> None:
    pdf_bytes = make_text_pdf(
        ["Students need 75% attendance to sit semester examinations."]
    )
    pages = extract_pdf_pages(pdf_bytes, "college_rules.pdf")
    chunks = chunk_pages(pages)
    embedding_client = KeywordEmbeddingClient()
    embedder = OllamaEmbedder(client=embedding_client)
    store = ChromaVectorStore(path=tmp_path / "chroma")

    store.add_chunks(
        document_id_for(pdf_bytes),
        chunks,
        embedder.embed_texts([chunk.text for chunk in chunks]),
    )
    answer_generator = ContextEchoAnswerGenerator()
    service = RAGService(Retriever(embedder, store), answer_generator)

    direct_answer, direct_sources = service.ask(
        "What attendance is required for semester exams?"
    )
    paraphrased_answer, paraphrased_sources = service.ask(
        "How much attendance do students need before tests?"
    )
    irrelevant_answer, irrelevant_sources = service.ask(
        "How many books can students borrow from the library?"
    )
    absent_answer, absent_sources = service.ask(
        "What scholarship score is required?"
    )

    assert direct_answer == chunks[0].text
    assert paraphrased_answer == chunks[0].text
    assert [source.source for source in direct_sources] == ["college_rules.pdf"]
    assert [source.source for source in paraphrased_sources] == ["college_rules.pdf"]
    assert irrelevant_answer == UNKNOWN_ANSWER
    assert absent_answer == UNKNOWN_ANSWER
    assert irrelevant_sources == []
    assert absent_sources == []
    assert answer_generator.questions == [
        "What attendance is required for semester exams?",
        "How much attendance do students need before tests?",
    ]
    assert all(model == EMBEDDING_MODEL for model in embedding_client.models)
