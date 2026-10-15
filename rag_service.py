from llm_service import UNKNOWN_ANSWER, OllamaAnswerGenerator
from models import RetrievedChunk
from retriever import DEFAULT_RESULT_COUNT, Retriever


class RAGService:
    def __init__(self, retriever: Retriever, answer_generator: OllamaAnswerGenerator) -> None:
        self.retriever = retriever
        self.answer_generator = answer_generator

    def ask(
        self, question: str, limit: int = DEFAULT_RESULT_COUNT
    ) -> tuple[str, list[RetrievedChunk]]:
        question = question.strip()
        if not question:
            raise ValueError("Enter a question before asking.")

        chunks = self.retriever.retrieve(question, limit=limit)
        if not chunks:
            return UNKNOWN_ANSWER, []
        answer = self.answer_generator.generate(question, chunks)
        return answer, chunks
