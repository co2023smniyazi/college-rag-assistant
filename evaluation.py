from collections.abc import Iterable, Mapping

from llm_service import UNKNOWN_ANSWER


def calculate_retrieval_accuracy(results: Iterable[Mapping[str, object]]) -> float:
    all_questions = list(results)
    if not all_questions:
        return 0.0
    correct = sum(row["retrieval_correct"] is True for row in all_questions)
    return correct / len(all_questions) * 100


def calculate_safe_refusal_rate(results: Iterable[Mapping[str, object]]) -> float:
    unanswerable = [row for row in results if row["answerable"] is False]
    if not unanswerable:
        return 0.0
    refused = 0
    for row in unanswerable:
        answer = row.get("answer")
        if (
            isinstance(answer, str)
            and answer.strip().casefold() == UNKNOWN_ANSWER.casefold()
        ):
            refused += 1
    return refused / len(unanswerable) * 100
