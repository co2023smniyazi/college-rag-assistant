import json

import pytest

from evaluation import calculate_retrieval_accuracy, calculate_safe_refusal_rate
from llm_service import UNKNOWN_ANSWER


def test_dataset_has_50_balanced_question_categories(project_root) -> None:
    questions = json.loads(
        (project_root / "data" / "sample_questions.json").read_text(encoding="utf-8")
    )

    assert len(questions) == 50
    assert sum(question["category"] == "direct" for question in questions) == 20
    assert sum(question["category"] == "paraphrased" for question in questions) == 20
    assert sum(question["category"] == "unanswerable" for question in questions) == 10
    assert all(
        "fictional" in question["expected_source"]
        for question in questions
        if question["answerable"]
    )


def test_retrieval_accuracy_uses_all_questions() -> None:
    results = [
        {"answerable": True, "retrieval_correct": True},
        {"answerable": True, "retrieval_correct": False},
        {"answerable": False, "retrieval_correct": True},
    ]

    assert calculate_retrieval_accuracy(results) == pytest.approx(200 / 3)


def test_safe_refusal_rate_counts_only_exact_refusals() -> None:
    results = [
        {"answerable": False, "answer": UNKNOWN_ANSWER},
        {"answerable": False, "answer": "Maybe the fee is $100."},
        {"answerable": True, "answer": UNKNOWN_ANSWER},
    ]

    assert calculate_safe_refusal_rate(results) == 50
