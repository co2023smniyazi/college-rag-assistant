import csv
import json
from pathlib import Path

from app_services import build_rag_service
from evaluation import calculate_retrieval_accuracy, calculate_safe_refusal_rate
from llm_service import UNKNOWN_ANSWER


PROJECT_DIR = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_DIR / "data" / "sample_questions.json"
RESULTS_PATH = PROJECT_DIR / "evaluation_results.csv"


def main() -> None:
    questions = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    store, _, service = build_rag_service(PROJECT_DIR / "chroma_db")
    if not store.list_documents():
        raise SystemExit(
            "No documents are indexed. Start the app, index PDFs, then run evaluation."
        )

    results: list[dict[str, object]] = []
    for item in questions:
        answer, sources = service.ask(item["question"])
        retrieved = [source.source for source in sources]
        answerable = item["answerable"]
        results.append(
            {
                "id": item["id"],
                "category": item["category"],
                "question": item["question"],
                "answerable": answerable,
                "expected_source": item["expected_source"],
                "retrieved_sources": "; ".join(retrieved),
                "retrieval_correct": (
                    item["expected_source"] in retrieved
                    if answerable
                    else not retrieved
                ),
                "answer": answer,
                "safe_refusal": (
                    answer.strip().casefold()
                    == UNKNOWN_ANSWER.casefold()
                    if not answerable
                    else ""
                ),
                "expected_answer": item["expected_answer"],
                "human_correctness_1_5": "",
                "human_relevance_1_5": "",
                "human_groundedness_1_5": "",
                "human_clarity_1_5": "",
                "human_completeness_1_5": "",
                "human_notes": "",
            }
        )
        print(f"{item['id']}: {answer}")

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    with RESULTS_PATH.open("w", newline="", encoding="utf-8-sig") as results_file:
        writer = csv.DictWriter(results_file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    retrieval_accuracy = calculate_retrieval_accuracy(results)
    refusal_rate = calculate_safe_refusal_rate(results)
    print(f"\nRetrieval accuracy (answerable questions): {retrieval_accuracy:.1f}%")
    print(f"Safe refusal rate (unanswerable questions): {refusal_rate:.1f}%")
    print(f"Per-question results: {RESULTS_PATH}")
    print("Answer quality ratings still require human review.")


if __name__ == "__main__":
    main()
