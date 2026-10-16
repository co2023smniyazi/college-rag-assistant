# Testing and Evaluation

## Unit tests

Activate the repository environment, then run:

```powershell
.\venv\Scripts\Activate.ps1
python -m pytest -q
```

The tests use fake Ollama clients and temporary Chroma directories. They do not
require the live Ollama service, downloaded models, network access, or actual
college documents. Coverage includes PDF extraction and errors, chunk boundaries
and metadata, embedding failures, model/service status, Chroma persistence and
deletion, duplicate IDs, legacy embedding re-indexing, end-to-end PDF-to-answer
flow for direct/paraphrased/irrelevant/absent questions, exact refusal handling,
and evaluation metrics.

## Question dataset

`data/sample_questions.json` contains exactly 50 entries:

- 20 direct questions
- 20 paraphrased questions
- 10 unanswerable questions

The answerable entries target the synthetic one-page
`tests/fixtures/fictional_demo_handbook.pdf`. This is software test data, not a
public or official college document. No real college document was available for
this implementation.

## Run the RAG evaluation

1. Start the app and index `tests/fixtures/fictional_demo_handbook.pdf`.
2. Confirm both Ollama models are available. The evaluator needs a working
   Ollama service and both models.
3. From the repository root, run:

   ```powershell
   python evaluate.py
   ```

4. Review generated `evaluation_results.csv` and its source evidence manually.

The script records each question, expected source, retrieved sources, answer,
retrieval outcome, and exact refusal outcome. It does **not** score answer
correctness or quality automatically. The reviewer should fill in the CSV's
human-rating columns with integer values from 1 to 5 for correctness,
relevance, groundedness, clarity, and completeness, and document notes.

## Metric definitions and limitations

- **Retrieval accuracy:** answerable questions whose expected source appears
  among retrieved passages divided by the 40 answerable questions. It does not
  mix retrieval accuracy with the separate absent-answer test.
- **Safe refusal rate:** unanswerable questions answered with the exact required
  string, divided by the 10 unanswerable questions.
- **Answer quality:** correctness, relevance, groundedness, clarity, and
  completeness, each human-rated 1–5. These must not be reported as measured
  until the reviewer completes the ratings.

The 50-question fixture was run in this environment with the local
`nomic-embed-text` and `llama3.2:3b` models on 2026-10-16. All 40 answerable
questions retrieved the expected source and produced a non-refusal answer; all
10 unanswerable questions returned the exact refusal. The unanswerable examples
also retrieved candidate passages, confirming that similarity alone does not
establish answerability. These are fixture results, not a claim of performance
on public or real college documents. Answer quality still needs human review,
and the limited one-document dataset is not a substitute for broader
retrieval/grounding evaluation.

The live distance check against the bundled questions found the largest
answerable cosine distance at 0.6331, so the configured 0.64 cutoff retains all
answerable examples with the tested model. The unanswerable distances overlap
that range; no cosine cutoff alone can reliably separate them without dropping
answerable questions. The locally hosted chat model therefore receives the
retrieved evidence and must return an explicit supported/unsupported result;
the app enforces the exact refusal for unsupported output.
