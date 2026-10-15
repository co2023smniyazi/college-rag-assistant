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
deletion, duplicate IDs, retrieval filtering, exact refusal without relevant
evidence, and evaluation metrics.

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

- **Retrieval accuracy:** number of questions with the expected retrieval
  outcome divided by all 50 questions. Answerable items are correct when the
  expected source appears among retrieved passages. Unanswerable items are
  correct when no sufficiently relevant passage is returned.
- **Safe refusal rate:** unanswerable questions answered with the exact required
  string, divided by the 10 unanswerable questions.
- **Answer quality:** correctness, relevance, groundedness, clarity, and
  completeness, each human-rated 1–5. These must not be reported as measured
  until the reviewer completes the ratings.

Actual values are produced only by running `evaluate.py` against the indexed
fixture with both local models. No evaluation values are prefilled or claimed
here. Fixture results do not establish performance on public or real college
documents. The limited one-document dataset is not a substitute for broader
retrieval/grounding evaluation.
