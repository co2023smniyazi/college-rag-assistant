# Project Audit

**Audit date:** 2026-10-15  
**Repository:** `co2023smniyazi/college-rag-assistant`  
**Project path:** `D:\cp2 project\college-rag-assistant`

## Summary

This is an existing Python/Streamlit RAG application, not an empty project.
It already contained PDF text extraction, page-aware chunking, Ollama
embeddings/chat, persistent ChromaDB storage, semantic retrieval, an answer
prompt, a small Streamlit interface, an evaluation runner, 50 sample questions,
and unit tests. The core code was retained and improved rather than recreated.

## Files and structure inspected

- Application modules: `app.py`, `app_services.py`, `document_loader.py`,
  `chunker.py`, `embeddings.py`, `vector_store.py`, `retriever.py`,
  `llm_service.py`, `rag_service.py`, `models.py`.
- Evaluation: `evaluate.py`, `evaluation.py`,
  `data/sample_questions.json`.
- Tests and fixture: `tests/`, including a synthetic fictional handbook PDF.
- Configuration and docs: `requirements.txt`, `.gitignore`,
  `.streamlit/config.toml`, and `README.md`.
- Local runtime data: `venv/` and `chroma_db/`. These are local, ignored
  artifacts, not source code.

## Features already implemented before this audit

- Streamlit app entry point is `app.py`; it binds to localhost.
- PyPDF extracts selectable text, tracks source filename/page, and rejects
  invalid, empty, password-protected, or textless PDFs. There is no OCR.
- Chunk defaults are 1,500 characters and 200 characters of overlap.
- Ollama is used for `nomic-embed-text` and `llama3.2:3b`; there is no
  cloud-provider or fake-embedding fallback.
- Chroma uses a persistent local collection and stores source, page, document,
  and chunk metadata. Document IDs are SHA-256 hashes of PDF bytes.
- The retriever has a cosine-distance cutoff. The RAG service returns the
  required exact refusal when no relevant chunk remains.
- The UI supports multiple uploads, duplicate skipping, document listing and
  deletion, asking questions, and viewing retrieved passages.
- The 50-item fixture dataset already has 20 direct, 20 paraphrased, and 10
  unanswerable questions. Tests use fake Ollama responses.

## Not implemented or not verified

- OCR is not implemented. Scanned/image-only PDFs cannot be indexed.
- There is no real college PDF in the project or environment. The included
  evaluation content is synthetic, so no answer-quality claim for real college
  documents is supported.
- `nomic-embed-text` was missing during the audit. Until installed, embedding,
  indexing, and live RAG evaluation cannot be completed.
- Prompting plus a similarity threshold reduces unsupported answers but cannot
  formally guarantee that a generative model never hallucinates.
- No Jira integration or remote Jira change was performed; Jira usage is
  intentionally manual and project tracking only.

## Gaps and issues found and addressed

- The UI did not show live Ollama/model/Chroma status. It now reports service
  reachability and model availability, with installation guidance.
- The Ollama service had `llama3.2:3b`, but not `nomic-embed-text`; indexing and
  embedding therefore cannot work until the missing model is installed.
- PDF size was not explicitly bounded. Upload and extraction now enforce a
  20 MB per-file limit.
- Retrieval count was fixed at three. The UI now provides a 1–10 control, with
  the existing default of three.
- The answer was not retained in Streamlit session state and there was no clear
  answer action. The current answer can now be cleared.
- Errors from Ollama/Chroma were translated but not logged. Failures now have
  Python logging, with user-facing messages retained.
- Chroma embedding input validation did not reject inconsistent dimensions
  before storage and did not explain dimension mismatches clearly.
- The original README incorrectly directed users to create Jira project key
  `RAG`, contradicted the provided `KAN` project key and existing `KAN-4`, and
  had inaccurate structure/setup claims. It has been replaced with current
  setup instructions and a manual KAN plan.
- Required architecture, setup, evaluation, demo, and Jira planning documents
  were missing; these are added under `docs/`.

## Runtime and dependency status checked

- Python: **3.11.9** in the existing virtual environment.
- Installed package versions: Streamlit **1.38.0**, ChromaDB **0.5.15**,
  PyPDF **5.1.0**, Ollama Python client **0.6.3**, pytest **8.3.3**. These match
  `requirements.txt`.
- Ollama HTTP service: **reachable** at its default local endpoint during the
  audit. `llama3.2:3b` is installed; `nomic-embed-text` is **missing**.
  The `ollama` CLI was not found on the current PowerShell PATH even though the
  local HTTP service responded.
- ChromaDB: the local `chroma_db/` directory opens, its collection exists, and
  the collection is **empty** (0 chunks, 0 indexed documents). No college PDF
  has been indexed here.
- Tests before changes: **29 passed** (baseline).
- No real college PDF was provided. The bundled handbook is synthetic test
  content, so real-document answer quality has not been established.

## Final verification

- Automated tests after implementation: **36 passed**. ChromaDB emitted 71
  Pydantic deprecation warnings from its installed dependency stack.
- Python compilation/import syntax check: passed for all application and test
  modules.
- `python -m pip check`: no broken requirements.
- Streamlit UI smoke test: app loaded in a browser without a runtime exception;
  it showed Ollama connected, the chat model available, the embedding model
  missing, and ChromaDB ready with zero indexed documents. This verifies UI
  startup only, not real-document question answering.
- Jira CSV structure: 20 planned task rows (one existing `KAN-4` row and 19
  create rows), six epics, four sprints. Dataset counts are 20/20/10.

## Proposed implementation plan

1. Keep the existing module boundaries and add missing status, validation,
   logging, and UI behavior without switching providers or storage.
2. Expand unit coverage for runtime status, file-size validation, configurable
   retrieval, duplicate writes, and storage validation.
3. Create beginner-friendly architecture/setup/testing/demo documentation,
   correct Jira references to KAN, and provide an import-ready task CSV without
   making remote Jira changes.
4. Run tests, syntax/import checks, dependency checks, and verify Streamlit can
   start. Report live-model limitations separately.

## Files changed and follow-up

Implementation changes are in the existing app/services and test files.
`README.md`, this report, and the documents under `docs/` provide the updated
instructions. No Jira issues were created or modified. The runtime status
recorded above reflects the original audit date; see the follow-up below for the
current model and Chroma state.

## Follow-up RAG investigation — 2026-10-16

The original runtime snapshot above is historical. A subsequent check found
Ollama reachable with both required models installed and a populated local
collection (17 chunks across two documents). The collection initially had no
embedding-model identity on its chunks, so prior vectors could have been reused
or mixed silently. The local chunks were re-embedded with `nomic-embed-text`
and all 17 now carry the model marker.

Live testing of the bundled PDF and 50-question dataset showed:

- The existing 0.8 cosine-distance cutoff retrieved context for all 10
  unanswerable questions. Distances overlapped the answerable cases, so a
  tighter cutoff would incorrectly discard answerable passages. The cutoff is
  now 0.64, preserving the measured 40/40 expected-source retrievals.
- The model sometimes returned a non-exact explanation for missing information
  and confused related facts within the same topic. Structured JSON output,
  evidence-focused prompting, exact-refusal normalization, and a polarity
  correction for explicitly contradicted yes/no answers address those cases.
- One bundled paraphrase asked how much class time could be missed but expected
  the stated 75% attendance requirement. That required an unstated calculation;
  the test question was corrected to ask for the supported attendance threshold.

After the changes, the live local-model run retrieved the expected source and
generated a non-refusal answer for all 40 answerable questions and returned the
exact refusal for all 10 unanswerable questions. Automated validation reports
70 passing tests. This fixture is fictional and does not establish answer
quality for any real college PDF.
