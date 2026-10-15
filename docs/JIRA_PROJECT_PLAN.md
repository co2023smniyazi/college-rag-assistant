# Jira Project Plan

Jira is used only for planning; the RAG application has no Jira integration.
The supplied existing project key is **KAN**, and the existing **KAN-4 —
Define Functional Requirements** task must be kept. This repository environment
has no authorized Jira API/browser integration, so no Jira issue, workflow,
epic, or sprint was created or modified.

## Six epics

| Epic name | Scope |
| --- | --- |
| Requirements and Dataset | Confirm scope, collect approved documents, prepare and review 50 questions |
| Local LLM Setup | Install Ollama and verify the required local models |
| RAG Pipeline | PDF extraction, chunks, embeddings, Chroma, retrieval, context prompt, citations, refusal |
| User Interface | Streamlit user experience and source presentation |
| Testing and Evaluation | Functional checks, human quality review, and defect fixes |
| Documentation and Demonstration | Setup/help docs and a repeatable project demo |

## Tasks, descriptions, and epic association

The companion [CSV](jira_tasks.csv) contains the task list, descriptions,
epic names, sprint assignment, and existing issue marker. Each description is
written to be usable as a Jira issue description.

| Task | Epic | Sprint |
| --- | --- | --- |
| Define Functional Requirements (existing KAN-4) | Requirements and Dataset | Sprint 1 |
| Collect College PDF Documents | Requirements and Dataset | Sprint 1 |
| Prepare 50 Sample Questions | Requirements and Dataset | Sprint 1 |
| Install and Configure Ollama | Local LLM Setup | Sprint 1 |
| Test Local LLM | Local LLM Setup | Sprint 1 |
| Implement PDF Text Extraction | RAG Pipeline | Sprint 2 |
| Implement Text Chunking | RAG Pipeline | Sprint 2 |
| Generate Document Embeddings | RAG Pipeline | Sprint 2 |
| Configure ChromaDB Vector Database | RAG Pipeline | Sprint 2 |
| Implement Semantic Search | RAG Pipeline | Sprint 2 |
| Connect Retrieved Context to LLM | RAG Pipeline | Sprint 3 |
| Create RAG Prompt | RAG Pipeline | Sprint 3 |
| Display Source Documents and Pages | RAG Pipeline | Sprint 3 |
| Handle Questions with Missing Information | RAG Pipeline | Sprint 3 |
| Develop Streamlit User Interface | User Interface | Sprint 3 |
| Display Retrieved Sources in UI | User Interface | Sprint 3 |
| Perform Functional Testing | Testing and Evaluation | Sprint 4 |
| Evaluate Answer Quality | Testing and Evaluation | Sprint 4 |
| Fix Identified Bugs | Testing and Evaluation | Sprint 4 |
| Prepare Documentation and Demonstration | Documentation and Demonstration | Sprint 4 |

## Four one-week sprints

| Sprint | Work | Goal |
| --- | --- | --- |
| Sprint 1 — Foundation | Requirements, documents, questions, Ollama setup/test | Agree on scope and prove the local models run |
| Sprint 2 — RAG Pipeline | Extraction, chunking, embeddings, Chroma, retrieval | Retrieve page-aware evidence from PDFs |
| Sprint 3 — Application | LLM context, prompt, citations, missing-info handling, UI | Deliver grounded answers with visible sources |
| Sprint 4 — Verification and Demo | Functional tests, answer evaluation, bug fixes, docs, demonstration | Verify behavior and prepare a reproducible handoff |

No sprint dates are assumed; set dates to the team's actual calendar in Jira.

## Safe manual setup

1. Open the existing **College RAG Assistant** Jira project and verify its key
   is `KAN`. Do not create another project.
2. Confirm that `KAN-4` already exists. Keep it and associate it with the
   **Requirements and Dataset** epic; do not create a second requirements issue.
3. Create the six epics above using Jira's **Create** flow with issue type
   **Epic** (Jira UI labels may vary).
4. Import the CSV through Jira's CSV import wizard if available. First exclude
   the row whose **Existing Jira Key** is `KAN-4`; that row is included only to
   document the plan and must not be imported. Import the other 19 tasks.
5. Map `Summary`, `Description`, and `Issue Type`. Map `Epic Name` to the
   epic-link field available in your Jira version; if the importer does not
   support epic linkage, assign each task through the Jira backlog/issue
   interface after import. Do not map the custom `Existing Jira Key` or `Sprint`
   fields as new issue keys.
6. Create four one-week sprints in the Scrum backlog, add tasks according to
   the sprint table, and set the actual start/end dates.
7. If permitted by the existing project configuration, update the workflow to
   **To Do → In Progress → Code Review → Testing → Done**. Jira project
   permissions or plan tier may prevent workflow editing; use the closest
   existing statuses instead of creating duplicate statuses.

Check issue titles in the backlog before importing to avoid duplicates. The CSV
is a planning aid only and does not mean Jira has been changed.
