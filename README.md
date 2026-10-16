# College RAG Assistant

A locally hosted question-answering app for college PDFs. It retrieves relevant
passages and asks a local Ollama model to answer from those passages. It does not
send documents to a paid cloud API. When there is not enough relevant evidence,
the answer is exactly: **Information not found in the available documents.**

> The bundled PDF and its sample facts are fictional test data, not real college
> policy. No public or college-specific documents are bundled.

## Features

- Upload multiple PDFs (up to 20 MB each), extract selectable text, and keep
  source filenames and 1-based page numbers.
- Split text into 1,500-character chunks with 200-character overlap.
- Use Ollama `nomic-embed-text` embeddings and persistent local ChromaDB.
- Search for up to 10 relevant passages (3 by default); filter low-relevance
  results before asking the LLM.
- Use Ollama `llama3.2:3b` for answers and show the retrieved source pages and
  supporting passage text.
- List and safely remove indexed PDFs; identical PDF bytes are not indexed twice.
- Show Ollama, model, and ChromaDB status in the interface.
- Run the 50-question evaluation dataset and record measured retrieval and
  refusal results. Human answer-quality ratings are left for a reviewer.

Scanned/image-only PDFs are not OCR-processed. If no selectable text is found,
the app explains that OCR is required.

## Architecture

```text
PDF upload -> PyPDF extraction -> page-aware chunking -> Ollama embeddings
           -> persistent ChromaDB

Question -> Ollama question embedding -> Chroma similarity search -> relevance
         -> local llama3.2:3b -> answer + retrieved source names/pages
```

The application is local. Streamlit binds to `127.0.0.1`; ChromaDB data is kept
in the ignored `chroma_db/` directory. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## MCP server

The project includes a separate, read-only Model Context Protocol (MCP) server
in `mcp_server.py`, built with the official Python MCP SDK's FastMCP API. It
reuses the existing `build_rag_service`, Chroma collection, retriever, and RAG
answer generator; it does not create a second index or alter Streamlit. Its
tools are `list_indexed_documents`, `search_college_documents`,
`ask_college_assistant`, and `get_document_context`. Search and context results
include source names, page numbers, text, and retrieval distances. When no
relevant context is found, the tools say so and the answering tool returns the
existing exact refusal rather than generating an unsupported answer.

The server uses **stdio transport**. An MCP-compatible desktop client starts it
as a local subprocess, so no network port is opened. Install Python dependencies
in the project's Python 3.11 environment as shown above, then add an MCP server
entry to the client's configuration. For Claude Desktop on Windows, the
configuration is typically `%APPDATA%\Claude\claude_desktop_config.json`; merge
this entry into its existing `mcpServers` object and update the project path if
needed:

```json
{
  "mcpServers": {
    "college-rag-assistant": {
      "command": "D:\\cp2 project\\college-rag-assistant\\venv\\Scripts\\python.exe",
      "args": ["D:\\cp2 project\\college-rag-assistant\\mcp_server.py"],
      "cwd": "D:\\cp2 project\\college-rag-assistant"
    }
  }
}
```

Restart the MCP client after saving its configuration. The server reads the
same `chroma_db/` directory as the app and creates its RAG services when a tool
is first called. Ollama must be running and have `nomic-embed-text` and
`llama3.2:3b` installed. Set `OLLAMA_HOST` in the MCP client's process
environment only if Ollama is not at its default local address; do not put
credentials or secrets in this source file. MCP tools are read-only and have no
arbitrary filesystem or shell access.

To start the MCP server manually from PowerShell:

```powershell
.\venv\Scripts\python.exe .\mcp_server.py
```

This starts the stdio protocol server; it is intended to be launched by an MCP
client, not opened in a browser. Test it with:

```powershell
python -m pytest -q tests/test_mcp_server.py
```

The Streamlit UI and MCP server are separate processes. Both may read the same
local ChromaDB data, but avoid indexing or deleting documents concurrently from
both processes. This configuration is local-only. A remote deployment needs
durable storage for ChromaDB and a reachable Ollama service (or a separately
designed remote inference setup); stdio is not a network transport for remote
clients.

### Deployment readiness

No cloud host or deployment configuration is selected in this repository. The
current Windows/Python setup supports local stdio clients that can launch the
configured Python subprocess. The Streamlit UI and MCP process can run side by
side on a Linux VM if both use the same project files and durable `chroma_db/`
storage. A remote MCP client cannot connect directly to stdio over a network:
it must be able to launch the process remotely (for example, through a
carefully configured SSH stdio command), or the server must be adapted to
authenticated Streamable HTTP. This project does not configure either remote
option, authentication, a durable cloud vector store, or remote Ollama
inference. Hosting platforms that do not support long-running Python
subprocesses or persistent local storage are not suitable without architectural
changes. Cloudflare Workers do not host this Python Streamlit/Ollama/Chroma
application as-is.

## Technologies

- Python 3.11 (the existing environment is Python 3.11.9)
- Streamlit, Ollama, ChromaDB, PyPDF, the MCP Python SDK, pytest
- Ollama models: `llama3.2:3b` and `nomic-embed-text`

Pinned dependency versions are in `requirements.txt`. The existing `venv/`
already contains these dependencies.

## Setup and run

Prerequisites: Python 3.11, Git, Ollama, and enough local storage/memory for the
models. From PowerShell in this project folder:

```powershell
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
ollama pull llama3.2:3b
ollama pull nomic-embed-text
streamlit run app.py
```

If you need to create the existing virtual environment on a new machine:

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The Ollama desktop application normally starts its local service. If it is not
running, start Ollama or run `ollama serve` in another terminal. The default
local API address is `http://localhost:11434`; set `OLLAMA_HOST` if using another
local Ollama endpoint. No API key is needed.

## Index documents and ask questions

1. Open the local Streamlit URL (normally `http://localhost:8501`).
2. Check that Ollama and both required models show as available. If the
   embedding model is missing, install it with
   `ollama pull nomic-embed-text`.
3. Upload one or more text-based PDFs in the left sidebar and choose
   **Index uploaded PDFs**. The app reports readable pages and chunk counts.
4. Find indexed document names below the upload area. Choose a document and
   select **Remove selected document** to delete its vectors from ChromaDB.
5. Enter a question, choose **Ask**, and review the answer, retrieved documents,
   page numbers, and supporting passages. Adjust the retrieval count in the
   sidebar if needed. Use **Clear current answer** to clear the displayed result.

Identical PDF bytes are skipped by a content hash. The app does not save uploaded
PDFs to a temporary upload folder. Important answers should still be checked
against the cited source passages.

## Tests and evaluation

Run unit tests (mocks and temporary Chroma databases; no live Ollama required):

```powershell
python -m pytest -q
```

The `data/sample_questions.json` dataset has 20 direct questions, 20
paraphrased questions, and 10 unanswerable questions, all tied to the
fictional handbook. To measure it, first index
`tests/fixtures/fictional_demo_handbook.pdf` in the app, keep Ollama running, then
run:

```powershell
python evaluate.py
```

The evaluator measures source retrieval accuracy across all 50 questions and
exact-refusal rate on the 10 unanswerable questions. It saves individual
outputs to ignored `evaluation_results.csv`. Correctness, relevance,
groundedness, clarity, and completeness require human review and are not
fabricated by the script. See [docs/TESTING_AND_EVALUATION.md](docs/TESTING_AND_EVALUATION.md).

## Troubleshooting and limitations

- **Ollama unavailable:** start the Ollama app or run `ollama serve`.
- **Missing model:** run `ollama pull llama3.2:3b` and
  `ollama pull nomic-embed-text`. The application never substitutes fake or
  cloud embeddings.
- **PDF rejected or no text:** check it is a valid, unlocked PDF under 20 MB.
  Scanned documents need OCR, which is not implemented.
- **No answer:** confirm the right PDF is indexed and actually contains evidence
  for the question. Low-relevance results are refused.
- **Chroma errors or dimension mismatch:** do not delete `chroma_db/` unless you
  intentionally want to lose the local index and rebuild it consistently.
- **Slow answers:** local generation speed depends on available computer
  resources. This project does not claim a successful real-document answer
  until tested with a real PDF and both local models.

Project audit, detailed setup, demonstration steps, and the manual Jira plan are
in the [docs/](docs/) folder. Jira is project tracking only; no Jira integration
is part of the runtime. The current Jira project key is **KAN**, and existing
issue **KAN-4** must be preserved.
