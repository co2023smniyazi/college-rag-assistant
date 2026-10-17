# College RAG Assistant

A question-answering app for a shared library of public college PDFs. It
retrieves relevant passages and asks an Ollama model to answer from those
passages. Local development uses local Ollama and Chroma by default; deployment
can configure a remote Ollama endpoint and persistent remote Chroma server.
When there is not enough relevant evidence, the answer is
exactly: **Information not found in the available documents.**

> The bundled PDF and its sample facts are fictional test data, not real college
> policy. No public or college-specific documents are bundled.

## Features

- Upload multiple PDFs (up to 20 MB each), extract selectable text, and keep
  source filenames and 1-based page numbers.
- Split text into 1,500-character chunks with 200-character overlap.
- Use Ollama `nomic-embed-text` embeddings with local ChromaDB or a configured
  remote Chroma server;
  each chunk records its embedding model, and legacy unmarked chunks are
  re-embedded before retrieval.
- Search for up to 10 relevant passages (3 by default), apply a cosine-distance
  cutoff calibrated against the bundled question set, and show distances and
  passage text for inspection.
- Use Ollama `llama3.2:3b` for answers. Its structured JSON response is
  validated and unsupported answers become the exact required refusal.
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
         -> llama3.2:3b -> answer + retrieved source names/pages
```

For local runs, ChromaDB data is kept in the ignored `chroma_db/` directory.
For a cloud deployment, configure a reachable Ollama service and a remote
Chroma server using durable storage; Streamlit Cloud's local filesystem is not
persistent. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

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
as a local subprocess, so no network port is opened. Install MCP dependencies
with `python -m pip install -r requirements-mcp.txt`, then add an MCP server
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
python -m pip install -r requirements-dev.txt
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
- Streamlit, Ollama, ChromaDB, and PyPDF are the web app's runtime dependencies
  in `requirements.txt`.
- The MCP SDK is in `requirements-mcp.txt`; pytest is in `requirements-dev.txt`.
- Ollama models: `llama3.2:3b` and `nomic-embed-text`

Pinned dependency versions are in their corresponding requirements files. The
existing `venv/` already contains the full development dependencies.

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
python -m pip install -r requirements-dev.txt
```

The Ollama desktop application normally starts its local service. If it is not
running, start Ollama or run `ollama serve` in another terminal. The default
local API address is `http://localhost:11434`. The app defaults to local Ollama.
For remote Ollama, set `OLLAMA_MODE=remote` and `OLLAMA_REMOTE_HOST` to an
endpoint reachable from the app. The endpoint must have both required models
installed. `OLLAMA_HOST` remains supported as a backwards-compatible remote
host setting. Remote endpoints should use HTTPS; use `OLLAMA_API_KEY` for a
reverse-proxy bearer token or the Cloudflare Access service-token credentials
described below.

## Streamlit Community Cloud deployment

### Required architecture

Streamlit Community Cloud cannot reach Ollama on your development computer via
`localhost`, and its local filesystem is not durable storage. Deploy two
separately reachable services:

1. A secured Ollama server with **both** `llama3.2:3b` and
   `nomic-embed-text`. Put it behind HTTPS and authentication (for example, an
   authenticated reverse proxy or a Cloudflare Tunnel with an Access service
   token). Do not expose Ollama's unauthenticated port to the public internet.
2. A remote Chroma server with a persistent volume/disk attached. Protect it
   with TLS and authentication. This app uses Chroma's HTTP client; data
   persists across app restarts only if the Chroma server's backing storage is
   durable. Run a server version compatible with the pinned Python Chroma
   client.

The configured Chroma collection is deliberately shared: all visitors can
search and remove documents. Only upload public, non-sensitive college PDFs.
This app does not provide user accounts or per-user access control.

### Deploy on Streamlit Community Cloud

1. Push the application to the selected GitHub branch. In Streamlit Community
   Cloud, create/select the app using this repository, branch, and `app.py`.
2. Select Python 3.11 in Advanced settings. Use the repository's root
   `requirements.txt`; no additional LLM provider package or paid API is used.
3. In **Manage app → Settings → Secrets**, enter the service endpoints and
   credentials. Replace all placeholders:

   ```toml
   OLLAMA_MODE = "remote"
   OLLAMA_REMOTE_HOST = "https://ollama.example.com"
   OLLAMA_API_KEY = "replace-with-reverse-proxy-bearer-token"

   CHROMA_HOST = "chroma.example.com"
   CHROMA_PORT = 443
   CHROMA_SSL = true
   CHROMA_AUTH_TOKEN = "replace-with-chroma-bearer-token"
   ```

   `OLLAMA_API_KEY` is optional if the endpoint is secured with Cloudflare
   Access instead; for Access use `CF_ACCESS_CLIENT_ID` and
   `CF_ACCESS_CLIENT_SECRET` instead of the bearer key. Credentials are sent
   only in request headers to their configured HTTPS service. Never commit
   `.streamlit/secrets.toml`, paste real credentials into source, or put
   secrets in GitHub issues.
4. Configure the Ollama server to have both models installed:

   ```sh
   ollama pull llama3.2:3b
   ollama pull nomic-embed-text
   ollama list
   ```

   A remote host has to allow network traffic from Streamlit Community Cloud
   and serve Ollama's API through the configured secure endpoint. A tunnel
   service must stay running. `localhost` is not a valid remote setting.
5. Configure the Chroma server with a durable mounted data directory and
   bearer-token authentication matching the app secret. In this app config,
   `CHROMA_HOST` is a hostname only; put the port and TLS setting in
   `CHROMA_PORT` and `CHROMA_SSL`. If the Chroma service is unavailable, the
   app shows an explicit database connection error.
6. Save Secrets and reboot the app. Confirm the Ollama status is **Connected**,
   the language model is **Ready**, and document search is ready after indexing.
   Upload a public text-based PDF and select **Index uploaded PDFs**. Local
   Chroma files are not uploaded automatically; documents must be indexed into
   the configured remote collection.
7. Test a directly worded question, a paraphrase, and a question unsupported
   by the PDF. Confirm the evidence panels contain the retrieved pages and the
   unsupported case returns exactly
   `Information not found in the available documents.`.

The Ollama endpoint and the embedding model are independently checked by
listing models on that same Ollama service; both chat and document/question
embeddings use the configured Ollama endpoint. Secrets may also be supplied as
environment variables. Local development defaults to the local Ollama client
and `chroma_db/` when remote configuration is omitted.

This design does **not** switch to a paid hosted LLM API. Self-hosting the
models and Chroma requires a reachable server, storage, bandwidth, and
maintenance; any hosting/tunnel costs depend on the provider and chosen plan.
Check provider pricing before deploying. The repository cannot provision
those services or confirm they are reachable from your Cloud app.

Cloud deployment has not been tested from this development workspace. Verify
the live app after configuring these services; do not treat a successful Git
push or local test as evidence of a successful Cloud deployment.

### MCP limitation in Cloud

Community Cloud deploys the Streamlit app, not the local MCP stdio subprocess.
The MCP server remains a local client integration: install
`requirements-mcp.txt` and configure a client to start `mcp_server.py` as
documented above. Do not expose the stdio server as a public network service.
The Cloud secrets are not automatically passed to a separately launched local
MCP process. For that process, configure `OLLAMA_HOST` in its environment and,
for an Access-protected HTTPS endpoint, also set both
`CF_ACCESS_CLIENT_ID` and `CF_ACCESS_CLIENT_SECRET` as environment variables.
A remote MCP deployment would require a separately designed authenticated
network transport and durable/shared storage.

## Index documents and ask questions

1. Open the Streamlit URL (normally `http://localhost:8501` locally, or the
   assigned app URL in Community Cloud).
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
PDFs to a temporary upload folder. Indexed data belongs to the configured
ChromaDB filesystem and is not synchronized between local and Cloud instances.
Important answers should still be checked against the cited source passages.

## Tests and evaluation

Run unit tests (mocks and temporary Chroma databases; no live Ollama required):

```powershell
python -m pip install -r requirements-dev.txt
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

The evaluator measures source retrieval accuracy for the 40 answerable
questions and exact-refusal rate for the 10 unanswerable questions. It saves
individual outputs to ignored `evaluation_results.csv`. Correctness, relevance,
groundedness, clarity, and completeness require human review and are not
fabricated by the script. See
[docs/TESTING_AND_EVALUATION.md](docs/TESTING_AND_EVALUATION.md).

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
  Untagged legacy chunks are re-embedded from their stored text with
  `nomic-embed-text`; a dimension mismatch is surfaced and blocks retrieval
  rather than mixing vectors. In that case, keep a backup and re-upload the
  original PDFs to rebuild the collection.
- **Slow answers:** local generation speed depends on available computer
  resources. This project does not claim a successful real-document answer
  until tested with a real PDF and both local models.

Project audit, detailed setup, demonstration steps, and the manual Jira plan are
in the [docs/](docs/) folder. Jira is project tracking only; no Jira integration
is part of the runtime. The current Jira project key is **KAN**, and existing
issue **KAN-4** must be preserved.
