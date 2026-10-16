# College RAG Assistant

A question-answering app for college PDFs. It retrieves relevant passages and
asks an Ollama model to answer from those passages. Ollama runs locally by
default; if `OLLAMA_HOST` is configured, document text and questions are sent to
that endpoint. When there is not enough relevant evidence, the answer is
exactly: **Information not found in the available documents.**

> The bundled PDF and its sample facts are fictional test data, not real college
> policy. No public or college-specific documents are bundled.

## Features

- Upload multiple PDFs (up to 20 MB each), extract selectable text, and keep
  source filenames and 1-based page numbers.
- Split text into 1,500-character chunks with 200-character overlap.
- Use Ollama `nomic-embed-text` embeddings and persistent local ChromaDB;
  each chunk records its embedding model, and legacy unmarked chunks are
  re-embedded before retrieval.
- Search for up to 10 relevant passages (3 by default), apply a cosine-distance
  cutoff calibrated against the bundled question set, and show distances and
  passage text for inspection.
- Use Ollama `llama3.2:3b` for answers. Its structured answerability result is
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

For local runs, Streamlit binds to `127.0.0.1` and ChromaDB data is kept in the
ignored `chroma_db/` directory. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

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
local API address is `http://localhost:11434`. Set `OLLAMA_HOST` if using a
different local or remote Ollama endpoint. The endpoint must be reachable from
the process running this app and have both required models installed. For a
remote Ollama server protected by Cloudflare Access, configure the optional
service-token credentials described below.

## Streamlit Community Cloud deployment

### Python and dependencies

The project was developed and tested with **Python 3.11.9**. Streamlit
Community Cloud currently defaults new deployments to Python 3.12; select
**Python 3.11** in Advanced settings to match the tested runtime. Root
`requirements.txt` contains the packages imported by the Streamlit app. The MCP
SDK and pytest are not needed by the web entrypoint and are in
`requirements-mcp.txt` and `requirements-dev.txt`, respectively. Community
Cloud should use the root `requirements.txt`.

Check the production requirements locally:

```powershell
python -m pip install --dry-run --ignore-installed -r requirements.txt
```

The reported message **"Error installing requirements"** alone does not
identify the failing package or resolver conflict. No Streamlit Cloud build log
was available in this workspace, so the exact cause is not verified. Open the
app in Community Cloud, select **Manage app**, and copy/download the complete
build log, including the first `ERROR:` line and the selected Python version,
to diagnose a specific Cloud failure.

### Deployment steps and Ollama configuration

1. In Streamlit Community Cloud, choose **Create app** and select this GitHub
   repository, the intended branch, and `app.py` as the entrypoint.
2. Open **Advanced settings** and select Python 3.11.
3. Streamlit Cloud cannot connect to Ollama running on your PC using
   `localhost`; that address points back to the Cloud app container. To use
   the Ollama models on a Windows PC, create a Cloudflare Tunnel and protect
   its public hostname with Cloudflare Access before adding the app secrets:

   1. Confirm Ollama runs on the PC and both models are installed:

      ```powershell
      ollama list
      ```

      The output must include `llama3.2:3b` and `nomic-embed-text`.
   2. Install `cloudflared` on that PC (for example, with
      `winget install --id Cloudflare.cloudflared --exact`).
   3. In the Cloudflare Zero Trust dashboard, create a **named Cloudflare
      Tunnel** for the PC. Follow the dashboard's Windows instructions to run
      the tunnel as a service. Keep the tunnel token private and never commit
      or paste it into this repository.
   4. In the tunnel's **Published application routes** (sometimes labelled
      **Public Hostnames**), add a hostname such as
      `ollama.yourdomain.com` and route it to `http://localhost:11434`.
      Cloudflare DNS for the domain must be managed by Cloudflare.
   5. In **Cloudflare Access → Applications**, create a self-hosted
      application for that same hostname. Create a **Service Auth** policy
      that allows only the service token you create under **Access → Service
      credentials**. Do not leave the hostname publicly accessible without
      this Access policy.
   6. In Streamlit Community Cloud, open **Manage app → Settings → Secrets**
      and add the HTTPS hostname and both service-token values:

      ```toml
      OLLAMA_HOST = "https://ollama.yourdomain.com"
      CF_ACCESS_CLIENT_ID = "your-service-token-client-id"
      CF_ACCESS_CLIENT_SECRET = "your-service-token-client-secret"
      ```

      Replace every example value. Keep the service token only in Streamlit
      Secrets; do not put it in source code, a committed file, or a chat.
   7. Restart the Streamlit app. Confirm Ollama shows **Connected** and both
      models show **Ready** before uploading and indexing the PDF in the
      deployed app. Your local Chroma index is not copied to Streamlit Cloud;
      upload the PDF there and select **Index uploaded PDFs**.

   If Ollama is on a different remote server already, use its secured HTTPS
   endpoint and configure Access credentials if required. Do not use the
   Streamlit app URL or the local model directory as `OLLAMA_HOST`.

The simplest app secrets shape for an already-secured remote Ollama server is:

```toml
OLLAMA_HOST = "https://ollama.example.com"
```

The Streamlit app reads the `OLLAMA_HOST` environment variable first, then
`st.secrets["OLLAMA_HOST"]`. The configured endpoint must be network-reachable
from Cloud, have both `nomic-embed-text` and `llama3.2:3b` installed, and be
appropriately secured. For a Cloudflare Access-protected HTTPS endpoint,
configure both service-token secrets in Streamlit Community Cloud under
**Manage app → Settings → Secrets**:

```toml
OLLAMA_HOST = "https://<your-access-protected-ollama-host>"
CF_ACCESS_CLIENT_ID = "<Cloudflare service-token client ID>"
CF_ACCESS_CLIENT_SECRET = "<Cloudflare service-token client secret>"
```

Both Access credentials must be configured together. They are sent as
`CF-Access-Client-Id` and `CF-Access-Client-Secret` HTTP headers only to the
configured non-local HTTPS Ollama endpoint. Local loopback Ollama connections
do not receive these headers. Environment variables with the same names are also supported by the Streamlit
app; when either credential environment variable is set, both must be set
there. Credentials are not included in the app's connection errors or logs.
This change only authenticates requests: it does not create or host an
Ollama server, Cloudflare Tunnel, DNS name, or Access policy.

### Indexed documents and persistence

Cloud does not automatically receive your local Ollama process, downloaded
models, ignored `chroma_db/` directory, or local indexed documents. A new Cloud
instance starts with a new/empty Chroma collection. Uploaded original PDF bytes
are processed in memory and not saved by the app; extracted chunks and vectors
are written to the project-relative `chroma_db/` directory. Chroma's local
`PersistentClient` is not a durable cloud storage service: indexed data may be
lost when the app restarts, rebuilds, or moves to another instance. Do not rely
on this deployment for persistent or sensitive document storage. Durable
production use requires a persistent external vector store, backup/restore,
and appropriate user isolation/access controls.

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
