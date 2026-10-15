# Local Setup Guide (Windows PowerShell)

## Requirements

- Windows 10/11, Python 3.11, and Git.
- Ollama installed and running locally.
- Disk space and memory for the `llama3.2:3b` and `nomic-embed-text` models.

## Install dependencies

The repository already has a `venv/` environment. From the repository root:

```powershell
.\venv\Scripts\Activate.ps1
python --version
python -m pip install -r requirements.txt
```

To create the environment on another computer:

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Install the local models

Run these in PowerShell after installing Ollama:

```powershell
ollama pull llama3.2:3b
ollama pull nomic-embed-text
ollama list
```

Both model names should appear in `ollama list`. Ollama's desktop application
usually starts its local service. If needed, open a second terminal and run:

```powershell
ollama serve
```

The app uses the local Ollama API (default `http://localhost:11434`) and needs
no API key. You may set `OLLAMA_HOST` before launching if Ollama listens at a
different local address.

## Start and use the application

From the project root with the virtual environment active:

```powershell
streamlit run app.py
```

Open the URL printed by Streamlit, normally `http://localhost:8501`. The app
binds to `127.0.0.1`. Check the visible service/model statuses, upload text-based
PDFs (maximum 20 MB each), select **Index uploaded PDFs**, then ask a question.
The interface reports page/chunk counts and shows source pages with the answer.

## Local files and privacy

- Chroma vectors and extracted text are stored in ignored `chroma_db/`.
- Uploaded PDFs are read from Streamlit memory and are not saved to an upload
  directory by this app.
- `venv/`, `.env`, local data, and test caches are ignored by Git.
- Do not index confidential student records. Avoid committing college PDFs.
- Scanned PDFs require OCR; this project does not implement it.

## Common problems

- **Ollama not reachable:** start the desktop app or `ollama serve`.
- **Embedding model missing:** run `ollama pull nomic-embed-text`.
- **Chat model missing:** run `ollama pull llama3.2:3b`.
- **Invalid/encrypted/oversized PDF:** use a valid unlocked PDF under 20 MB.
- **No selectable text:** use a text-based PDF or OCR it separately before
  upload. This app does not claim OCR support.
- **Chroma dimension mismatch:** stop and review the error; do not delete the
  database unless you intend to discard and rebuild the index.

See the root [README](../README.md) for tests and evaluation instructions.
