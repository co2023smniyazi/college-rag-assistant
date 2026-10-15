# Demonstration Script

Use a non-sensitive, text-based PDF approved for demonstration. Do not present
the synthetic fixture as an official college handbook.

## Before the demo

1. Start Ollama and check that `ollama list` shows `llama3.2:3b` and
   `nomic-embed-text`.
2. Activate `venv/` and run `streamlit run app.py`.
3. Open the localhost URL. Confirm the service, model, and Chroma status.
4. Choose an approved PDF under 20 MB. Do not use confidential student data.

## Walkthrough

1. Explain that PDFs and vectors stay local and that the app does not use paid
   APIs.
2. Upload the PDF; index it and show the reported readable pages/chunks.
3. Ask one question whose answer is directly supported by the PDF.
4. Point out the answer, source filename, page number, retrieval distance, and
   passage preview. Verify the text against the PDF.
5. Ask an unrelated question. Demonstrate the exact safe-refusal message.
6. Show the retrieval-count control, indexed document list, and safe removal.
7. Clear the displayed answer.

## Close

Explain that scanned PDFs need OCR, which is not built into this project; model
quality and speed depend on the local computer; and test fixture results are
not a guarantee of real-world answer quality. Do not claim live-model or
real-document evaluation unless it was actually run.
