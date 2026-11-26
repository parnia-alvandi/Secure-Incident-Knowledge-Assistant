Secure Incident Knowledge Assistant

A local question-answering assistant for computer-security incident documents (PDF, DOCX, TXT). It retrieves the relevant passages, asks a local LLM to answer only from those passages, and shows the source document, page and excerpt for every answer. If the documents do not contain enough evidence, it says so instead of guessing.

question -> retrieval (embeddings + Chroma) -> evidence gate -> local LLM (Ollama) -> answer + sources

Everything runs on your machine (CPU only). No document text is sent to an external service.

Requirements
Windows 11 (developed and tested there), 16 GB RAM recommended, no GPU needed
Python 3.10 or newer
Ollama installed and running
Setup (PowerShell)
powershell
git clone <repo-url>
cd <repo-folder>

python -m venv venv
.\venv\Scripts\Activate.ps1

pip install torch --index-url https://download.pytorch.org/whl/cpu    # CPU-only PyTorch
pip install -r requirements.txt

ollama pull qwen2.5:3b-instruct                                       # the local LLM (about 2 GB)
Documents

The document collection is not part of this repository. Copy the course documents into data/docs/ (keep the original file names; data/manifest.yaml describes them).

Build the index (once)

The embedding model (BAAI/bge-small-en-v1.5) must be downloaded on the first run. config.yaml runs offline by default, so allow the download once:

In config.yaml set embedding.offline: false.
Run:
powershell
python scripts/build_index.py
python scripts/verify_index.py        # optional: checks that the index is complete and consistent
Set embedding.offline: true again. From now on no internet connection is needed.
Run
powershell
python scripts/ask.py --question "Which earlier NIST publication does SP 800-61r3 supersede?"
python scripts/ask.py                       # interactive; an empty line or "exit" quits
python scripts/ask.py --question "..." --no-llm   # retrieval only (Ollama not needed)
python scripts/ask.py --question "..." --json     # machine-readable output

Ollama must be running (it starts with Windows after installation; otherwise open the Ollama app). On a laptop CPU one answer takes roughly 30-45 seconds; the first one is slower while the model loads.

How to read the output
Field	Meaning
Status: ANSWERED	The answer is grounded in the retrieved passages; every claim is cited as [n].
Status: INSUFFICIENT EVIDENCE	The collection does not contain enough information. No answer is invented.
Status: UNVERIFIED	The model replied without a valid citation, so the answer is not trusted.
Sources (cited)	Document title, file, page and the exact excerpt the answer relies on.
Closest passages	Shown for transparency when nothing could be cited.
Test run: 5 questions + 1 unanswerable question

Run each command and keep the output (the files in docs/demo/ are the documented results):

powershell
mkdir docs\demo
python scripts/ask.py --question "<question 1>" | Out-File -Encoding utf8 docs\demo\q1.txt
python scripts/ask.py --question "<question 2>" | Out-File -Encoding utf8 docs\demo\q2.txt
python scripts/ask.py --question "<question 3>" | Out-File -Encoding utf8 docs\demo\q3.txt
python scripts/ask.py --question "<question 4>" | Out-File -Encoding utf8 docs\demo\q4.txt
python scripts/ask.py --question "<question 5>" | Out-File -Encoding utf8 docs\demo\q5.txt
python scripts/ask.py --question "<a question the documents cannot answer>" | Out-File -Encoding utf8 docs\demo\q6_unanswerable.txt

Expected behaviour: questions 1-5 return ANSWERED with sources whose excerpts support the answer; question 6 returns INSUFFICIENT EVIDENCE. Check the excerpts yourself: the assistant can be wrong.

Configuration

All settings are in config.yaml (no model names or paths are hard-coded elsewhere):

Key	Default	Purpose
models.embedding	BAAI/bge-small-en-v1.5	embedding model (CPU friendly)
llm.model	qwen2.5:3b-instruct	local model served by Ollama
chunking.chunk_tokens / overlap_tokens	256 / 40	chunk size and overlap
retrieval.top_k	5	passages sent to the LLM
embedding.offline	true	use only locally cached models
Tests
powershell
pytest -q
Project layout
src/sia/        ingestion, cleaning, chunking, embeddings, vector store, retrieval, answering, LLM client
scripts/        build_index.py, ask.py, verify_index.py and the evaluation tools
tests/          unit tests
data/           manifest.yaml (documents go in data/docs/, not committed)
config.yaml     central configuration
Known limitations
Documents and questions are English; the embedding model is English-only.
Answers are only as good as the retrieved passages. Retrieval finds the right passage in the top 5 for most, not all, questions, and the 3B model is sometimes too cautious and answers INSUFFICIENT EVIDENCE although the evidence is present.
Scanned PDFs without a text layer are flagged during ingestion and cannot be answered from.
This is a read-only knowledge assistant: it does not scan networks, run commands or modify anything.
Work in progress: hybrid search (BM25 + RRF) and cross-encoder re-ranking are being evaluated (scripts/eval_retrieval.py) and are not used by ask.py yet.