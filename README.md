# Secure Incident Knowledge Assistant

Local RAG + bounded ReAct assistant for computer-security incident knowledge.
Everything runs on this machine (CPU): documents, embeddings, vector index and the LLM.
(Full documentation will be completed in Phase 9.)

**Status:** baseline RAG is working (ingestion, chunking, local index, retrieval, grounded answers with sources,
"insufficient evidence" handling). Hybrid search, re-ranking, the ReAct agent and Persian support are not done yet.

## Setup (Windows, Python 3.11)

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
ollama pull qwen2.5:3b-instruct
```

The Ollama app (or `ollama serve`) must be running when you ask questions. Documents live in `data/docs`
and are described in `data/manifest.yaml`.

## Run

```powershell
python scripts/build_index.py        # chunk + embed + store; about 2-3 minutes, rebuilds from scratch
python scripts/verify_index.py       # structural checks and a retrieval smoke test
python scripts/ask.py --question "What are the four phases of the incident response life cycle?"
python scripts/ask.py                # interactive mode (empty line quits)
python scripts/ask.py --question "..." --json       # machine-readable result
python scripts/ask.py --question "..." --no-llm     # retrieval only, no Ollama needed
```

One answer takes about 30 seconds on a laptop CPU (mostly prompt processing). The first question also loads the
embedding model (about 5 seconds).

### What an answer contains

`Status` is one of:

| status | meaning |
|---|---|
| `ANSWERED` | the answer cites at least one passage that was really given to the model |
| `INSUFFICIENT EVIDENCE` | the collection does not answer the question (the model refused, or the evidence gate blocked it) |
| `UNVERIFIED` | an answer without a valid citation; do not trust it |
| `NO ANSWER` | the model returned no usable text |
| `ERROR` | the LLM could not be reached; the message says what to do |

`ANSWERED` means the answer is **cited**, not that it is guaranteed correct. Sources (document title, page or
section, excerpt) are built from the index metadata, never by the model, so a source cannot be invented.

## Evaluate

```powershell
python scripts/calibrate_gate.py                                   # evidence-gate statistics (no LLM)
python scripts/eval_baseline.py --name baseline_eval_v1r           # 80 questions, about 40 minutes, resumable
python scripts/make_demo.py --name baseline_eval_v1r               # writes docs/baseline_demo.md
```

Results of the baseline iterations: `docs/baseline_iterations.md`. Demo with real outputs: `docs/baseline_demo.md`.

## Configuration

All settings are in `config.yaml`: models, chunking (`256/40/40` tokens), `retrieval.top_k`, the optional evidence
gate `retrieval.min_score` (off by default), `answering.prompt_version`, `llm.*`, `embedding.offline`
(use only the locally cached model).

## Tests

```powershell
pytest
```

## Known limitations

- Retrieval is vector-only: the right passage reaches the model for about 80% of the test questions.
- The model is conservative: it refused about 40% of the answerable test questions (and 100% of the unanswerable ones).
- English only; the interface is a command line.
- Measured on 80 questions that were written from the same documents: a baseline, not a final evaluation.
