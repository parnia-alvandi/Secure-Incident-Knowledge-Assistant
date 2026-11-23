"""Ask the local knowledge base (baseline RAG: retrieve -> evidence gate -> local LLM -> sources).

Examples (PowerShell, project root):
  python scripts/ask.py --question "What are the four phases of incident response?"
  python scripts/ask.py --question "..." --json          # machine-readable result
  python scripts/ask.py --question "..." --no-llm        # retrieval only (no Ollama needed)
  python scripts/ask.py                                  # interactive: empty line or 'exit' quits
Each answer on CPU takes roughly 30-45 seconds (prefill + generation with the 3B model).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sia.answering.pipeline import (STATUS_ANSWERED, STATUS_ERROR, STATUS_INSUFFICIENT,  # noqa: E402
                                    STATUS_UNVERIFIED, AnswerResult, RAGPipeline)
from sia.config import load_settings                                                     # noqa: E402
from sia.llm.ollama_client import OllamaClient                                           # noqa: E402
from sia.retrieval.vector_retriever import (IndexNotReadyError, RetrievedChunk,          # noqa: E402
                                            VectorRetriever)
from sia.runtime import apply_offline_mode, quiet_noisy_loggers                          # noqa: E402

STATUS_LABEL = {STATUS_ANSWERED: "ANSWERED", STATUS_INSUFFICIENT: "INSUFFICIENT EVIDENCE",
                STATUS_UNVERIFIED: "UNVERIFIED (no valid citation)", STATUS_ERROR: "ERROR",
                "no_answer": "NO ANSWER (the model returned no text)"}


def format_results(results: list[RetrievedChunk], excerpt_chars: int) -> str:
    lines = []
    for r in results:
        c = r.chunk
        text = " ".join(c.text.split())
        excerpt = text if len(text) <= excerpt_chars else text[:excerpt_chars].rstrip() + " ..."
        lines.append(f"[{r.rank}] score={r.score:.3f}  {c.title}\n"
                     f"    {c.filename} | {c.location()}  (chunk {c.chunk_id})\n"
                     f"    \"{excerpt}\"")
    return "\n".join(lines)


def _refs(title: str, refs) -> list[str]:
    if not refs:
        return []
    out = [f"{title}:"]
    for ref in refs:
        tag = f"[{ref.n}] " if ref.n else "- "
        out.append(f"  {tag}{ref.title}\n      {ref.filename} | {ref.location}  (score {ref.score:.3f})\n"
                   f"      \"{ref.excerpt}\"")
    return out


def format_answer(res: AnswerResult) -> str:
    t = res.timings
    lines = [f"\nQ: {res.question}", f"Status: {STATUS_LABEL.get(res.status, res.status)}"]
    if res.answer:
        lines.append(f"Answer: {res.answer}")
    if res.reason:
        lines.append(f"Reason: {res.reason}")
    for w in res.warnings:
        lines.append(f"Warning: {w}")
    lines += _refs("Sources (cited)", res.sources)
    lines += _refs("Closest passages (not cited, shown for transparency)", res.closest)
    if res.others:
        lines.append("Also retrieved, not cited by the answer:")
        for ref in res.others:
            lines.append(f"  [{ref.n}] {ref.title[:70]} | {ref.filename} | {ref.location}  (score {ref.score:.3f})")
    parts = [f"retrieval {t.get('retrieval_s', 0):.2f}s"]
    if "llm_total_s" in t:
        parts.append(f"LLM {t['llm_total_s']:.1f}s (load {t['llm_load_s']:.1f}s; prompt "
                     f"{t['prompt_tokens']} tok @ {t.get('prompt_tokens_per_s')} tok/s; generated "
                     f"{t['completion_tokens']} tok @ {t.get('gen_tokens_per_s')} tok/s)")
    if t.get("retries"):
        parts.append(f"repair retries {t['retries']}")
    parts.append(f"total {t.get('total_s', 0):.1f}s | best score {t.get('best_score')}")
    lines.append("Timing: " + " | ".join(parts))
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--index-dir", type=Path, default=None)
    ap.add_argument("--question", "-q", default=None)
    ap.add_argument("--k", type=int, default=None, help="passages to retrieve (default: retrieval.top_k)")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--no-llm", action="store_true", help="retrieval only; Ollama is not used")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    s = load_settings(args.config)
    apply_offline_mode(s.embedding.offline)
    logging.basicConfig(level=logging.WARNING)
    quiet_noisy_loggers()
    try:
        retriever = VectorRetriever.from_index(args.index_dir or s.paths.vector_db_dir, s)
    except IndexNotReadyError as exc:
        print(f"ERROR: {exc}")
        return 2
    if args.k:
        retriever.top_k = args.k
    load_s = retriever.warmup()
    if not args.json:
        print(f"(embedding model ready in {load_s:.1f}s)")

    pipeline = None
    if not args.no_llm:
        llm = OllamaClient.from_settings(s)
        problem = llm.check()
        if problem:
            print(f"ERROR: {problem}\n(use --no-llm for retrieval only)")
            return 3
        pipeline = RAGPipeline.from_settings(retriever, llm, s)

    def handle(question: str) -> None:
        if pipeline is None:
            t0 = time.perf_counter()
            results = retriever.retrieve(question)
            dt = time.perf_counter() - t0
            if args.json:
                print(json.dumps({"question": question, "retrieval_seconds": round(dt, 3),
                                  "results": [{"rank": r.rank, "score": round(r.score, 4),
                                               "chunk_id": r.chunk.chunk_id, "title": r.chunk.title,
                                               "filename": r.chunk.filename,
                                               "location": r.chunk.location(), "text": r.chunk.text}
                                              for r in results]}, ensure_ascii=False, indent=2))
            else:
                print(f"\nQ: {question}   (retrieval {dt * 1000:.0f} ms)")
                print(format_results(results, s.answering.excerpt_chars))
            return
        if not args.json:
            print("\n(generating on CPU; this takes about 30-45 seconds...)")
        res = pipeline.ask(question)
        print(json.dumps(res.to_dict(), ensure_ascii=False, indent=2) if args.json
              else format_answer(res))

    if args.question:
        handle(args.question)
        return 0
    print("Ask a question (empty line or 'exit' to quit).")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() in {"exit", "quit"}:
            break
        handle(q)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
