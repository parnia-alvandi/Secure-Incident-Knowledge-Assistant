"""Baseline RAG evaluation with the real LLM (phase 4). Resumable: results are appended to a JSONL
file after every question, so Ctrl+C is safe and re-running continues where it stopped.

  python scripts/eval_baseline.py                  # 40 unanswerable + 40 answerable, seed 42
  python scripts/eval_baseline.py --limit 3        # quick smoke run
  python scripts/eval_baseline.py --report-only    # rebuild summary and review sheet from the JSONL
Takes about 40 minutes for the full set (roughly 30 s per question on this CPU).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sia.answering.pipeline import RAGPipeline                        # noqa: E402
from sia.config import load_settings                                  # noqa: E402
from sia.evaluation.baseline import (doc_kind_of, format_summary, gold_chunk_ids, load_done,  # noqa: E402
                                     review_markdown, select_questions, summarize,
                                     unsupported_numbers)
from sia.evaluation.dataset import load_questions                     # noqa: E402
from sia.llm.ollama_client import OllamaClient                        # noqa: E402
from sia.retrieval.vector_retriever import IndexNotReadyError, VectorRetriever  # noqa: E402
from sia.runtime import apply_offline_mode, quiet_noisy_loggers       # noqa: E402


def write_reports(rows: list[dict], out_dir: Path, name: str) -> None:
    summary = summarize(rows)
    text = format_summary(summary)
    (out_dir / f"{name}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / f"{name}_review.md").write_text(review_markdown(rows, text), encoding="utf-8")
    print("\n" + text)
    print(f"\nFiles: {out_dir / (name + '_summary.json')}, {out_dir / (name + '_review.md')}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--index-dir", type=Path, default=None)
    ap.add_argument("--questions-dir", type=Path, default=ROOT / "eval" / "questions")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "eval" / "results")
    ap.add_argument("--n-answerable", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None, help="only the first N selected questions")
    ap.add_argument("--min-score", type=float, default=None, help="override retrieval.min_score")
    ap.add_argument("--name", default="baseline_eval",
                    help="result file prefix (use another name to keep a previous run, e.g. baseline_eval_v2)")
    ap.add_argument("--prompt-version", type=int, choices=(1, 2), default=None,
                    help="override answering.prompt_version")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = args.out_dir / f"{args.name}.jsonl"
    selected = select_questions(load_questions(args.questions_dir), args.n_answerable, args.seed)
    if args.limit:
        selected = selected[: args.limit]
    done = load_done(jsonl)
    if args.report_only:
        rows = [done[q["id"]] for q in selected if q["id"] in done]
        if not rows:
            print("No results yet.")
            return 1
        write_reports(rows, args.out_dir, args.name)
        return 0

    s = load_settings(args.config)
    apply_offline_mode(s.embedding.offline)
    logging.basicConfig(level=logging.WARNING)
    quiet_noisy_loggers()
    try:
        retriever = VectorRetriever.from_index(args.index_dir or s.paths.vector_db_dir, s)
    except IndexNotReadyError as exc:
        print(f"ERROR: {exc}")
        return 2
    retriever.warmup()
    llm = OllamaClient.from_settings(s)
    problem = llm.check()
    if problem:
        print(f"ERROR: {problem}")
        return 3
    pipeline = RAGPipeline.from_settings(retriever, llm, s)
    if args.min_score is not None:
        pipeline.min_score = args.min_score
    if args.prompt_version is not None:
        pipeline.prompt_version = args.prompt_version
    print(f"Selected {len(selected)} questions (seed {args.seed}); {len(done)} already done; "
          f"gate min_score={pipeline.min_score}; "
          f"prompt v{pipeline.prompt_version}; max_retries={pipeline.max_retries}; results -> {jsonl.name}")

    todo = [q for q in selected if q["id"] not in done]
    errors_in_a_row = 0
    try:
        for i, q in enumerate(todo, start=1):
            res = pipeline.ask(q["question"])
            retrieved = retriever.retrieve(q["question"])             # same call the pipeline made
            gold = gold_chunk_ids(q, retriever.chunks_by_id)
            cited_ids = {src.chunk_id for src in res.sources}
            shown_ids = cited_ids | {o.chunk_id for o in res.others} | {c.chunk_id for c in res.closest}
            cited_text = [retriever.chunks_by_id[c].text for c in cited_ids]
            row = {
                "id": q["id"], "answerable": bool(q.get("answerable")),
                "category": q.get("category"), "kind": doc_kind_of(q.get("doc")),
                "question": q["question"], "gold_answer": q.get("answer"),
                "gold_doc": q.get("doc"), "gold_page": q.get("evidence_page"),
                "status": res.status, "answer": res.answer, "reason": res.reason,
                "warnings": res.warnings, "prompt_version": pipeline.prompt_version,
                "retries": res.timings.get("retries", 0),
                "gold_in_prompt": bool(gold & {r.chunk.chunk_id for r in retrieved}) if gold else None,
                "gold_rank": next((r.rank for r in retrieved if r.chunk.chunk_id in gold), None),
                "cited_gold": bool(gold & cited_ids) if gold else None,
                "unsupported_numbers": unsupported_numbers(res.answer, cited_text) if cited_text else [],
                "sources": [{"n": x.n, "chunk_id": x.chunk_id, "title": x.title,
                             "filename": x.filename, "location": x.location, "excerpt": x.excerpt}
                            for x in res.sources],
                "shown_chunk_ids": sorted(shown_ids), "timings": res.timings,
            }
            with jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            done[q["id"]] = row
            print(f"[{i}/{len(todo)}] {q['id']:8s} {res.status:22s} {res.timings.get('total_s', 0):5.1f}s  "
                  f"{q['question'][:60]!r}")
            errors_in_a_row = errors_in_a_row + 1 if res.status == "error" else 0
            if errors_in_a_row >= 3:
                print(f"\nStopping: 3 errors in a row ({res.reason}). Fix the problem and re-run to resume.")
                break
    except KeyboardInterrupt:
        print("\nInterrupted: results so far are saved; re-run to continue.")

    rows = [done[q["id"]] for q in selected if q["id"] in done]
    write_reports(rows, args.out_dir, args.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
