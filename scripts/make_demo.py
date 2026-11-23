"""Build docs/baseline_demo.md from the recorded evaluation results (no LLM call, nothing re-run).

  python scripts/make_demo.py

Human verdicts are read from eval/results/verdicts.json if it exists, e.g.
  {"DDO-08": {"verdict": "correct", "note": "matches the reference"},
   "SYN11-09": {"verdict": "partly", "note": "right time, wrong role"}}
Allowed verdicts: correct | partly | wrong. Without a verdict the document says "pending".
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sia.config import load_settings                                   # noqa: E402
from sia.evaluation.baseline import (demo_markdown, demo_selection, format_summary,  # noqa: E402
                                     load_done, select_questions, summarize)
from sia.evaluation.dataset import load_questions                      # noqa: E402
from sia.storage.chunk_store import MANIFEST_FILE, load_manifest       # noqa: E402

ALLOWED = {"correct", "partly", "wrong"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--questions-dir", type=Path, default=ROOT / "eval" / "questions")
    ap.add_argument("--results-dir", type=Path, default=ROOT / "eval" / "results")
    ap.add_argument("--name", default="baseline_eval", help="result file prefix used by eval_baseline.py")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "baseline_demo.md")
    ap.add_argument("--n-answerable", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    s = load_settings(args.config)
    selected = select_questions(load_questions(args.questions_dir), args.n_answerable, args.seed)
    done = load_done(args.results_dir / f"{args.name}.jsonl")
    rows = [done[q["id"]] for q in selected if q["id"] in done]
    if len(rows) < len(selected):
        print(f"WARNING: only {len(rows)} of {len(selected)} selected questions have results; "
              f"run scripts/eval_baseline.py to finish.")
    if not rows:
        print("No results to report.")
        return 1

    verdict_path = args.results_dir / (f"verdicts_{args.name}.json" if args.name != "baseline_eval" else "verdicts.json")
    verdicts = json.loads(verdict_path.read_text(encoding="utf-8")) if verdict_path.is_file() else {}
    bad = {k: v for k, v in verdicts.items() if v.get("verdict") not in ALLOWED}
    if bad:
        print(f"ERROR: invalid verdict value(s) in {verdict_path}: {sorted(bad)}; allowed: {sorted(ALLOWED)}")
        return 2

    index_dir = s.paths.vector_db_dir
    m = load_manifest(index_dir / MANIFEST_FILE)
    setup = {"LLM": f"{s.llm.model} via Ollama (temperature {s.llm.temperature}, num_ctx {s.llm.num_ctx}, "
                    f"num_predict {s.llm.num_predict}), local CPU",
             "Embedding model": s.models.embedding,
             "Prompt": f"version {s.answering.prompt_version} (max {s.answering.max_retries} repair retry)",
             "Retrieval": f"vector search (Chroma, cosine), top-{s.retrieval.top_k}; evidence gate "
                          f"{'off' if s.retrieval.min_score is None else s.retrieval.min_score}"}
    if m:
        setup["Index"] = (f"{m.n_documents} documents, {m.n_chunks} chunks, chunking "
                          f"{m.chunking['chunk_tokens']}/{m.chunking['overlap_tokens']}/{m.chunking['min_tokens']} "
                          f"(size/overlap/min tokens), built {m.created_at}")
        setup["Libraries"] = ", ".join(f"{k} {v}" for k, v in m.libraries.items())

    demo_rows = demo_selection(selected, done)
    md = demo_markdown(demo_rows, format_summary(summarize(rows)), setup, verdicts,
                       args.n_answerable, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md, encoding="utf-8")
    pending = [r["id"] for r in demo_rows if r["id"] not in verdicts]
    print(f"Written {args.out} ({len(demo_rows)} demo questions: {[r['id'] for r in demo_rows]})")
    if pending:
        print(f"Verdict pending for: {pending}  (create {verdict_path}; see the script docstring)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
