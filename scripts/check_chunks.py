"""Quote-integrity check: does chunking (and cleaning) keep every evidence quote intact?

For every answerable question (unique doc/page/quote; Persian questions share the
quote of their English pair) we check that the squashed evidence_quote is fully
inside ONE chunk of the right document and page.

Outcome per quote:
  ok               quote is inside a single chunk
  split            quote is on the page but straddles a chunk boundary
  lost_by_cleaning quote was in the ingested page text but the cleaner removed part of it
  wrong_page       quote is in the document, but not on evidence_page
  not_found        quote is nowhere in the document text
  doc_missing      question refers to a file that was not ingested

Examples:
  python scripts/check_chunks.py
  python scripts/check_chunks.py --sweep 128:20:20,256:40:40,384:60:60
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sia.evaluation.dataset import load_questions, squash, unique_quotes  # noqa: E402
from sia.ingestion.pipeline import ingest_directory                # noqa: E402
from sia.processing.chunker import ChunkingConfig, chunk_document  # noqa: E402
from sia.processing.cleaner import clean_pages                     # noqa: E402
from sia.processing.tokens import make_hf_counter, whitespace_counter  # noqa: E402


def evaluate(docs, quotes, cfg: ChunkingConfig, count, clean: bool = True):
    by_name = {d.meta.filename: d for d in docs}
    built = {}
    for name, d in by_name.items():
        pages = d.pages if not clean else clean_pages(d.pages)[0]
        built[name] = (pages, chunk_document(d.meta, pages, cfg, count))

    outcome, failures = Counter(), []
    for q in quotes:
        d = by_name.get(q["doc"])
        if d is None:
            res = "doc_missing"
        else:
            pages, chunks = built[q["doc"]]
            raw = {p.page: squash(p.text) for p in d.pages}
            cleaned = {p.page: squash(p.text) for p in pages}
            cands = [c for c in chunks if c.page == q["page"]]
            if any(q["sq"] in squash(c.text) for c in cands):
                res = "ok"
            elif q["sq"] in cleaned.get(q["page"], ""):
                res = "split"
            elif q["sq"] in raw.get(q["page"], ""):
                res = "lost_by_cleaning"
            elif any(q["sq"] in t for t in raw.values()):
                res = "wrong_page"
            else:
                res = "not_found"
        outcome[res] += 1
        if res != "ok":
            failures.append((res, q))
    n_chunks = sum(len(c) for _, c in built.values())
    return outcome, failures, n_chunks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docs-dir", type=Path, default=ROOT / "data" / "docs")
    ap.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifest.yaml")
    ap.add_argument("--questions-dir", type=Path, default=ROOT / "eval" / "questions")
    ap.add_argument("--chunk-tokens", type=int, default=256)
    ap.add_argument("--overlap", type=int, default=40)
    ap.add_argument("--min-tokens", type=int, default=40)
    ap.add_argument("--sweep", help="comma list of chunk:overlap:min, e.g. 128:20:20,256:40:40")
    ap.add_argument("--no-clean", action="store_true")
    ap.add_argument("--counter", choices=["bge", "whitespace"], default="bge")
    ap.add_argument("--max-show", type=int, default=25, help="failures to print per config")
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.ERROR)

    count = make_hf_counter() if args.counter == "bge" else whitespace_counter
    docs = [d for d in ingest_directory(args.docs_dir, args.manifest).documents
            if d.status == "ok"]
    quotes = unique_quotes(load_questions(args.questions_dir))
    print(f"Documents ingested: {len(docs)} | unique evidence quotes: {len(quotes)} | "
          f"counter={args.counter} clean={'off' if args.no_clean else 'on'}")

    configs = ([tuple(int(x) for x in s.split(":")) for s in args.sweep.split(",")]
               if args.sweep else [(args.chunk_tokens, args.overlap, args.min_tokens)])
    hard_fail = False
    print(f"\n{'chunk/ovl/min':>14} {'chunks':>7} {'ok':>5} {'split':>6} {'cleaned':>8} "
          f"{'wrongpg':>8} {'missing':>8}  ok-rate")
    all_failures = {}
    for ct, ov, mn in configs:
        out, failures, n_chunks = evaluate(docs, quotes, ChunkingConfig(ct, ov, mn),
                                           count, not args.no_clean)
        total = sum(out.values())
        print(f"{f'{ct}/{ov}/{mn}':>14} {n_chunks:>7} {out['ok']:>5} {out['split']:>6} "
              f"{out['lost_by_cleaning']:>8} {out['wrong_page']:>8} "
              f"{out['not_found'] + out['doc_missing']:>8}  {out['ok'] / total:.1%}")
        all_failures[(ct, ov, mn)] = failures
        if out["lost_by_cleaning"] or out["not_found"] or out["doc_missing"] or out["wrong_page"]:
            hard_fail = True

    for cfg_key, failures in all_failures.items():
        if not failures:
            continue
        print(f"\n--- non-ok quotes for {cfg_key[0]}/{cfg_key[1]}/{cfg_key[2]} "
              f"(first {args.max_show}) ---")
        grouped = defaultdict(list)
        for res, q in failures:
            grouped[res].append(q)
        shown = 0
        for res, qs in grouped.items():
            for q in qs:
                if shown >= args.max_show:
                    break
                print(f"[{res}] {q['id']} | {q['doc']} p.{q['page']} | {q['quote'][:90]!r}")
                shown += 1
    if hard_fail:
        print("\nHARD FAILURE: some quotes were lost by cleaning, missing or on the wrong page.")
    return 1 if hard_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
