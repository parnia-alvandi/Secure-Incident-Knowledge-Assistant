"""Phase 5 retrieval evaluation: BM25 vs vector search on the English answerable questions.

Reports, per system: chunk-level Hit@1/3/5/10, MRR@10, document-level Hit@5 and how many distinct
documents the top 5 come from. Then the RESCUE analysis: gold chunks that BM25 finds and the vector
search misses (and vice versa), and the union (upper bound for a hybrid). Everything is also shown for
the identifier questions (e.g. "NIST SP 800-61r3") and for two seeded halves (dev / test).

  python scripts/eval_retrieval.py                       # vector + BM25 (approved field set)
  python scripts/eval_retrieval.py --skip-vector         # BM25 only: no embedding model needed
  python scripts/eval_retrieval.py --ablate-fields       # effect of filename / title / section text
  python scripts/eval_retrieval.py --sensitivity         # hybrid: k_rrf x candidate depth (informational)

The success criteria are fixed BEFORE looking at results (printed at the end, informational):
  C1  union Hit@10 (vector + BM25) is at least --min-union-gain (default 3 points) above vector alone;
  C2  on identifier questions BM25 chunk Hit@5 is higher than the vector Hit@5 (needs >= 10 such questions).
Field choices are judged on the dev half and confirmed on the test half; nothing is tuned here.

Phase 5b adds the hybrid system (RRF of vector + BM25) with its own pre-registered criterion:
  H1  hybrid MRR@10 is higher than the best single retriever on BOTH the dev and the test half;
  H2  hybrid chunk Hit@5 is not lower than BM25's on BOTH halves.
The sensitivity grid is informational and is never used to change the defaults (k_rrf=60, depth=50).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sia.config import load_settings                                          # noqa: E402
from sia.evaluation.dataset import load_questions                             # noqa: E402
from sia.evaluation.retrieval_eval import (build_items, evaluate_system, rescue_stats,   # noqa: E402
                                           split_indices, summarize)
from sia.retrieval.bm25 import ALL_FIELDS, DEFAULT_FIELDS, Bm25Retriever      # noqa: E402
from sia.retrieval.hybrid import DEFAULT_DEPTH, DEFAULT_K_RRF, HybridRetriever  # noqa: E402
from sia.retrieval.vector_retriever import IndexNotReadyError                 # noqa: E402
from sia.runtime import apply_offline_mode, quiet_noisy_loggers               # noqa: E402
from sia.storage.chunk_store import CHUNKS_FILE, MANIFEST_FILE, load_manifest, read_chunks  # noqa: E402

KS = (1, 3, 5, 10)
ABLATION = [                                    # (label, fields)
    ("text only", ("text",)),
    ("section+text", ("section", "text")),
    ("title+text", ("title", "text")),
    ("filename+text", ("filename", "text")),
    ("title+section+text", ("title", "section", "text")),
    ("filename+title+section+text", ALL_FIELDS),
]


def pct(x: float) -> str:
    return "  n/a" if x != x else f"{x:5.1%}"


def row(label: str, s: dict) -> str:
    h = s["chunk_hit"]
    mrr_key = next(k for k in s if k.startswith("mrr@"))
    return (f"  {label:30s} n={s['n']:4d} | chunk Hit@1/3/5/10: "
            + " ".join(pct(h[k]) for k in KS)
            + f" | MRR {s[mrr_key]:.3f} | doc Hit@5 {pct(s['doc_hit@5'])}"
            f" | docs in top5 {s['distinct_docs_top5']:.2f}")


class CachedRetriever:
    """Replays stored top-N lists (question -> results), so the fusion grid needs no re-embedding."""
    def __init__(self, table: dict):
        self.table = table

    def retrieve(self, query: str, k: int | None = None):
        return self.table[query][:k]


def mrr10(run, idx) -> float:
    return summarize(run, idx, KS)["mrr@10"]


def hit(run, idx, k: int) -> float:
    return summarize(run, idx, KS)["chunk_hit"][k]


def print_rescue(r: dict, items, vector_name: str, other_name: str, show: int) -> None:
    print(f"    k={r['k']:2d}: {vector_name} {pct(r['base_hit'])} | {other_name} {pct(r['other_hit'])} | "
          f"union {pct(r['union_hit'])} (+{r['union_gain'] * 100:.1f} pts) | "
          f"rescued {len(r['rescued'])}/{r['base_misses']} vector misses | lost {len(r['lost'])} | "
          f"both {r['both']} | neither {r['neither']}")
    for i in r["rescued"][:show]:
        print(f"        rescued: {items[i].id} | {items[i].doc} | {items[i].question[:70]!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--index-dir", type=Path, default=None)
    ap.add_argument("--questions-dir", type=Path, default=ROOT / "eval" / "questions")
    ap.add_argument("--skip-vector", action="store_true", help="BM25 only (no model, no Chroma)")
    ap.add_argument("--ablate-fields", action="store_true")
    ap.add_argument("--fields", default=",".join(DEFAULT_FIELDS),
                    help="BM25 field set for the main comparison")
    ap.add_argument("--k1", type=float, default=1.5)
    ap.add_argument("--b", type=float, default=0.75)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--depth", type=int, default=DEFAULT_DEPTH, help="candidates per retriever before RRF")
    ap.add_argument("--k-rrf", type=int, default=DEFAULT_K_RRF)
    ap.add_argument("--sensitivity", action="store_true", help="hybrid grid over k_rrf and depth")
    ap.add_argument("--show", type=int, default=10, help="rescued questions listed per cut-off")
    ap.add_argument("--min-union-gain", type=float, default=0.03)
    ap.add_argument("--out", type=Path, default=ROOT / "eval" / "results" / "retrieval_5a.json")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    s = load_settings(args.config)
    apply_offline_mode(s.embedding.offline)
    logging.basicConfig(level=logging.WARNING)
    quiet_noisy_loggers()
    index_dir = args.index_dir or s.paths.vector_db_dir
    if not index_dir.is_absolute():
        index_dir = ROOT / index_dir
    if load_manifest(index_dir / MANIFEST_FILE) is None:
        print(f"ERROR: no complete index in {index_dir}. Run: python scripts/build_index.py")
        return 2
    chunks = read_chunks(index_dir / CHUNKS_FILE)
    by_id = {c.chunk_id: c for c in chunks}

    items, unmatched = build_items(load_questions(args.questions_dir), by_id)
    if not items:
        print("ERROR: no evaluable questions (check --questions-dir)")
        return 2
    all_idx = list(range(len(items)))
    dev_idx, test_idx = split_indices(items, args.seed)
    ident_idx = [i for i, it in enumerate(items) if it.has_identifier]
    subsets = {"all": all_idx, "dev half": dev_idx, "test half": test_idx,
               "identifier questions": ident_idx}
    print(f"Index: {index_dir} ({len(chunks)} chunks)")
    print(f"Evaluable English answerable questions: {len(items)} "
          f"(dropped {len(unmatched)} whose quote is not inside a single chunk) | "
          f"identifier questions: {len(ident_idx)} | split seed {args.seed}: "
          f"dev {len(dev_idx)} / test {len(test_idx)}")

    fields = tuple(f.strip() for f in args.fields.split(",") if f.strip())
    runs, report = {}, {"n": len(items), "unmatched_gold": unmatched, "fields": fields,
                        "k1": args.k1, "b": args.b, "systems": {}, "rescue": {}}

    vector = None
    if not args.skip_vector:
        from sia.retrieval.vector_retriever import VectorRetriever
        try:
            vector = VectorRetriever.from_index(index_dir, s)
        except IndexNotReadyError as exc:
            print(f"ERROR: {exc}")
            return 2
        print(f"\nLoading the embedding model (cold start {vector.warmup():.1f} s)...")
        runs["vector"] = evaluate_system("vector", vector, items, k=max(KS))

    bm25 = Bm25Retriever(chunks, top_k=s.retrieval.top_k, k1=args.k1, b=args.b, fields=fields)
    print(f"BM25 ({'+'.join(fields)}, k1={args.k1}, b={args.b}): vocabulary {bm25.vocab_size}, "
          f"mean length {bm25.avg_len:.0f} tokens")
    runs["bm25"] = evaluate_system("bm25", bm25, items, k=max(KS))
    if vector is not None:
        hybrid = HybridRetriever(vector, bm25, top_k=s.retrieval.top_k, depth=args.depth, k_rrf=args.k_rrf)
        print(f"HYBRID (RRF, k_rrf={args.k_rrf}, depth={args.depth} per retriever)")
        runs["hybrid"] = evaluate_system("hybrid", hybrid, items, k=max(KS))
        report["hybrid"] = {"k_rrf": args.k_rrf, "depth": args.depth}

    names = list(runs)
    for name in names:
        lat = runs[name].latency_ms
        print(f"\n{name.upper()}  (mean retrieval latency {sum(lat) / len(lat):.1f} ms)")
        report["systems"][name] = {"latency_ms_mean": sum(lat) / len(lat)}
        for label, idx in subsets.items():
            summ = summarize(runs[name], idx, KS)
            print(row(label, summ))
            report["systems"][name][label] = summ

    if "vector" in runs:
        print("\nRESCUE ANALYSIS: does BM25 find gold chunks that the vector search misses?")
        for label, idx in subsets.items():
            print(f"  [{label}, n={len(idx)}]")
            report["rescue"][label] = {}
            for k in (5, 10):
                r = rescue_stats(runs["vector"], runs["bm25"], idx, k)
                print_rescue(r, items, "vector", "bm25", args.show if label == "all" else 0)
                report["rescue"][label][f"k{k}"] = {**r, "rescued": [items[i].id for i in r["rescued"]],
                                                    "lost": [items[i].id for i in r["lost"]]}

    if "hybrid" in runs:
        print("\nHYBRID vs THE UNION CEILING (union = gold found by vector OR BM25 in their own top-k)")
        report["hybrid_vs_union"] = {}
        for label, idx in subsets.items():
            parts = []
            for k in (5, 10):
                u = rescue_stats(runs["vector"], runs["bm25"], idx, k)["union_hit"]
                parts.append(f"k={k}: hybrid {pct(hit(runs['hybrid'], idx, k))} / union {pct(u)}")
            print(f"  {label:22s} " + " | ".join(parts))
            report["hybrid_vs_union"][label] = parts

    if args.sensitivity and vector is not None:
        pool = 100
        vec_lists = {it.question: vector.retrieve(it.question, k=pool) for it in items}
        bm_lists = {it.question: bm25.retrieve(it.question, k=pool) for it in items}
        cv, cb = CachedRetriever(vec_lists), CachedRetriever(bm_lists)
        print("\nSENSITIVITY (hybrid; informational, defaults are NOT chosen from this table)")
        print(f"  {'k_rrf':>5} {'depth':>5} | dev: MRR@10  Hit@5 Hit@10 | test: MRR@10  Hit@5 Hit@10")
        report["sensitivity"] = []
        for k_rrf in (10, 30, 60, 100):
            for depth in (20, 50, 100):
                run = evaluate_system("g", HybridRetriever(cv, cb, s.retrieval.top_k, depth, k_rrf),
                                      items, k=max(KS))
                d = (mrr10(run, dev_idx), hit(run, dev_idx, 5), hit(run, dev_idx, 10))
                t = (mrr10(run, test_idx), hit(run, test_idx, 5), hit(run, test_idx, 10))
                print(f"  {k_rrf:5d} {depth:5d} |      {d[0]:.3f} {pct(d[1])} {pct(d[2])} |"
                      f"       {t[0]:.3f} {pct(t[1])} {pct(t[2])}")
                report["sensitivity"].append({"k_rrf": k_rrf, "depth": depth, "dev": d, "test": t})

    if args.ablate_fields:
        print("\nFIELD ABLATION (BM25 only; every chunk of a document repeats its filename/title, "
              "so these fields raise df of those words and give all chunks of one document the same bonus)")
        report["ablation"] = {}
        for label, flds in ABLATION:
            r = Bm25Retriever(chunks, top_k=s.retrieval.top_k, k1=args.k1, b=args.b, fields=flds)
            run = evaluate_system(label, r, items, k=max(KS))
            print(f"\n  {label}")
            report["ablation"][label] = {}
            for sub in ("dev half", "test half", "identifier questions"):
                summ = summarize(run, subsets[sub], KS)
                print("  " + row(sub, summ))
                report["ablation"][label][sub] = summ
                if "vector" in runs:
                    rs = rescue_stats(runs["vector"], run, subsets[sub], 10)
                    print(f"      vs vector @10: rescued {len(rs['rescued'])}, lost {len(rs['lost'])}, "
                          f"union {pct(rs['union_hit'])} (+{rs['union_gain'] * 100:.1f} pts)")

    if "vector" in runs:
        r10 = rescue_stats(runs["vector"], runs["bm25"], all_idx, 10)
        c1 = r10["union_gain"] >= args.min_union_gain
        print("\nPRE-REGISTERED CRITERIA (informational)")
        print(f"  [{'PASS' if c1 else 'FAIL'}] C1 union Hit@10 gain {r10['union_gain'] * 100:+.1f} pts "
              f"(needed >= {args.min_union_gain * 100:.0f}); rescued {len(r10['rescued'])} gold chunks")
        if len(ident_idx) >= 10:
            v5 = summarize(runs["vector"], ident_idx, KS)["chunk_hit"][5]
            b5 = summarize(runs["bm25"], ident_idx, KS)["chunk_hit"][5]
            print(f"  [{'PASS' if b5 > v5 else 'FAIL'}] C2 identifier questions: BM25 Hit@5 {pct(b5)} "
                  f"vs vector {pct(v5)} (n={len(ident_idx)})")
        else:
            print(f"  [----] C2 inconclusive: only {len(ident_idx)} identifier questions (< 10)")

    if "hybrid" in runs:
        print("\nPRE-REGISTERED CRITERIA, phase 5b (informational)")
        ok_h1 = ok_h2 = True
        for label, idx in (("dev half", dev_idx), ("test half", test_idx)):
            h, v, b = (mrr10(runs[n], idx) for n in ("hybrid", "vector", "bm25"))
            h5, b5 = hit(runs["hybrid"], idx, 5), hit(runs["bm25"], idx, 5)
            ok_h1 &= h > max(v, b)
            ok_h2 &= h5 >= b5
            print(f"  [{label}] MRR@10 hybrid {h:.3f} vs vector {v:.3f} / bm25 {b:.3f} | "
                  f"Hit@5 hybrid {pct(h5)} vs bm25 {pct(b5)}")
        print(f"  [{'PASS' if ok_h1 else 'FAIL'}] H1 hybrid MRR@10 above the best single retriever on both halves")
        print(f"  [{'PASS' if ok_h2 else 'FAIL'}] H2 hybrid Hit@5 not below BM25 on both halves")
        report["criteria_5b"] = {"H1": ok_h1, "H2": ok_h2}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
