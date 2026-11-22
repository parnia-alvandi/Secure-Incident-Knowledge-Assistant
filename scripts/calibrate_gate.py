"""Calibrate the evidence gate (retrieval.min_score) from retrieval scores only; no LLM needed.

Compares the best retrieval score of answerable and unanswerable English questions, finds the
threshold that keeps a target share of answerable questions, and validates it on held-out halves.

  python scripts/calibrate_gate.py
  python scripts/calibrate_gate.py --retention 0.95
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sia.config import load_settings                                   # noqa: E402
from sia.evaluation.dataset import load_questions                      # noqa: E402
from sia.evaluation.gate import (auc, blocked, percentile, retained, score_questions,  # noqa: E402
                                 split_half, threshold_for_retention)
from sia.retrieval.vector_retriever import IndexNotReadyError, VectorRetriever          # noqa: E402
from sia.runtime import apply_offline_mode, quiet_noisy_loggers                         # noqa: E402

FEATURES = ("top1", "mean3", "margin")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--index-dir", type=Path, default=None)
    ap.add_argument("--questions-dir", type=Path, default=ROOT / "eval" / "questions")
    ap.add_argument("--retention", type=float, default=0.98,
                    help="share of answerable questions the gate must let through")
    ap.add_argument("--splits", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "eval" / "results" / "gate_calibration.json")
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
    retriever.warmup()

    qs = [q for q in load_questions(args.questions_dir) if q.get("lang") == "en"]
    ans = score_questions(retriever, [q for q in qs if q.get("answerable")])
    una = score_questions(retriever, [q for q in qs if not q.get("answerable")])
    cats = sorted({r["category"] for r in una if r["category"]})
    print(f"English questions scored: {len(ans)} answerable, {len(una)} unanswerable {cats}")

    print("\nBest retrieval score (top1) distribution:")
    for name, rows in [("answerable", ans)] + [(f"unanswerable/{c}", [r for r in una if r["category"] == c])
                                              for c in cats]:
        v = [r["top1"] for r in rows]
        print(f"  {name:28s} n={len(v):3d}  min={min(v):.3f}  p10={percentile(v, 10):.3f}  "
              f"median={percentile(v, 50):.3f}  p90={percentile(v, 90):.3f}  max={max(v):.3f}")

    print("\nHow well does each signal separate answerable from unanswerable? (AUC: 0.5 = chance, 1.0 = perfect)")
    for f in FEATURES:
        row = [f"all={auc([r[f] for r in ans], [r[f] for r in una]):.3f}"]
        for c in cats:
            row.append(f"{c}={auc([r[f] for r in ans], [r[f] for r in una if r['category'] == c]):.3f}")
        print(f"  {f:7s} " + "  ".join(row))

    a_top = [r["top1"] for r in ans]
    print("\nThreshold on top1, calibrated and measured on ALL data (optimistic):")
    print(f"  {'retention target':>16} {'tau':>7} {'answerable kept':>16} " +
          " ".join(f"{'blocked ' + c:>22}" for c in cats))
    summary = {}
    for target in (1.0, 0.99, 0.98, 0.95, 0.90):
        tau = threshold_for_retention(a_top, target)
        per_cat = {c: blocked([r["top1"] for r in una if r["category"] == c], tau) for c in cats}
        print(f"  {target:>16.2f} {tau:>7.3f} {retained(a_top, tau):>16.1%} " +
              " ".join(f"{per_cat[c]:>22.1%}" for c in cats))
        summary[str(target)] = {"tau": tau, "kept": retained(a_top, tau), "blocked": per_cat}

    print(f"\nHeld-out check: tau fitted on one random half, measured on the other half "
          f"({args.splits} splits, retention target {args.retention:.2f}):")
    rng = random.Random(args.seed)
    kept_all, blocked_all = [], {c: [] for c in cats}
    for _ in range(args.splits):
        a_fit, a_test = split_half(a_top, rng)
        tau = threshold_for_retention(a_fit, args.retention)
        kept_all.append(retained(a_test, tau))
        for c in cats:
            u = [r["top1"] for r in una if r["category"] == c]
            u_fit, u_test = split_half(u, rng)
            blocked_all[c].append(blocked(u_test, tau))
    print(f"  answerable kept (held-out): mean {sum(kept_all) / len(kept_all):.1%}  "
          f"min {min(kept_all):.1%}  max {max(kept_all):.1%}")
    for c in cats:
        b = blocked_all[c]
        print(f"  blocked {c:14s} (held-out): mean {sum(b) / len(b):.1%}  min {min(b):.1%}  max {max(b):.1%}")

    tau = threshold_for_retention(a_top, args.retention)
    print(f"\nHardest unanswerable questions (highest top1; these would pass the gate at tau={tau:.3f}):")
    for r in sorted(una, key=lambda r: -r["top1"])[:8]:
        print(f"  {r['id']:8s} {str(r['category']):13s} top1={r['top1']:.3f}  {r['question'][:64]!r}"
              f"\n           -> {r['top1_title'][:50]} | {r['top1_loc']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"retention_target": args.retention, "tau": tau,
                                    "summary": summary, "answerable": ans, "unanswerable": una},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSuggested retrieval.min_score for retention {args.retention:.2f}: {tau:.3f} "
          f"(NOT applied; decide together after seeing the table). Details: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
