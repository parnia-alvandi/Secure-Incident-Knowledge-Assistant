"""Download (once) and sanity-check the cross-encoder re-ranker (phase 5c).

The project normally runs offline (embedding.offline: true). The re-ranker is a separate model,
so if it is not in the local Hugging Face cache yet, run this ONCE with network access:

    python scripts/fetch_reranker.py

It ignores embedding.offline, downloads models.reranker, scores two example pairs and prints
the time and the score range. Afterwards everything works offline again.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sia.config import load_settings                              # noqa: E402
from sia.retrieval.reranker import CrossEncoderReranker, RerankerUnavailableError   # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=None)
    args = ap.parse_args()
    s = load_settings(args.config)
    print(f"Model: {s.models.reranker}")
    reranker = CrossEncoderReranker(s.models.reranker)
    t0 = time.perf_counter()
    try:
        scores = reranker.score(
            "How should an organization contain a ransomware infection?",
            ["Isolate the infected hosts from the network immediately to stop ransomware from spreading.",
             "The cafeteria menu changes every Tuesday and includes vegetarian options."])
    except RerankerUnavailableError as exc:
        print(f"ERROR: {exc}")
        return 2
    print(f"Loaded and scored in {time.perf_counter() - t0:.1f} s")
    print(f"  relevant passage:   {scores[0]:+.3f}\n  irrelevant passage: {scores[1]:+.3f}")
    print("OK: the relevant passage should score clearly higher." if scores[0] > scores[1]
          else "WARNING: unexpected order, check the model name in config.yaml.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
