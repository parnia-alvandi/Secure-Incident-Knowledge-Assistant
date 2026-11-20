"""Ingestion quality check.

  python scripts/ingest_check.py                       # table of all documents
  python scripts/ingest_check.py --show file.pdf --page 3   # view extracted text
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sia.config import load_settings  # noqa: E402
from sia.ingestion.pipeline import ingest_directory  # noqa: E402
from sia.logging_setup import setup_logging  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", help="filename whose extracted text to print")
    ap.add_argument("--page", type=int, default=None, help="page number (PDF only)")
    ap.add_argument("--chars", type=int, default=1500)
    args = ap.parse_args()

    s = load_settings()
    setup_logging(s.logging.level)
    rep = ingest_directory(s.paths.docs_dir, s.ingestion.manifest_path,
                           s.ingestion.min_chars_per_page, s.ingestion.supported_extensions)

    if args.show:
        doc = next((d for d in rep.documents if d.meta.filename == args.show), None)
        if doc is None:
            sys.exit(f"Not found: {args.show}")
        pages = [p for p in doc.pages if args.page in (None, p.page)]
        for p in pages[:3]:
            print(f"--- page {p.page} ({len(p.text)} chars) ---")
            print(p.text[: args.chars])
        return

    print(f"\n{'FILE':44} {'TYPE':5} {'PAGES':>5} {'CHARS':>8} {'STATUS':7} {'SUSP':>4} META")
    for d in rep.documents:
        m = d.meta
        meta = "complete" if not m.missing_fields else "MISSING: " + ",".join(m.missing_fields)
        print(f"{m.filename[:44]:44} {m.file_type:5} {len(d.pages):>5} {d.n_chars:>8} "
              f"{d.status:7} {len(d.suspicious_pages):>4} {meta}")
        if d.error:
            print(f"    -> {d.error}")

    print(f"\nTotal: {len(rep.documents)} | ok: {rep.count('ok')} | "
          f"empty: {rep.count('empty')} | error: {rep.count('error')}")
    if rep.manifest_orphans:
        print("Manifest entries without a file:", rep.manifest_orphans)
    if rep.skipped_files:
        print("Skipped (unsupported type):", rep.skipped_files)


if __name__ == "__main__":
    main()