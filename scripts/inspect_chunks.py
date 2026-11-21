"""Inspect cleaning + chunking on the real documents (no embedding, no index).

Examples (PowerShell, from the project root):
  python scripts/inspect_chunks.py                         # stats table for all documents
  python scripts/inspect_chunks.py --file 800-61r3         # chunks of one document
  python scripts/inspect_chunks.py --file 800-61r3 --page 2 --raw   # raw page text (repr)
  python scripts/inspect_chunks.py --chunk-tokens 128 --overlap 20  # try another setting
"""
from __future__ import annotations

import argparse
import logging
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sia.ingestion.pipeline import ingest_directory            # noqa: E402
from sia.processing.chunker import ChunkingConfig, chunk_document  # noqa: E402
from sia.processing.cleaner import clean_pages                 # noqa: E402
from sia.processing.structure import annotate_chunks            # noqa: E402
from sia.processing.tokens import make_hf_counter, whitespace_counter  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docs-dir", type=Path, default=ROOT / "data" / "docs")
    ap.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifest.yaml")
    ap.add_argument("--file", help="only documents whose filename contains this text")
    ap.add_argument("--page", type=int, help="only this PDF page (with --file)")
    ap.add_argument("--raw", action="store_true", help="print repr() of the ingested page text")
    ap.add_argument("--show", type=int, default=5, help="how many chunks to print (with --file)")
    ap.add_argument("--chunk-tokens", type=int, default=256)
    ap.add_argument("--overlap", type=int, default=40)
    ap.add_argument("--min-tokens", type=int, default=40)
    ap.add_argument("--small", action="store_true",
                    help="list chunks shorter than --min-tokens (what are they?)")
    ap.add_argument("--sections", action="store_true",
                    help="print the detected section headings per page (audit of structure.py)")
    ap.add_argument("--no-clean", action="store_true", help="skip header/footer removal")
    ap.add_argument("--counter", choices=["bge", "whitespace"], default="bge")
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.ERROR)

    count = make_hf_counter() if args.counter == "bge" else whitespace_counter
    cfg = ChunkingConfig(args.chunk_tokens, args.overlap, args.min_tokens)
    hard_max = cfg.chunk_tokens + cfg.min_tokens

    report = ingest_directory(args.docs_dir, args.manifest)
    docs = [d for d in report.documents if d.status == "ok"
            and (not args.file or args.file.lower() in d.meta.filename.lower())]
    if not docs:
        print("No matching documents.")
        return 1

    single = args.file is not None
    rows, all_tokens, total_removed = [], [], 0
    n_with_section, total_blank, type_counts = 0, 0, {}
    for d in docs:
        pages, creport = (d.pages, None) if args.no_clean else clean_pages(d.pages)
        chunks = chunk_document(d.meta, pages, cfg, count)
        annotate_chunks(chunks, pages, d.meta.file_type)
        for c in chunks:
            c.doc_chunk_count = len(chunks)
        n_with_section += sum(1 for c in chunks if c.section)
        for c in chunks:
            type_counts[c.content_type] = type_counts.get(c.content_type, 0) + 1
        if creport and creport.blank_pages:
            total_blank += len(creport.blank_pages)
        toks = [c.n_tokens for c in chunks]
        all_tokens += toks
        removed = creport.lines_removed if creport else 0
        total_removed += removed
        small = sum(1 for t in toks if t < cfg.min_tokens)
        over = sum(1 for t in toks if t > hard_max)
        rows.append((d.meta.filename, len(d.pages), len(chunks),
                     min(toks), int(statistics.median(toks)), max(toks), small, over, removed))

        if args.small:
            for c in chunks:
                if c.n_tokens < cfg.min_tokens:
                    print(f"[small] {c.chunk_id} {d.meta.filename[:30]} p.{c.page} "
                          f"tokens={c.n_tokens} {c.text[:110]!r}")
        if args.sections:
            last = object()
            print(f"\n--- sections of {d.meta.filename} (page: section -> chunk location) ---")
            for c in chunks:
                if c.section != last:
                    print(f"  {c.location()}  [{c.content_type}]")
                    last = c.section
        if single:
            print(f"\n=== {d.meta.filename} | {d.meta.title}")
            if creport and creport.patterns:
                print("Removed header/footer lines (example -> pages):")
                for line, n in sorted(creport.patterns.items(), key=lambda kv: -kv[1]):
                    print(f"   {n:3d} x  {line!r}")
            if args.raw:
                page = args.page or next(p.page for p in d.pages if p.text)
                text = next(p.text for p in d.pages if p.page == page or (page and p.page is None))
                print(f"\n--- raw ingested text, page {page} (first 1500 chars, repr) ---")
                print(repr(text[:1500]))
            shown = [c for c in chunks if args.page is None or c.page == args.page]
            for c in shown[: args.show]:
                print(f"\n[{c.chunk_id}] {c.location()} tokens={c.n_tokens} type={c.content_type}")
                print(repr(c.text[:400] + ("..." if len(c.text) > 400 else "")))

    print(f"\nChunking: chunk={cfg.chunk_tokens} overlap={cfg.overlap_tokens} "
          f"min={cfg.min_tokens} (hard max {hard_max}) counter={args.counter} "
          f"clean={'off' if args.no_clean else 'on'}")
    print(f"{'FILE':58} {'PG':>3} {'CHUNKS':>6} {'MIN':>4} {'MED':>4} {'MAX':>4} {'SMALL':>5} {'OVER':>4} {'HDR':>4}")
    for r in rows:
        print(f"{r[0][:58]:58} {r[1]:>3} {r[2]:>6} {r[3]:>4} {r[4]:>4} {r[5]:>4} {r[6]:>5} {r[7]:>4} {r[8]:>4}")
    print(f"\nTOTAL docs={len(rows)} chunks={len(all_tokens)} "
          f"tokens median={int(statistics.median(all_tokens))} "
          f"p10={sorted(all_tokens)[len(all_tokens)//10]} "
          f"p90={sorted(all_tokens)[len(all_tokens)*9//10]} "
          f"header/footer lines removed={total_removed}")
    print(f"blank pages emptied={total_blank} | chunks with a section="
          f"{n_with_section} ({n_with_section / len(all_tokens):.0%}) | content types={type_counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
