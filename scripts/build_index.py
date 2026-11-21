"""Build (or rebuild) the local index from data/docs. The index is always rebuilt from scratch.

Examples (PowerShell, project root):
  python scripts/build_index.py --dry-run          # chunk + annotate only, writes nothing
  python scripts/build_index.py                    # full build into paths.vector_db_dir
  # ablation: other chunk size into a separate directory
  python scripts/build_index.py --chunk-tokens 128 --overlap 20 --min-tokens 20 --out-dir vector_db_128
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import psutil                                                    # noqa: E402

from sia.config import load_settings                             # noqa: E402
from sia.indexing.builder import build_index, collect_chunks     # noqa: E402
from sia.ingestion.pipeline import ingest_directory              # noqa: E402
from sia.processing.chunker import ChunkingConfig               # noqa: E402
from sia.processing.tokens import make_hf_counter               # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="index directory (default: paths.vector_db_dir)")
    ap.add_argument("--chunk-tokens", type=int, default=None)
    ap.add_argument("--overlap", type=int, default=None)
    ap.add_argument("--min-tokens", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true", help="chunk and annotate only; no embedding, no files")
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    s = load_settings(args.config)
    logging.basicConfig(level=getattr(logging, s.logging.level, logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = ChunkingConfig(args.chunk_tokens or s.chunking.chunk_tokens,
                         args.overlap if args.overlap is not None else s.chunking.overlap_tokens,
                         args.min_tokens if args.min_tokens is not None else s.chunking.min_tokens)
    out_dir = args.out_dir or s.paths.vector_db_dir
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir

    report = ingest_directory(s.paths.docs_dir, s.ingestion.manifest_path,
                              s.ingestion.min_chars_per_page, s.ingestion.supported_extensions)
    docs = [d for d in report.documents if d.status == "ok"]
    skipped = [{"filename": d.meta.filename, "status": d.status, "error": d.error}
               for d in report.documents if d.status != "ok"]
    for sk in skipped:
        print(f"SKIPPED {sk['filename']}: {sk['status']} ({sk['error']})")
    print(f"Documents: {len(docs)} ok, {len(skipped)} skipped | "
          f"chunking {cfg.chunk_tokens}/{cfg.overlap_tokens}/{cfg.min_tokens} | "
          f"model {s.models.embedding} | out {out_dir}")

    count = make_hf_counter(s.models.embedding)
    if args.dry_run:
        chunks, infos = collect_chunks(docs, cfg, count)
        n_sec = sum(i["chunks_with_section"] for i in infos)
        types: dict[str, int] = {}
        for i in infos:
            for k, v in i["content_types"].items():
                types[k] = types.get(k, 0) + v
        print(f"DRY RUN: {len(chunks)} chunks | with section: {n_sec} ({n_sec / len(chunks):.0%}) | "
              f"content types: {types} | blank pages emptied: "
              f"{sum(len(i['blank_pages']) for i in infos)}")
        return 0

    from sia.embedding.embedder import Embedder                  # lazy imports: heavy
    from sia.storage.vector_store import ChromaVectorStore

    embedder = Embedder(s.models.embedding, s.embedding.batch_size, s.embedding.max_seq_length,
                        s.embedding.query_instruction)
    store = ChromaVectorStore(out_dir / "chroma", s.index.collection_name)
    manifest = build_index(docs, skipped, chunk_cfg=cfg, count_tokens=count, embedder=embedder,
                           store=store, out_dir=out_dir, show_progress=True)
    rss = psutil.Process().memory_info().rss / 1024**2
    print(f"\nINDEX BUILT: {manifest.n_chunks} chunks from {manifest.n_documents} documents, "
          f"dim={manifest.embedding_dim}")
    print(f"timings: {manifest.timings} | process RSS now: {rss:.0f} MB")
    print(f"files: {out_dir / 'chunks.jsonl'}, {out_dir / 'index_manifest.json'}, {out_dir / 'chroma'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
