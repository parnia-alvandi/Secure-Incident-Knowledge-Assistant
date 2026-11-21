"""Verify a built index: structure, embeddings, and a first retrieval smoke test.

Checks (structural ones decide the exit code):
  1. manifest exists (index complete) and matches the configured embedding model
  2. chunks.jsonl, manifest and Chroma agree on the number of chunks and on the ids
  3. stored vectors have the right dimension and unit norm
  4. re-embedding a few chunks reproduces the stored vectors (determinism)
  5. retrieval smoke test on the English answerable questions: chunk-level Hit@k and MRR,
     document-level Hit@k, latency. Informational: the final evaluation is phase 8.

Examples:
  python scripts/verify_index.py
  python scripts/verify_index.py --instruction both          # query prefix on / off
  python scripts/verify_index.py --index-dir vector_db_128   # ablation index
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np                                                    # noqa: E402

from sia.config import load_settings                                  # noqa: E402
from sia.evaluation.dataset import answerable, load_questions, squash  # noqa: E402
from sia.runtime import apply_offline_mode, quiet_noisy_loggers      # noqa: E402
from sia.storage.chunk_store import (CHUNKS_FILE, MANIFEST_FILE, load_manifest,  # noqa: E402
                                     read_chunks)

FAILS = 0


def check(ok: bool, label: str, detail: str = "") -> bool:
    global FAILS
    FAILS += 0 if ok else 1
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f" - {detail}" if detail else ""))
    return ok


def retrieval_smoke(embedder, store, chunks_by_id, doc_kind, questions, use_instruction, ks, max_miss):
    max_k = max(ks)
    ranks_chunk, ranks_doc, ranks_page, t_embed, t_query = [], [], [], [], []
    by_kind: dict[str, list[int | None]] = {}
    misses = []
    for q in questions:
        t0 = time.perf_counter()
        qv = embedder.embed_query(q["question"], use_instruction=use_instruction)
        t1 = time.perf_counter()
        hits = store.query(qv, k=max_k)
        t2 = time.perf_counter()
        t_embed.append(t1 - t0)
        t_query.append(t2 - t1)
        rank_chunk = rank_doc = rank_page = None
        for h in hits:
            c = chunks_by_id[h.chunk_id]
            if rank_doc is None and c.filename == q["doc"]:
                rank_doc = h.rank
            same_page = c.filename == q["doc"] and c.page == q.get("evidence_page")
            if rank_page is None and same_page:
                rank_page = h.rank
            if rank_chunk is None and same_page and q["sq"] in squash(c.text):
                rank_chunk = h.rank
        ranks_chunk.append(rank_chunk)
        ranks_doc.append(rank_doc)
        ranks_page.append(rank_page)
        by_kind.setdefault(doc_kind.get(q["doc"], "?"), []).append(rank_chunk)
        if rank_chunk is None and len(misses) < max_miss:
            top = chunks_by_id[hits[0].chunk_id]
            misses.append(f"{q['id']} | {q['doc']} p.{q.get('evidence_page')} | {q['question'][:70]!r}"
                          f" -> top1: {top.filename} p.{top.page} score={hits[0].score:.3f}")

    n = len(questions)

    def hit(ranks, k):
        return sum(1 for r in ranks if r is not None and r <= k) / n

    mrr = sum(1 / r for r in ranks_chunk if r) / n
    print(f"    questions={n} | latency: embed {np.mean(t_embed)*1000:.0f} ms, "
          f"chroma query {np.mean(t_query)*1000:.0f} ms (mean)")
    print("    chunk-level Hit@k : " + "  ".join(f"@{k}={hit(ranks_chunk, k):.1%}" for k in ks)
          + f"  MRR@{max_k}={mrr:.3f}")
    print("    page-level  Hit@k : " + "  ".join(f"@{k}={hit(ranks_page, k):.1%}" for k in ks)
          + "   (right document and page, any chunk)")
    print("    doc-level   Hit@k : " + "  ".join(f"@{k}={hit(ranks_doc, k):.1%}" for k in ks))
    for kind, ranks in sorted(by_kind.items()):
        ok5 = sum(1 for r in ranks if r is not None and r <= 5) / len(ranks)
        print(f"    chunk Hit@5 for {kind:9s}: {ok5:.1%} ({len(ranks)} questions)")
    for m in misses:
        print(f"    miss: {m}")
    return hit(ranks_chunk, 5)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--index-dir", type=Path, default=None)
    ap.add_argument("--questions-dir", type=Path, default=ROOT / "eval" / "questions")
    ap.add_argument("--instruction", choices=["on", "off", "both"], default="on")
    ap.add_argument("--skip-retrieval", action="store_true")
    ap.add_argument("--max-miss", type=int, default=8)
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    s = load_settings(args.config)
    apply_offline_mode(s.embedding.offline)
    quiet_noisy_loggers()
    index_dir = args.index_dir or s.paths.vector_db_dir
    if not index_dir.is_absolute():
        index_dir = ROOT / index_dir
    print(f"Index: {index_dir}")

    manifest = load_manifest(index_dir / MANIFEST_FILE)
    if not check(manifest is not None, "manifest present (index complete)"):
        return 1
    print(f"    built {manifest.created_at} | model {manifest.embedding_model} | "
          f"dim {manifest.embedding_dim} | chunking {manifest.chunking} | "
          f"{manifest.n_documents} docs, {manifest.n_chunks} chunks")
    print(f"    timings {manifest.timings}")
    check(manifest.embedding_model == s.models.embedding, "index model == configured model",
          f"index={manifest.embedding_model} config={s.models.embedding}")

    chunks = read_chunks(index_dir / CHUNKS_FILE)
    by_id = {c.chunk_id: c for c in chunks}
    check(len(chunks) == manifest.n_chunks == len(by_id), "chunks.jsonl count == manifest count, ids unique",
          f"jsonl={len(chunks)} manifest={manifest.n_chunks} unique={len(by_id)}")

    from sia.embedding.embedder import Embedder                       # lazy: heavy imports
    from sia.storage.vector_store import ChromaVectorStore

    store = ChromaVectorStore(index_dir / "chroma", manifest.collection)
    check(store.count() == len(chunks), "Chroma count == chunks.jsonl count", f"chroma={store.count()}")
    check(set(store.ids()) == set(by_id), "Chroma ids == chunk ids")

    rng = random.Random(0)
    sample = rng.sample(sorted(by_id), min(200, len(by_id)))
    vecs = store.get_embeddings(sample)
    norms = np.linalg.norm(vecs, axis=1)
    check(vecs.shape[1] == manifest.embedding_dim, "vector dimension", f"{vecs.shape[1]}")
    check(float(np.max(np.abs(norms - 1))) < 1e-3, "vectors are unit-norm",
          f"max |norm-1| = {float(np.max(np.abs(norms - 1))):.2e}")

    embedder = Embedder(manifest.embedding_model, s.embedding.batch_size,
                        s.embedding.max_seq_length, manifest.query_instruction)
    probe = sample[:5]
    again = embedder.embed_passages([by_id[i].embed_text() for i in probe])
    cos = np.sum(again * store.get_embeddings(probe), axis=1)
    check(float(cos.min()) > 0.999, "re-embedding reproduces stored vectors", f"min cosine {cos.min():.6f}")

    if not args.skip_retrieval:
        qs = answerable(load_questions(args.questions_dir), lang="en")
        doc_kind = {d["filename"]: d.get("doc_kind") or "?" for d in manifest.documents}
        modes = {"on": [True], "off": [False], "both": [True, False]}[args.instruction]
        for flag in modes:
            print(f"\nRetrieval smoke test, query instruction {'ON' if flag else 'OFF'} "
                  f"(English answerable questions; caveat: questions were written from the passages)")
            retrieval_smoke(embedder, store, by_id, doc_kind, qs, flag, (1, 3, 5, 10), args.max_miss)

    print(f"\n{'ALL STRUCTURAL CHECKS PASSED' if FAILS == 0 else f'{FAILS} STRUCTURAL CHECK(S) FAILED'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
