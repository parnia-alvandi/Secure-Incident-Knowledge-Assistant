"""Verify the ground truth of the evaluation questions.

For every answerable question, the `evidence_quote` must really exist in the
document text produced by OUR ingestion pipeline (and on the stated page, for PDFs).
A question whose quote cannot be found is rejected, so a wrong or invented
ground truth can never enter the test set.

Usage (from the project root, venv active):
    python scripts/verify_questions.py
    python scripts/verify_questions.py --dir eval/questions
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sia.config import PROJECT_ROOT, load_settings  
from sia.ingestion.pipeline import ingest_directory  
from sia.logging_setup import setup_logging  

REQUIRED_KEYS = ("id", "doc", "question", "answer", "evidence_quote",
                 "evidence_page", "type", "answerable", "lang")
VALID_TYPES = {"factual", "numeric", "list", "procedure"}
MIN_QUOTE_CHARS = 25   
MIN_ABSENT_TERM_CHARS = 6   


def squash(text: str) -> str:
    """Lowercase and keep only letters/digits, so line breaks, spacing, hyphenation
    and quote styles in the PDF text cannot cause false mismatches."""
    text = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"[^0-9a-z]+", "", text)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="eval/questions")
    args = ap.parse_args()

    s = load_settings()
    setup_logging("WARNING")
    rep = ingest_directory(s.paths.docs_dir, s.ingestion.manifest_path,
                           s.ingestion.min_chars_per_page, s.ingestion.supported_extensions)

    corpus = {d.meta.filename: {p.page: squash(p.text) for p in d.pages}
              for d in rep.documents if d.status == "ok"}

    full_text = {fn: "".join(pages.values()) for fn, pages in corpus.items()}

    qdir = Path(args.dir)
    if not qdir.is_absolute():
        qdir = PROJECT_ROOT / qdir
    files = sorted(qdir.glob("*.json"))
    if not files:
        print(f"No question files found in {qdir}")
        return 1

    problems: list[str] = []
    seen_ids: set[str] = set()
    per_doc: dict[str, int] = {}
    n_answerable = n_unanswerable = 0

    for fp in files:
        try:
            items = json.loads(fp.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            problems.append(f"{fp.name}: invalid JSON ({exc})")
            continue
        for it in items:
            qid = it.get("id", "<no id>")
            missing = [k for k in REQUIRED_KEYS if k not in it]
            if missing:
                problems.append(f"{qid}: missing keys {missing}")
                continue
            if qid in seen_ids:
                problems.append(f"{qid}: duplicate id")
            seen_ids.add(qid)
            if it["type"] not in VALID_TYPES:
                problems.append(f"{qid}: invalid type '{it['type']}'")

            if not it["answerable"]:
                n_unanswerable += 1
                terms = it.get("absent_terms")
                if not terms or not isinstance(terms, list):
                    problems.append(f"{qid}: unanswerable question needs a non-empty 'absent_terms' list")
                    continue
                for term in terms:
                    sq = squash(term)
                    if len(sq) < MIN_ABSENT_TERM_CHARS:
                        problems.append(f"{qid}: absent term '{term}' is too short to be reliable")
                        continue
                    hits = [fn for fn, txt in full_text.items() if sq in txt]
                    if hits:
                        problems.append(f"{qid}: absent term '{term}' FOUND in {hits} -> question may be answerable")
                continue
            n_answerable += 1
            per_doc[it["doc"]] = per_doc.get(it["doc"], 0) + 1

            pages = corpus.get(it["doc"])
            if pages is None:
                problems.append(f"{qid}: document '{it['doc']}' not found or not ingested")
                continue
            quote = squash(it["evidence_quote"] or "")
            if len(quote) < MIN_QUOTE_CHARS:
                problems.append(f"{qid}: evidence_quote too short/empty")
                continue
            found = [pg for pg, txt in pages.items() if quote in txt]
            if not found:
                problems.append(f"{qid}: quote NOT FOUND in {it['doc']}")
            elif it["evidence_page"] is not None and it["evidence_page"] not in found:
                problems.append(f"{qid}: WRONG PAGE (stated p.{it['evidence_page']}, found on {found})")

    print(f"\nFiles: {len(files)} | answerable: {n_answerable} | unanswerable: {n_unanswerable}")
    for doc, n in sorted(per_doc.items()):
        print(f"  {n:>3} answerable questions  <- {doc}")
    if problems:
        print(f"\n{len(problems)} PROBLEM(S):")
        for p in problems:
            print("  -", p)
        return 1
    print("\nAll evidence quotes verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
