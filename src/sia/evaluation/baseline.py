"""Phase 4 baseline evaluation helpers: question selection, gold detection, metrics, review file.

Pure functions (no models); scripts/eval_baseline.py runs the pipeline and calls these.
Nothing here judges the *correctness* of an answer text: that is done by a human in the review
file (an automatic faithfulness check is phase 7/8). What is automatic: status distribution,
whether the gold passage was retrieved / cited, and a simple "numbers not found in the passages"
signal.
"""
from __future__ import annotations

import math
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from sia.answering.parse import has_answer_text
from sia.evaluation.dataset import squash

_NUM = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?%?")
_CITE = re.compile(r"\[\d+(?:\s*[,;]\s*\d+)*\]")


def doc_kind_of(filename: str | None) -> str:
    return "synthetic" if (filename or "").startswith("syn_") else "public"


def select_questions(questions: list[dict], n_answerable: int = 40, seed: int = 42) -> list[dict]:
    """All English unanswerable questions + a seeded sample of answerable ones
    (half public documents, half synthetic), in a seeded shuffled order."""
    en = [q for q in questions if q.get("lang") == "en"]
    una = [q for q in en if not q.get("answerable")]
    ans = [q for q in en if q.get("answerable") and q.get("evidence_quote")]
    rng = random.Random(seed)
    pub = sorted((q for q in ans if doc_kind_of(q.get("doc")) == "public"), key=lambda q: q["id"])
    syn = sorted((q for q in ans if doc_kind_of(q.get("doc")) == "synthetic"), key=lambda q: q["id"])
    n_pub = n_answerable // 2
    picked = rng.sample(pub, min(n_pub, len(pub))) + \
        rng.sample(syn, min(n_answerable - n_pub, len(syn)))
    out = una + picked
    rng.shuffle(out)
    return out


def gold_chunk_ids(question: dict, chunks_by_id: dict) -> set[str]:
    """Chunks of the right document and page that contain the evidence quote."""
    if not question.get("answerable") or not question.get("evidence_quote"):
        return set()
    sq = squash(question["evidence_quote"])
    return {cid for cid, c in chunks_by_id.items()
            if c.filename == question["doc"] and c.page == question.get("evidence_page")
            and sq in squash(c.text)}


def _norm_number(token: str) -> str:
    return token.replace(",", "").rstrip("%").rstrip(".")


def numbers_in(text: str) -> set[str]:
    out = set()
    for tok in _NUM.findall(text):
        n = _norm_number(tok)
        if len(n) > 1 or "." in n:                 # ignore single digits (list numbering)
            out.add(n)
    return out


def unsupported_numbers(answer: str, passages: list[str]) -> list[str]:
    """Numbers (2+ digits, decimals, percentages) in the answer that appear in none of the passages."""
    return sorted(numbers_in(_CITE.sub(" ", answer)) - numbers_in(" ".join(passages)))


def load_done(path: Path) -> dict[str, dict]:
    import json
    done: dict[str, dict] = {}
    if Path(path).is_file():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["id"]] = row
    return done


def _pct(n: int, d: int) -> str:
    return f"{n}/{d} ({n / d:.0%})" if d else "n/a"


def summarize(rows: list[dict]) -> dict:
    una = [r for r in rows if not r["answerable"]]
    ans = [r for r in rows if r["answerable"]]
    s: dict = {"n": len(rows), "n_unanswerable": len(una), "n_answerable": len(ans)}

    def status_count(rs):
        return Counter(r["status"] for r in rs)

    s["unanswerable"] = {"status": dict(status_count(una)), "by_category": {}}
    by_cat = defaultdict(list)
    for r in una:
        by_cat[r.get("category") or "?"].append(r)
    for c, rs in by_cat.items():
        s["unanswerable"]["by_category"][c] = {"n": len(rs), "status": dict(status_count(rs))}

    answered_ans = [r for r in ans if r["status"] == "answered"]
    s["answerable"] = {
        "status": dict(status_count(ans)),
        "gold_in_prompt": sum(1 for r in ans if r.get("gold_in_prompt")),
        "answered": len(answered_ans),
        "cited_gold": sum(1 for r in answered_ans if r.get("cited_gold")),
        "answered_with_unsupported_numbers": sum(1 for r in answered_ans if r.get("unsupported_numbers")),
    }
    s["unanswerable"]["answered_with_unsupported_numbers"] = sum(
        1 for r in una if r["status"] == "answered" and r.get("unsupported_numbers"))

    def no_text(r):                      # produced something, but no answer text (only citation markers)
        return r["status"] == "no_answer" or (
            r["status"] in ("answered", "unverified") and not has_answer_text(r.get("answer", "")))

    s["unanswerable"]["without_text"] = sum(1 for r in una if no_text(r))
    s["unanswerable"]["answered_with_text"] = sum(
        1 for r in una if r["status"] == "answered" and not no_text(r))
    s["answerable"]["without_text"] = sum(1 for r in ans if no_text(r))
    with_text = [r for r in answered_ans if not no_text(r)]
    s["answerable"]["answered_with_text"] = len(with_text)
    s["answerable"]["cited_gold_with_text"] = sum(1 for r in with_text if r.get("cited_gold"))
    s["retries"] = sum(1 for r in rows if r.get("timings", {}).get("retries"))

    totals = [r["timings"]["total_s"] for r in rows if r.get("timings", {}).get("total_s") is not None]
    llm_rows = [r for r in rows if "prompt_tokens_per_s" in r.get("timings", {})
                and r["timings"]["prompt_tokens_per_s"]]
    s["latency"] = {
        "mean_s": round(statistics.mean(totals), 1) if totals else None,
        "p50_s": round(statistics.median(totals), 1) if totals else None,
        "p90_s": round(sorted(totals)[math.ceil(0.9 * len(totals)) - 1], 1) if totals else None,   # nearest rank
        "llm_calls": len(llm_rows),
        "prompt_tok_per_s": round(statistics.mean(r["timings"]["prompt_tokens_per_s"] for r in llm_rows), 1)
        if llm_rows else None,
    }
    return s


def format_summary(s: dict) -> str:
    u, a = s["unanswerable"], s["answerable"]
    n_u, n_a = s["n_unanswerable"], s["n_answerable"]
    lines = [f"Questions evaluated: {s['n']} ({n_u} unanswerable, {n_a} answerable)", ""]
    ref = u["status"].get("insufficient_evidence", 0)
    lines.append(f"UNANSWERABLE: correct refusals {_pct(ref, n_u)}; "
                 f"wrongly answered {_pct(u['status'].get('answered', 0) + u['status'].get('unverified', 0), n_u)}; "
                 f"errors {u['status'].get('error', 0)}")
    for c, d in sorted(u["by_category"].items()):
        lines.append(f"   {c:13s} refused {_pct(d['status'].get('insufficient_evidence', 0), d['n'])}; "
                     f"answered {_pct(d['status'].get('answered', 0), d['n'])}; "
                     f"unverified {_pct(d['status'].get('unverified', 0), d['n'])}")
    lines.append(f"   answered although unanswerable, WITH real answer text: {u['answered_with_text']}; "
                 f"and with numbers absent from the passages: {u['answered_with_unsupported_numbers']}")
    lines.append(f"   replies without answer text (citation markers only / no_answer): {u['without_text']}")
    lines.append("")
    st = a["status"]
    lines.append(f"ANSWERABLE: answered {_pct(st.get('answered', 0), n_a)}; "
                 f"wrongly refused {_pct(st.get('insufficient_evidence', 0), n_a)}; "
                 f"unverified {_pct(st.get('unverified', 0), n_a)}; "
                 f"no_answer {_pct(st.get('no_answer', 0), n_a)}; errors {st.get('error', 0)}")
    lines.append(f"   replies without answer text (citation markers only / no_answer): {_pct(a['without_text'], n_a)}")
    lines.append(f"   answered WITH real answer text: {_pct(a['answered_with_text'], n_a)}")
    lines.append(f"   gold passage in the prompt (retrieval top-k): {_pct(a['gold_in_prompt'], n_a)}")
    lines.append(f"   among answered: cites the gold passage {_pct(a['cited_gold'], a['answered'])} "
                 f"(with real text: {_pct(a['cited_gold_with_text'], a['answered_with_text'])}); "
                 f"numbers not found in the passages {_pct(a['answered_with_unsupported_numbers'], a['answered'])}")
    if s.get("retries"):
        lines.append(f"   repair retries used on {s['retries']} question(s)")
    lat = s["latency"]
    lines += ["", f"LATENCY (s, whole question): mean {lat['mean_s']}  p50 {lat['p50_s']}  p90 {lat['p90_s']} "
                  f"| LLM calls {lat['llm_calls']} | prefill {lat['prompt_tok_per_s']} tok/s",
              "", "Answer TEXT correctness is judged by hand: see the review file."]
    return "\n".join(lines)


def review_markdown(rows: list[dict], summary_text: str) -> str:
    out = ["# Baseline RAG evaluation: review sheet", "", "```", summary_text, "```", "",
           "Mark each answered question: correct / partly / wrong, and whether the cited passage "
           "really supports the answer.", ""]
    for r in rows:
        kind = r.get("category") if not r["answerable"] else r.get("kind")
        out += [f"## {r['id']}  ({'unanswerable: ' if not r['answerable'] else 'answerable, '}{kind})",
                f"- **Question:** {r['question']}",
                f"- **Gold answer:** {r.get('gold_answer')}",
                f"- **System status:** `{r['status']}`" + (f" ({r['reason']})" if r.get("reason") else ""),
                f"- **System answer:** {r['answer'] or '(none)'}"]
        if r["answerable"]:
            out.append(f"- **Gold passage in prompt:** {r.get('gold_in_prompt')} | "
                       f"**cites gold passage:** {r.get('cited_gold')}"
                       f" | gold: {r.get('gold_doc')} p.{r.get('gold_page')}")
        if r.get("unsupported_numbers"):
            out.append(f"- **Numbers not found in the passages:** {r['unsupported_numbers']}")
        for src in r.get("sources", []):
            out.append(f"- cited [{src['n']}] {src['title'][:60]} | {src['filename']} | {src['location']}")
        out.append(f"- **Your verdict:** [ ] correct  [ ] partly  [ ] wrong      notes:")
        out.append("")
    return "\n".join(out)


# --- demo document (Task 2): built from recorded results, never from a new run -------------

def demo_selection(selected: list[dict], rows_by_id: dict[str, dict],
                   n_answerable: int = 5, n_unanswerable: int = 1) -> list[dict]:
    """Pre-registered rule: the first `n_answerable` answerable and the first `n_unanswerable`
    unanswerable questions in the seeded evaluation order (not chosen by outcome)."""
    ans: list[dict] = []
    una: list[dict] = []
    for q in selected:
        row = rows_by_id.get(q["id"])
        if row is None:
            continue
        if row["answerable"] and len(ans) < n_answerable:
            ans.append(row)
        elif not row["answerable"] and len(una) < n_unanswerable:
            una.append(row)
    return ans + una


VERDICT_LABEL = {"correct": "correct", "partly": "partly correct", "wrong": "wrong"}


def demo_markdown(demo_rows: list[dict], summary_text: str, setup: dict, verdicts: dict,
                  n_answerable: int, seed: int) -> str:
    out = ["# Baseline RAG demo (Task 2)", "",
           "All outputs below were produced by `scripts/eval_baseline.py` on this machine and are "
           "reproduced unedited. Nothing was re-run or hand-picked for this document.", "",
           "## Setup", ""]
    out += [f"- {k}: {v}" for k, v in setup.items()]
    out += ["", "## How the questions were chosen", "",
            f"The evaluation set is every English unanswerable question plus {n_answerable} answerable ones "
            f"(half public documents, half synthetic), sampled with seed {seed} and shuffled with the same "
            f"seed. This demo shows the **first 5 answerable and the first unanswerable question in that "
            f"order**. The rule was fixed before looking at outcomes; the full-set results are in the next "
            f"section.", "",
            "`answered` means: the model produced an answer with at least one valid citation. It does NOT "
            "mean the answer is correct. The verdicts are a human judgement.", ""]
    for r in demo_rows:
        kind = "unanswerable (not in the collection)" if not r["answerable"] else "answerable"
        out += [f"## {r['id']} ({kind})", "", f"**Question:** {r['question']}", ""]
        if r["answerable"]:
            out += [f"**Reference answer:** {r.get('gold_answer')}  ",
                    f"**Reference location:** {r.get('gold_doc')}, p. {r.get('gold_page')}", ""]
        out += [f"**System status:** `{r['status']}`"
                + (f" ({r['reason']})" if r.get("reason") else ""), "",
                f"**System answer:** {r['answer'] or '(none)'}", ""]
        for w in r.get("warnings", []):
            out.append(f"> Warning: {w}")
        if r.get("sources"):
            out += ["", "**Sources cited by the answer:**", ""]
            for s in r["sources"]:
                out += [f"- [{s['n']}] *{s['title']}*, `{s['filename']}`, {s['location']}",
                        f"  > {s['excerpt']}"]
        t = r.get("timings", {})
        out += ["", f"**Timing:** total {t.get('total_s')} s; retrieval {t.get('retrieval_s')} s; "
                    f"LLM {t.get('llm_total_s', 'n/a')} s; best retrieval score {t.get('best_score')}"]
        v = verdicts.get(r["id"])
        if v:
            out.append(f"\n**Human verdict:** {VERDICT_LABEL.get(v.get('verdict'), v.get('verdict'))}"
                       + (f". {v['note']}" if v.get("note") else ""))
        else:
            out.append("\n**Human verdict:** pending (not yet reviewed)")
        out.append("")
    judged = [verdicts[r["id"]]["verdict"] for r in demo_rows if r["id"] in verdicts]
    if judged:
        out += ["## Verdict tally for the demo questions", "",
                f"correct {judged.count('correct')}, partly correct {judged.count('partly')}, "
                f"wrong {judged.count('wrong')} (of {len(judged)} judged)", ""]
    out += ["## Results on the full evaluation set", "", "```", summary_text, "```", "",
            "## Known limitations", "",
            "- Single run, 80 questions, written from the same documents: a baseline, not a final evaluation.",
            "- Retrieval is vector-only (hybrid search and re-ranking come in phase 5); the correct passage "
            "reached the model for about 80% of the answerable questions.",
            "- Near-miss questions (topic covered, detail absent) are sometimes answered anyway; a check that "
            "the answer is supported by the cited passage is planned for phase 7.",
            "- Latency on this CPU is about 30 s per question, dominated by prompt processing.", ""]
    return "\n".join(out)
