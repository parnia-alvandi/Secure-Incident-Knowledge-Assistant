import json
import tempfile
from pathlib import Path

from _factories import make_chunk
from sia.evaluation.baseline import (doc_kind_of, format_summary, gold_chunk_ids, load_done,
                                     numbers_in, review_markdown, select_questions, summarize,
                                     unsupported_numbers)


def _qs():
    qs = []
    for i in range(30):
        qs.append({"id": f"P-{i:02d}", "lang": "en", "answerable": True, "doc": "guide.pdf",
                   "evidence_quote": "q", "evidence_page": 1, "question": "?", "answer": "a"})
        qs.append({"id": f"S-{i:02d}", "lang": "en", "answerable": True, "doc": "syn_01_x.docx",
                   "evidence_quote": "q", "evidence_page": None, "question": "?", "answer": "a"})
    for i in range(6):
        qs.append({"id": f"N-{i}", "lang": "en", "answerable": False, "category": "near_miss",
                   "question": "?", "doc": None})
    qs.append({"id": "FA-1", "lang": "fa", "answerable": True, "doc": "guide.pdf", "evidence_quote": "q"})
    return qs


def test_selection_is_seeded_stratified_and_english_only():
    a = select_questions(_qs(), 10, seed=7)
    b = select_questions(_qs(), 10, seed=7)
    assert [q["id"] for q in a] == [q["id"] for q in b]
    assert [q["id"] for q in a] != [q["id"] for q in select_questions(_qs(), 10, seed=8)]
    ans = [q for q in a if q["answerable"]]
    assert len(ans) == 10 and sum(doc_kind_of(q["doc"]) == "public" for q in ans) == 5
    assert sum(not q["answerable"] for q in a) == 6 and all(q["lang"] == "en" for q in a)


def test_gold_chunks_need_right_document_page_and_quote():
    c1 = make_chunk(1, text="Isolate infected hosts from the network.", page=3)
    c2 = make_chunk(2, text="Isolate infected hosts from the network.", page=4)
    chunks = {c1.chunk_id: c1, c2.chunk_id: c2}
    q = {"answerable": True, "doc": "guide.pdf", "evidence_page": 3,
         "evidence_quote": "isolate  infected hosts"}
    assert gold_chunk_ids(q, chunks) == {c1.chunk_id}
    assert gold_chunk_ids({**q, "doc": "other.pdf"}, chunks) == set()
    assert gold_chunk_ids({"answerable": False}, chunks) == set()


def test_unsupported_numbers_signal():
    assert numbers_in("within 72 hours, 1,000 hosts, 3.5% and 2021") == {"72", "1000", "3.5", "2021"}
    passages = ["Notify within 72 hours.", "About 1,000 hosts in 2021."]
    assert unsupported_numbers("Notify within 72 hours [1] and 48 hours [2], 1000 hosts.", passages) == ["48"]
    assert unsupported_numbers("Step 1. Isolate [1].", passages) == []      # single digits ignored


def _row(i, answerable, status, **kw):
    return {"id": f"X{i}", "answerable": answerable, "status": status, "question": "q", "answer": "a",
            "timings": {"total_s": 30.0 + i, "prompt_tokens_per_s": 60.0}, **kw}


def test_summary_counts_and_text():
    rows = [_row(1, False, "insufficient_evidence", category="near_miss"),
            _row(2, False, "answered", category="near_miss", unsupported_numbers=["48"]),
            _row(3, False, "insufficient_evidence", category="out_of_scope"),
            _row(4, True, "answered", gold_in_prompt=True, cited_gold=True),
            _row(5, True, "answered", gold_in_prompt=True, cited_gold=False, unsupported_numbers=["9"]),
            _row(6, True, "insufficient_evidence", gold_in_prompt=False, cited_gold=None),
            _row(7, True, "unverified", gold_in_prompt=True, cited_gold=None)]
    s = summarize(rows)
    assert s["unanswerable"]["status"] == {"insufficient_evidence": 2, "answered": 1}
    assert s["unanswerable"]["by_category"]["near_miss"]["n"] == 2
    a = s["answerable"]
    assert (a["answered"], a["cited_gold"], a["gold_in_prompt"], a["answered_with_unsupported_numbers"]) == (2, 1, 3, 1)
    assert s["latency"]["llm_calls"] == 7 and s["latency"]["p50_s"] == 34.0
    text = format_summary(s)
    assert "correct refusals 2/3" in text and "wrongly refused 1/4" in text
    md = review_markdown(rows, text)
    assert "## X1" in md and "Your verdict" in md


def test_resume_reads_previous_results():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "r.jsonl"
        assert load_done(p) == {}
        p.write_text(json.dumps({"id": "A"}) + "\n\n" + json.dumps({"id": "B"}) + "\n", encoding="utf-8")
        assert set(load_done(p)) == {"A", "B"}


def test_p90_uses_nearest_rank_even_for_tiny_samples():
    def lat(vals):
        return summarize([_row(i, True, "answered", timings={"total_s": v}) for i, v in enumerate(vals)])["latency"]
    assert lat([20.0, 31.0])["p90_s"] == 31.0                 # regression: used to return 20.0
    assert lat(list(map(float, range(1, 11))))["p90_s"] == 9.0
    assert lat([float(i) for i in range(1, 81)])["p90_s"] == 72.0
    assert lat([5.0])["p90_s"] == 5.0


def _demo_rows(n_ans=8, n_una=3):
    rows, selected = {}, []
    order = ["A", "N", "A", "A", "N", "A", "A", "A", "N", "A", "A"]
    a = n = 0
    for kind in order:
        if kind == "A":
            a += 1
            qid, ans = f"A{a}", True
        else:
            n += 1
            qid, ans = f"N{n}", False
        selected.append({"id": qid})
        rows[qid] = {"id": qid, "answerable": ans, "status": "answered" if ans else "insufficient_evidence",
                     "question": f"question {qid}", "gold_answer": "gold", "gold_doc": "d.pdf", "gold_page": 2,
                     "answer": f"answer {qid} [1]", "reason": None, "warnings": ["w1"] if qid == "A1" else [],
                     "sources": [{"n": 1, "title": "T", "filename": "d.pdf", "location": "p. 2", "excerpt": "exc"}],
                     "timings": {"total_s": 30.0, "retrieval_s": 0.1, "llm_total_s": 29.0, "best_score": 0.8}}
    return selected, rows


def test_demo_selection_rule_is_positional_not_outcome_based():
    from sia.evaluation.baseline import demo_selection
    selected, rows = _demo_rows()
    picked = demo_selection(selected, rows)
    assert [r["id"] for r in picked] == ["A1", "A2", "A3", "A4", "A5", "N1"]
    rows.pop("A2")                                   # a missing result is skipped, not replaced by a "good" one
    assert [r["id"] for r in demo_selection(selected, rows)][:2] == ["A1", "A3"]


def test_demo_markdown_never_invents_verdicts():
    from sia.evaluation.baseline import demo_markdown, demo_selection
    selected, rows = _demo_rows()
    picked = demo_selection(selected, rows)
    md = demo_markdown(picked, "SUMMARY TEXT", {"LLM": "qwen"}, {"A1": {"verdict": "wrong", "note": "mixed phases"}}, 40, 42)
    assert md.count("**Human verdict:** pending") == 5          # 6 questions, only A1 judged
    assert "**Human verdict:** wrong. mixed phases" in md
    assert "wrong 1 (of 1 judged)" in md and "SUMMARY TEXT" in md and "seed 42" in md
    assert "> Warning: w1" in md and "  > exc" in md and "- LLM: qwen" in md
    md2 = demo_markdown(picked, "S", {}, {}, 40, 42)
    assert "Verdict tally" not in md2 and md2.count("pending") >= 6


def test_summary_separates_real_answers_from_citation_only_replies():
    rows = [
        _row(1, True, "answered", answer="Notify within 24 hours. [1]", gold_in_prompt=True, cited_gold=True),
        _row(2, True, "answered", answer="[1]", gold_in_prompt=True, cited_gold=True),
        _row(3, True, "answered", answer="[4]\n[1]", gold_in_prompt=True, cited_gold=False),
        _row(4, True, "unverified", answer="[11]", gold_in_prompt=True),
        _row(5, True, "no_answer", answer="", gold_in_prompt=True),
        _row(6, False, "answered", answer="[2]", category="near_miss"),
        _row(7, False, "answered", answer="It takes 48 hours. [2]", category="near_miss"),
        _row(8, False, "insufficient_evidence", answer="Not stated...", category="out_of_scope"),
    ]
    s = summarize(rows)
    a, u = s["answerable"], s["unanswerable"]
    assert a["answered"] == 3 and a["answered_with_text"] == 1 and a["cited_gold_with_text"] == 1
    assert a["without_text"] == 4 and a["status"]["no_answer"] == 1
    assert u["without_text"] == 1 and u["answered_with_text"] == 1
    text = format_summary(s)
    assert "answered WITH real answer text: 1/5" in text and "replies without answer text" in text
    assert "no_answer 1/5" in text and "WITH real answer text: 1;" in text


def test_retries_are_counted_and_old_rows_without_the_field_still_work():
    rows = [_row(1, True, "answered", answer="Real text here.", timings={"total_s": 60.0, "retries": 1}),
            _row(2, True, "answered", answer="Other real text.", timings={"total_s": 30.0})]
    assert summarize(rows)["retries"] == 1
    assert "repair retries used on 1 question(s)" in format_summary(summarize(rows))
