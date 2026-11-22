from sia.answering.excerpts import best_excerpt

FILLER = " ".join(f"Filler sentence number {i} talks about budgets and schedules." for i in range(20))
KEY = "Defenders should isolate infected hosts from the network immediately."


def _collapse(s):
    return " ".join(s.split())


def test_short_text_is_returned_whole():
    assert best_excerpt("One sentence.\n  Another   one.", "x", 350) == "One sentence. Another one."
    assert best_excerpt("", "x") == ""


def test_best_matching_sentence_is_selected_and_text_is_verbatim():
    text = f"{FILLER} {KEY} {FILLER}"
    ex = best_excerpt(text, "How should infected hosts be isolated from the network?", 300)
    assert "isolate infected hosts" in ex and len(ex) <= 300 + 8
    body = ex.removeprefix("... ").removesuffix(" ...")
    assert body in _collapse(text)                   # nothing rewritten, only a cut-out
    assert ex.startswith("... ") and ex.endswith(" ...")


def test_no_overlap_falls_back_to_the_start():
    ex = best_excerpt(FILLER, "zebra giraffe", 200)
    assert ex.startswith("Filler sentence number 0") and not ex.startswith("...")


def test_one_huge_sentence_is_truncated_at_a_word_boundary():
    ex = best_excerpt("word " * 500 + "end.", "word", 100)
    assert len(ex) <= 100 + 4 and ex.endswith(" ...")
