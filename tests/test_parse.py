from sia.answering.parse import has_answer_text, parse_citations, split_structured


def test_has_answer_text():
    for empty in ("", "   ", "[1]", "[4]\n[1]", "[N1]", "[11]", "[1] [2] [3, 4]", "[ ]"):
        assert not has_answer_text(empty), empty
    for real in ("No.", "Within 24 hours.", "[1] 24 hours", "Isolate hosts [1]."):
        assert has_answer_text(real), real


def test_split_structured():
    assert split_structured("ANSWER: Isolate hosts.\nSOURCES: 1, 3") == ("Isolate hosts.", [1, 3])
    assert split_structured("answer: Two lines\nof text.\nsources: [2], [4]") == ("Two lines\nof text.", [2, 4])
    assert split_structured("ANSWER: Only an answer.") == ("Only an answer.", None)
    assert split_structured("Plain reply [1].") == ("Plain reply [1].", None)
    assert split_structured("ANSWER: Same line. SOURCES: 5") == ("Same line.", [5])
    assert split_structured("ANSWER:\nSOURCES: 1") == ("", [1])


def test_parse_citations():
    assert parse_citations("[1][2], [3, 4] [10] [1]", 4) == ([1, 2, 3, 4], [10])
    assert parse_citations("no citations", 4) == ([], [])
