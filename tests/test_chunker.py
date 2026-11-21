"""Unit tests for sia.processing.chunker (plain asserts, pytest-compatible).

A whitespace token counter is used so the tests are fast and offline.
"""
from sia.ingestion.models import DocumentMeta, PageText
from sia.processing.chunker import ChunkingConfig, chunk_document, chunk_text
from sia.processing.tokens import whitespace_counter as wc

CFG = ChunkingConfig(chunk_tokens=60, overlap_tokens=10, min_tokens=10)
HARD_MAX = CFG.chunk_tokens + CFG.min_tokens


def _sentences(n: int) -> str:
    # each sentence = 8 words
    return " ".join(f"Sentence number {i} has exactly seven words here." for i in range(n))


def _meta(file_type: str = "pdf") -> DocumentMeta:
    return DocumentMeta(doc_id="abc123def456", filename=f"x.{file_type}",
                        file_type=file_type, title="Test Doc")


def test_short_text_is_one_chunk():
    text = _sentences(3)
    spans = chunk_text(text, CFG, wc)
    assert len(spans) == 1
    assert text[spans[0].start:spans[0].end] == text


def test_chunks_respect_size_limit_and_are_verbatim_slices():
    text = _sentences(100)
    spans = chunk_text(text, CFG, wc)
    assert len(spans) > 5
    for a, b in spans:
        assert 0 < wc(text[a:b]) <= HARD_MAX


def test_chunks_overlap_and_cover_everything_in_order():
    text = _sentences(100)
    spans = chunk_text(text, CFG, wc)
    starts = [a for a, _ in spans]
    assert starts == sorted(set(starts))                       # strictly increasing
    for (a1, b1), (a2, b2) in zip(spans, spans[1:]):
        assert a2 < b1                                          # overlap exists
        assert b2 > b1                                          # always progresses
    assert spans[0].start == 0 and spans[-1].end == len(text)
    covered = set()
    for a, b in spans:
        covered.update(range(a, b))
    assert covered >= {i for i, ch in enumerate(text) if not ch.isspace()}


def test_overlap_is_sentence_aligned():
    text = _sentences(100)
    for a, b in chunk_text(text, CFG, wc):
        piece = text[a:b]
        assert piece.startswith("Sentence number")
        assert piece.endswith("here.")


def test_zero_overlap_gives_disjoint_chunks():
    cfg = ChunkingConfig(chunk_tokens=60, overlap_tokens=0, min_tokens=10)
    spans = chunk_text(_sentences(100), cfg, wc)
    for (a1, b1), (a2, b2) in zip(spans, spans[1:]):
        assert a2 >= b1


def test_tiny_tail_is_absorbed():
    # 33 sentences * 8 = 264 words, chunk 256 -> tail of 8 <= min_tokens -> one chunk
    cfg = ChunkingConfig(chunk_tokens=256, overlap_tokens=40, min_tokens=40)
    spans = chunk_text(_sentences(33), cfg, wc)
    assert len(spans) == 1


def test_oversized_line_without_punctuation_is_split():
    text = " ".join(f"w{i}" for i in range(1000))                # one 1000-word 'sentence'
    spans = chunk_text(text, CFG, wc)
    assert len(spans) > 10
    for a, b in spans:
        assert wc(text[a:b]) <= HARD_MAX


def test_single_giant_token_is_split_by_characters():
    def count(t: str) -> int:                                    # 1 token per 4 chars
        return -(-len(t) // 4)
    text = "x" * 3000                                            # 750 'tokens' in one word
    cfg = ChunkingConfig(chunk_tokens=100, overlap_tokens=0, min_tokens=0)
    spans = chunk_text(text, cfg, count)
    assert len(spans) >= 8
    for a, b in spans:
        assert count(text[a:b]) <= 100
    assert "".join(text[a:b] for a, b in spans) == text


def test_bullets_and_paragraphs_are_cut_points():
    text = "Intro sentence about the plan\n- first action item\n- second action item\n\nNew paragraph here"
    cfg = ChunkingConfig(chunk_tokens=5, overlap_tokens=0, min_tokens=0)
    pieces = [text[a:b] for a, b in chunk_text(text, cfg, wc)]
    assert pieces[0].startswith("Intro")
    assert any(p.startswith("- second") or p.startswith("- first") for p in pieces)
    assert pieces[-1] == "New paragraph here"


def test_newline_is_boundary_only_for_non_pdf():
    # sentences wrapped over two lines, as pypdf delivers them
    text = "\n".join(f"Sentence number {i} has\nexactly seven words here." for i in range(60))
    cfg = ChunkingConfig(chunk_tokens=56, overlap_tokens=6, min_tokens=10)
    # PDF mode: wraps are ignored, every chunk starts at a real sentence start
    pdf = [text[a:b] for a, b in chunk_text(text, cfg, wc, newline_is_boundary=False)]
    assert all(piece.startswith("Sentence number") for piece in pdf)
    # DOCX/TXT mode: each line is a unit, so overlap may start on a line
    txt = [text[a:b] for a, b in chunk_text(text, cfg, wc, newline_is_boundary=True)]
    assert any(piece.startswith("exactly") for piece in txt)


def test_oversized_pdf_text_without_punctuation_falls_back_to_lines():
    lines = [f"line {i} without any punctuation but with some words" for i in range(60)]
    text = "\n".join(lines)
    for a, b in chunk_text(text, CFG, wc, newline_is_boundary=False):
        assert all(ln in lines for ln in text[a:b].split("\n"))


def test_document_chunks_never_cross_pages_and_keep_metadata():
    pages = [PageText(page=1, text=_sentences(40)),
             PageText(page=2, text=""),
             PageText(page=3, text=_sentences(30))]
    chunks = chunk_document(_meta(), pages, CFG, wc)
    assert {c.page for c in chunks} == {1, 3}
    by_page = {1: pages[0].text, 3: pages[2].text}
    for c in chunks:
        assert c.text == by_page[c.page][c.char_start:c.char_end]
        assert c.n_tokens == wc(c.text) <= HARD_MAX
        assert (c.doc_id, c.title, c.filename) == ("abc123def456", "Test Doc", "x.pdf")
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_ids_are_unique_and_deterministic():
    pages = [PageText(page=i, text=_sentences(40)) for i in (1, 2, 3)]
    a = chunk_document(_meta(), pages, CFG, wc)
    b = chunk_document(_meta(), pages, CFG, wc)
    ids = [c.chunk_id for c in a]
    assert len(ids) == len(set(ids))
    assert ids == [c.chunk_id for c in b]
    assert ids[0] == "abc123def456-p001-00"


def test_docx_page_none_uses_pna_id():
    chunks = chunk_document(_meta("docx"), [PageText(page=None, text=_sentences(5))], CFG, wc)
    assert chunks[0].chunk_id == "abc123def456-pNA-00"
    assert chunks[0].page is None


def test_embed_text_has_context_header_but_text_is_unchanged():
    c = chunk_document(_meta(), [PageText(page=1, text="Body text.")], CFG, wc)[0]
    assert c.embed_text() == "Test Doc\nBody text."
    c.section = "Containment"
    assert c.embed_text() == "Test Doc - Containment\nBody text."
    assert c.text == "Body text."


def test_invalid_config_is_rejected():
    for kwargs in ({"chunk_tokens": 0}, {"chunk_tokens": 10, "overlap_tokens": 10},
                   {"min_tokens": -1}):
        try:
            ChunkingConfig(**kwargs)
        except ValueError:
            continue
        raise AssertionError(f"config accepted: {kwargs}")
