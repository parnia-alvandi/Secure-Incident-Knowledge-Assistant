"""Unit tests for sia.processing.cleaner (plain asserts, pytest-compatible)."""
from sia.ingestion.models import PageText
from sia.processing.cleaner import CleanConfig, clean_pages


NAMES = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"]


def _body(i: int) -> str:
    """12 body lines whose words differ per page (not only in digits)."""
    return "\n".join(f"{NAMES[i % 8]} line {NAMES[j % 8]} text {NAMES[(i + j) % 8]}x"
                     for j in range(12))


def _pdf_like(n_pages: int) -> list[PageText]:
    pages = []
    for i in range(1, n_pages + 1):
        text = f"NIST SP 800-61r3\nApril 2025\n{_body(i)}\nPage {i} of {n_pages}"
        pages.append(PageText(page=i, text=text))
    return pages


def test_repeated_header_and_page_number_are_removed():
    cleaned, report = clean_pages(_pdf_like(6))
    for i, p in enumerate(cleaned, start=1):
        assert "NIST SP 800-61r3" not in p.text
        assert "April 2025" not in p.text
        assert f"Page {i} of" not in p.text
        assert _body(i) in p.text
    assert report.lines_removed == 6 * 3
    assert any("NIST SP" in k for k in report.patterns)


def test_short_documents_are_untouched():
    pages = _pdf_like(3)
    cleaned, report = clean_pages(pages)
    assert [p.text for p in cleaned] == [p.text for p in pages]
    assert report.lines_removed == 0


def test_unique_first_line_is_kept():
    pages = [PageText(page=i, text=f"Heading {NAMES[i]} differs\n{_body(i)}")
             for i in range(1, 7)]
    cleaned, report = clean_pages(pages)
    assert [p.text for p in cleaned] == [p.text for p in pages]
    assert report.lines_removed == 0


def test_input_pages_are_not_modified():
    pages = _pdf_like(5)
    before = [p.text for p in pages]
    clean_pages(pages)
    assert [p.text for p in pages] == before


def test_long_repeated_lines_are_not_treated_as_headers():
    long_line = "This sentence repeats on every page but it is long. " * 4   # > 100 chars
    pages = [PageText(page=i, text=f"{long_line}\n{_body(i)}") for i in range(1, 7)]
    cleaned, report = clean_pages(pages, CleanConfig(max_line_chars=100))
    assert all(long_line.strip() in p.text for p in cleaned)
    assert report.lines_removed == 0


def test_empty_pages_are_preserved():
    pages = _pdf_like(5)
    pages.insert(2, PageText(page=99, text="", suspicious=True))
    cleaned, _ = clean_pages(pages)
    assert len(cleaned) == len(pages)
    assert cleaned[2].text == "" and cleaned[2].suspicious


def test_body_lines_that_differ_only_in_inner_digits_are_kept():
    # regression: masking every digit used to delete body text and section headings
    pages = []
    for i in range(1, 8):
        body = "\n".join(f"Paragraph about topic mentions CVE-2021-4422{i} in detail {j}x"
                         for j in range(3))
        pages.append(PageText(page=i, text=f"{i}. Section number {i} heading\n{body}\nclosing remark {NAMES[i]}"))
    cleaned, report = clean_pages(pages)
    assert [p.text for p in cleaned] == [p.text for p in pages]
    assert report.lines_removed == 0


def test_footer_with_trailing_page_number_and_bare_numbers_are_removed():
    pages = []
    for i in range(1, 7):
        text = f"{i}\n{_body(i)}\nIncident Response Guide {i + 10}"
        pages.append(PageText(page=i, text=text))
    cleaned, report = clean_pages(pages)
    for i, p in enumerate(cleaned, start=1):
        assert p.text == _body(i)
    assert report.lines_removed == 12


def test_page_counter_inside_a_sentence_is_not_masked():
    pages = [PageText(page=i, text=f"{_body(i)}\nAs shown on page {i} of the plan we proceed.")
             for i in range(1, 8)]
    cleaned, report = clean_pages(pages)
    assert [p.text for p in cleaned] == [p.text for p in pages]
    assert report.lines_removed == 0


def test_cisa_style_footers_with_page_counter_are_removed():
    # real footers seen in the CISA documents: counter at the start of a longer line
    pages = [PageText(page=i, text=f"{_body(i)}\nPage {i} of 8 | Product ID: AA24-060A")
             for i in range(1, 9)]
    cleaned, report = clean_pages(pages)
    assert all("Product ID" not in p.text for p in cleaned)
    assert all(_body(i) in p.text for i, p in enumerate(cleaned, start=1))
    pages2 = [PageText(page=i, text=f"{_body(i)}\nPage | {i} TLP:CLEAR") for i in range(1, 9)]
    cleaned2, _ = clean_pages(pages2)
    assert all("TLP:CLEAR" not in p.text for p in cleaned2)


def test_blank_pages_are_emptied_and_reported():
    pages = _pdf_like(5)
    pages[1] = PageText(page=2, text="This page has been left blank intentionally.\n\nES-4")
    cleaned, report = clean_pages(pages)
    assert cleaned[1].text == ""
    assert report.blank_pages == [2]
    assert cleaned[0].text != ""


def test_blank_marker_inside_a_long_page_is_kept():
    long_text = _body(1) + "\nNote: this page has been left blank intentionally in the printed copy."
    cleaned, report = clean_pages([PageText(page=1, text=long_text)] )
    assert cleaned[0].text == long_text
    assert report.blank_pages == []


def test_blank_pages_are_emptied_even_in_short_documents():
    pages = [PageText(page=1, text="real content here " * 5),
             PageText(page=2, text="This page left blank intentionally")]
    cleaned, report = clean_pages(pages)
    assert cleaned[1].text == "" and report.blank_pages == [2]
