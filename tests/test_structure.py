"""Unit tests for sia.processing.structure and Chunk.location()."""
from sia.ingestion.models import DocumentMeta, PageText
from sia.processing.chunker import ChunkingConfig, chunk_document
from sia.processing.structure import annotate_chunks, classify_content, find_headings
from sia.processing.tokens import whitespace_counter as wc


def _texts(text, ftype="pdf"):
    return [h.text for h in find_headings(text, ftype)]


def test_numbered_and_appendix_headings():
    text = "1. Purpose\nThis procedure describes the steps.\n2.1 Roles and Duties\nThe lead owns it.\nAppendix B Contacts"
    assert _texts(text) == ["1. Purpose", "2.1 Roles and Duties", "Appendix B Contacts"]


def test_all_caps_heading_but_not_tlp_banner():
    assert _texts("body text here.\nVERSION HISTORY\nFebruary 29, 2024: Initial version.") == ["VERSION HISTORY"]
    assert _texts("TLP:CLEAR\nsome body text that continues for a while.") == []


def test_title_case_heading_needs_blank_line_before():
    text = "intro sentence ends here.\n\nExecutive Summary\nIncident response is a critical part of risk management."
    assert _texts(text) == ["Executive Summary"]
    wrapped = "this is a long sentence that wraps over\nTwo Capitalized Words\nand continues here."
    assert _texts(wrapped) == []


def test_body_sentences_lists_captions_and_urls_are_not_headings():
    text = ("\nThe team should contain the incident.\n- Isolate the host\nFigure 1: Example Network Diagram\n"
            "See https://doi.org/10.6028/NIST.SP.800-61r3\n\nOrganizations should act, and fast.")
    assert _texts(text) == []


def test_docx_line_after_blank_is_a_heading():
    text = "Severity Levels\nCritical incidents require escalation within 15 minutes.\n\nescalation matrix overview\nRows follow."
    assert _texts(text, "docx") == ["Severity Levels", "escalation matrix overview"]
    assert _texts("a | b | c\n\nrow | x | y", "docx") == []


def test_content_type_detection():
    assert classify_content("Role | Owner | Phone\nLead | Ana | 123\nDeputy | Bo | 456") == "table"
    code = "function run() {\n  var x = foo(bar);\n  return x;\n}\nif (a == b) { c(); }"
    assert classify_content(code) == "code"
    assert classify_content("Organizations should contain the incident and recover systems.") == "prose"
    assert classify_content("") == "prose"


def _meta(ftype="pdf"):
    return DocumentMeta(doc_id="d0c0d0c0d0c0", filename=f"x.{ftype}", file_type=ftype, title="Doc")


def test_section_is_carried_across_pages_and_updated_mid_page():
    p1 = "1. Intro\n" + " ".join(f"Alpha sentence {i} is here." for i in range(10))
    p2 = " ".join(f"Beta sentence {i} is here." for i in range(10)) + "\n2. Scope\n" + \
         " ".join(f"Gamma sentence {i} is here." for i in range(30))
    pages = [PageText(1, p1), PageText(2, p2)]
    chunks = chunk_document(_meta(), pages, ChunkingConfig(20, 0, 0), wc)
    annotate_chunks(chunks, pages, "pdf")
    by_page = {1: [c for c in chunks if c.page == 1], 2: [c for c in chunks if c.page == 2]}
    assert all(c.section == "1. Intro" for c in by_page[1])
    assert by_page[2][0].section == "1. Intro"            # carried from page 1
    assert by_page[2][-1].section == "2. Scope"           # new heading inside page 2


def test_section_is_none_without_headings_and_location_labels():
    pages = [PageText(None, "plain line one here\nplain line two here\nplain line three here")]
    chunks = chunk_document(_meta("txt"), pages, ChunkingConfig(5, 0, 0), wc)
    for c in chunks:
        c.doc_chunk_count = len(chunks)
    annotate_chunks(chunks, pages, "txt")
    assert chunks[0].location() == f"part 1 of {len(chunks)}"
    pdf = chunk_document(_meta(), [PageText(7, "Body text.")], ChunkingConfig(5, 0, 0), wc)[0]
    pdf.section = "Containment"
    assert pdf.location() == "p. 7, section: Containment"
    docx = chunk_document(_meta("docx"), [PageText(None, "Body text.")], ChunkingConfig(5, 0, 0), wc)[0]
    docx.section, docx.doc_chunk_count = "Roles", 3
    assert docx.location() == "section: Roles (part 1 of 3)"
