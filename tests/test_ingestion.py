"""Tests for ingestion: normalization, loaders, metadata, error handling."""
import yaml
from docx import Document
from pypdf import PdfWriter

from sia.ingestion.normalize import normalize_text
from sia.ingestion.pipeline import ingest_directory


def _ingest(tmp_path, **kw):
    return ingest_directory(tmp_path, tmp_path / "manifest.yaml", **kw)


def test_normalize_dehyphenates_and_keeps_identifiers():
    out = normalize_text("The contain-\nment of CVE-2021-44228  is   key.")
    assert "containment" in out
    assert "CVE-2021-44228" in out
    assert "  " not in out


def test_normalize_ligature_and_control_chars():
    assert normalize_text("\ufb01le\x00 ok") == "file ok"


def test_txt_ok(tmp_path):
    (tmp_path / "a.txt").write_text("Incident response text. " * 10, encoding="utf-8")
    doc = _ingest(tmp_path).documents[0]
    assert doc.status == "ok" and doc.pages[0].page is None


def test_txt_cp1252_fallback(tmp_path):
    (tmp_path / "a.txt").write_bytes("caf\u00e9 incident report " .encode("cp1252") * 5)
    doc = _ingest(tmp_path).documents[0]
    assert doc.status == "ok" and "caf\u00e9" in doc.pages[0].text


def test_empty_txt_is_flagged(tmp_path):
    (tmp_path / "e.txt").write_text("   \n", encoding="utf-8")
    assert _ingest(tmp_path).documents[0].status == "empty"


def test_corrupt_pdf_does_not_crash(tmp_path):
    (tmp_path / "bad.pdf").write_bytes(b"this is not a pdf")
    (tmp_path / "good.txt").write_text("fine " * 30, encoding="utf-8")
    rep = _ingest(tmp_path)
    status = {d.meta.filename: d.status for d in rep.documents}
    assert status == {"bad.pdf": "error", "good.txt": "ok"}


def test_blank_pdf_is_empty(tmp_path):
    w = PdfWriter()
    w.add_blank_page(width=200, height=200)
    with (tmp_path / "blank.pdf").open("wb") as f:
        w.write(f)
    doc = _ingest(tmp_path).documents[0]
    assert doc.status == "empty" and doc.pages[0].page == 1


def test_docx_paragraphs_and_tables(tmp_path):
    d = Document()
    d.add_heading("Phishing Procedure", 1)
    d.add_paragraph("Report suspicious emails within four hours.")
    t = d.add_table(rows=1, cols=2)
    t.rows[0].cells[0].text = "Severity"
    t.rows[0].cells[1].text = "High"
    d.save(tmp_path / "p.docx")
    doc = _ingest(tmp_path).documents[0]
    assert doc.status == "ok"
    assert "four hours" in doc.pages[0].text
    assert "Severity | High" in doc.pages[0].text


def test_metadata_complete_and_incomplete(tmp_path):
    (tmp_path / "a.txt").write_text("alpha text " * 20, encoding="utf-8")
    (tmp_path / "b.txt").write_text("beta text " * 20, encoding="utf-8")
    manifest = {"documents": [{
        "filename": "a.txt", "title": "Doc A", "publisher": "Test",
        "doc_kind": "synthetic", "license": "CC0", "publication_date": "2026-01-01",
        "acquired_date": "2026-11-16", "language": "en",
        "topics": ["phishing"], "description": "A test document.",
    }]}
    (tmp_path / "manifest.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    docs = {d.meta.filename: d.meta for d in _ingest(tmp_path).documents}
    assert docs["a.txt"].missing_fields == []
    assert docs["a.txt"].title == "Doc A"
    assert "manifest_entry" in docs["b.txt"].missing_fields
    assert docs["b.txt"].title == "b"          # fallback from file name


def test_public_doc_requires_source_url(tmp_path):
    (tmp_path / "a.txt").write_text("alpha " * 30, encoding="utf-8")
    manifest = {"documents": [{
        "filename": "a.txt", "title": "A", "publisher": "X", "doc_kind": "public",
        "license": "PD", "publication_date": "2020-01-01", "acquired_date": "2026-11-16",
        "language": "en", "topics": ["x"], "description": "d"}]}
    (tmp_path / "manifest.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    assert "source_url" in _ingest(tmp_path).documents[0].meta.missing_fields