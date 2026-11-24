"""BM25 tokenizer (phase 5a): identifiers, stop-words, no stemming."""
from sia.retrieval.tokenizer import STOPWORDS, is_identifier, tokenize


def test_lowercase_and_punctuation():
    assert tokenize("Contain the Ransomware, then eradicate!") == ["contain", "ransomware", "eradicate"]


def test_identifier_is_kept_whole_and_in_pieces():
    toks = tokenize("NIST SP 800-61r3")
    assert "800-61r3" in toks and "800" in toks and "61r3" in toks
    assert "nist" in toks and "sp" in toks


def test_cve_and_internal_ids():
    assert tokenize("CVE-2021-45046") == ["cve-2021-45046", "cve", "2021", "45046"]
    toks = tokenize("AIT-IR-BR-009")
    assert toks[0] == "ait-ir-br-009" and {"ait", "ir", "br", "009"} <= set(toks)


def test_version_and_dotted_filename_stem():
    assert "3.1" in tokenize("see section 3.1")
    toks = tokenize("NIST.SP.800-61r3")
    assert "nist.sp.800-61r3" in toks and {"nist", "sp", "800", "61r3"} <= set(toks)


def test_underscore_and_slash_are_separators():
    assert tokenize("syn_ir_playbook") == ["syn", "ir", "playbook"]
    assert tokenize("backup/restore") == ["backup", "restore"]


def test_stopwords_and_single_letters_removed_but_digits_kept():
    assert tokenize("What is the policy of the company?") == ["policy", "company"]
    assert tokenize("it's a 5 step plan") == ["5", "step", "plan"]
    assert "the" in STOPWORDS and "not" not in STOPWORDS and "must" not in STOPWORDS


def test_no_stemming():
    assert tokenize("incident") == ["incident"] and tokenize("incidents") == ["incidents"]


def test_duplicates_are_preserved_by_the_tokenizer():
    assert tokenize("worm worm worm") == ["worm", "worm", "worm"]      # de-duplication is the retriever's job


def test_empty_none_and_unicode():
    assert tokenize("") == [] and tokenize(None) == [] and tokenize("   ") == []
    assert tokenize("Café ﬁrewall") == ["café", "firewall"]          # NFKC folds the ligature


def test_is_identifier():
    assert is_identifier("800-61r3") and is_identifier("sp800") and is_identifier("cve-2021-45046")
    assert not is_identifier("2021") and not is_identifier("ransomware") and not is_identifier("3.1")
