"""Helpers shared by the phase 4 tests (not collected by pytest)."""
from sia.processing.models import Chunk
from sia.retrieval.vector_retriever import RetrievedChunk


def make_chunk(i: int, text: str | None = None, n_tokens: int = 40, page: int | None = None,
               section: str | None = None, title: str = "Guide") -> Chunk:
    text = text or f"Passage {i} says that hosts should be isolated from the network."
    return Chunk(chunk_id=f"doc-p{i:03d}-00", doc_id="doc", filename="guide.pdf",
                 title=f"{title} {i}", doc_kind="public", publisher=None, publication_date=None,
                 page=page if page is not None else i, chunk_index=i, text=text,
                 n_tokens=n_tokens, char_start=0, char_end=len(text), section=section,
                 doc_chunk_count=10)


def make_results(n: int = 5, top_score: float = 0.8, **kw) -> list[RetrievedChunk]:
    return [RetrievedChunk(make_chunk(i + 1, **kw), round(top_score - 0.02 * i, 3), i + 1)
            for i in range(n)]


class FakeRetriever:
    def __init__(self, results):
        self.results, self.queries = results, []

    def retrieve(self, query, k=None):
        self.queries.append(query)
        return list(self.results)
