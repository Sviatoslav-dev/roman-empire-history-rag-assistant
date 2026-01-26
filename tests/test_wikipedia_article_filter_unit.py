from dataclasses import dataclass
from typing import cast

from data_ingestion.chunk_models import ArticleChunk, ChunkImageMention
from data_ingestion.wikipedia_image import WikipediaImage


@dataclass(frozen=True)
class _Quality:
    symbols_number: int
    citations_number: int
    has_problem_or_update_box: bool
    is_english_available: bool


class _FakePg:
    def __init__(self, *, qualities=None, images=None):
        self._qualities = qualities or {}
        self._images = images or {}

    def get_article_quality(self, title: str):
        return self._qualities.get(title)

    def get_image_by_url(self, url: str):
        return self._images.get(url)


class _FakeArticle:
    def __init__(
        self,
        title: str,
        *,
        visible_text: str = "",
        citations: int = 0,
        has_box: bool = False,
        english: bool = True,
        raise_on_quality_compute: Exception | None = None,
    ):
        self.title = title
        self._visible_text = visible_text
        self._citations = citations
        self._has_box = has_box
        self._english = english
        self._raise = raise_on_quality_compute

    def _get_visible_text(self):
        if self._raise:
            raise self._raise
        return self._visible_text

    def count_citations(self):
        if self._raise:
            raise self._raise
        return self._citations

    def has_problem_or_update_box(self):
        if self._raise:
            raise self._raise
        return self._has_box

    def is_english_article(self):
        if self._raise:
            raise self._raise
        return self._english


def test_is_license_allowed_allows_when_missing_isnt_called():
    from data_ingestion.wikipedia_article_filter import WikipediaArticleFilter

    f = WikipediaArticleFilter()
    assert f.is_license_allowed("") is True


def test_is_license_allowed_rejects_common_triggers_tokens_and_phrases():
    from data_ingestion.wikipedia_article_filter import WikipediaArticleFilter

    f = WikipediaArticleFilter()

    assert f.is_license_allowed("This is FAIR USE") is False
    assert f.is_license_allowed("licensed under Attribution something") is False
    assert f.is_license_allowed("All rights reserved") is False
    assert f.is_license_allowed("CC BY-SA 4.0") is True


def test_filter_articles_uses_db_quality_when_present(monkeypatch):
    import data_ingestion.wikipedia_article_filter as mod

    good = _FakeArticle("good")
    bad = _FakeArticle("bad")

    fake_pg = _FakePg(
        qualities={
            "good": _Quality(3000, 3, False, True),
            "bad": _Quality(1999, 999, False, True),
        }
    )
    monkeypatch.setattr(mod, "_postgres", fake_pg)

    out = mod.WikipediaArticleFilter().filter_articles(cast(list, [good, bad]))

    assert [a.title for a in out] == ["good"]


def test_filter_articles_fallback_computes_quality_when_db_missing(monkeypatch):
    import data_ingestion.wikipedia_article_filter as mod

    a1 = _FakeArticle("a1", visible_text="x" * 3001, citations=3, has_box=False, english=True)
    a2 = _FakeArticle("a2", visible_text="x" * 3001, citations=2, has_box=False, english=True)

    fake_pg = _FakePg(qualities={})
    monkeypatch.setattr(mod, "_postgres", fake_pg)

    out = mod.WikipediaArticleFilter().filter_articles(cast(list, [a1, a2]))

    assert [a.title for a in out] == ["a1"]


def test_filter_articles_exception_does_not_fail_entire_batch(monkeypatch):
    import data_ingestion.wikipedia_article_filter as mod

    ok = _FakeArticle("ok", visible_text="x" * 3001, citations=3)
    boom = _FakeArticle("boom", raise_on_quality_compute=RuntimeError("err"))

    fake_pg = _FakePg(qualities={})
    monkeypatch.setattr(mod, "_postgres", fake_pg)

    out = mod.WikipediaArticleFilter().filter_articles(cast(list, [boom, ok]))

    assert [a.title for a in out] == ["ok"]


def test_filter_chunk_images_keeps_when_no_metadata_or_allowed(monkeypatch):
    import data_ingestion.wikipedia_article_filter as mod

    img_allowed = WikipediaImage(url="https://upload.wikimedia.org/wikipedia/commons/a/a1/A.png")
    img_missing = WikipediaImage(url="https://upload.wikimedia.org/wikipedia/commons/b/b2/B.png")

    @dataclass
    class _ImgMeta:
        licence: str | None

    fake_pg = _FakePg(
        images={
            img_allowed.url: _ImgMeta(licence="CC BY-SA 4.0"),
            # img_missing absent => default allow
        }
    )
    monkeypatch.setattr(mod, "_postgres", fake_pg)

    chunks = [
        ArticleChunk(
            page_title="T",
            page_url="U",
            section_title="S",
            section_path="P",
            section_level=1,
            text_parts=["a"],
            images=[
                ChunkImageMention(image=img_allowed, caption="cap"),
                ChunkImageMention(image=img_missing, caption="cap2"),
            ],
            text="a",
        )
    ]

    out = mod.WikipediaArticleFilter().filter_chunk_images(chunks)

    assert len(out) == 1
    assert out[0] is not chunks[0]  # returns a new chunk
    assert [m.image.url for m in out[0].images] == [img_allowed.url, img_missing.url]


def test_filter_chunk_images_drops_forbidden_licence(monkeypatch):
    import data_ingestion.wikipedia_article_filter as mod

    img1 = WikipediaImage(url="https://upload.wikimedia.org/wikipedia/commons/a/a1/A.png")
    img2 = WikipediaImage(url="https://upload.wikimedia.org/wikipedia/commons/b/b2/B.png")

    @dataclass
    class _ImgMeta:
        licence: str | None

    fake_pg = _FakePg(
        images={
            img1.url: _ImgMeta(licence="fair use"),
            img2.url: _ImgMeta(licence="CC BY-SA"),
        }
    )
    monkeypatch.setattr(mod, "_postgres", fake_pg)

    chunk = ArticleChunk(
        page_title="T",
        page_url="U",
        section_title="S",
        section_path="P",
        section_level=1,
        text_parts=["a"],
        images=[
            ChunkImageMention(image=img1, caption="x"),
            ChunkImageMention(image=img2, caption="y"),
        ],
        text="a",
    )

    out = mod.WikipediaArticleFilter().filter_chunk_images([chunk])

    assert [m.image.url for m in out[0].images] == [img2.url]
