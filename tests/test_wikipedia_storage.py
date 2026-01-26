from __future__ import annotations

# Install import-time stubs *before* importing production modules.
from tests._import_stubs import install as _install_import_stubs

_install_import_stubs()

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import pytest


@dataclass(frozen=True)
class _Row:
    title: str
    url: str | None
    local_path: str


def test_get_downloaded_articles_returns_empty_when_postgres_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    import data_ingestion.wikipedia_storage as mod

    fake_pg = MagicMock(name="pg")
    fake_pg.enabled = False
    monkeypatch.setattr(mod, "get_pg_metadata_store", lambda: fake_pg)

    out = mod.WikipediaStorage().get_downloaded_articles()

    assert out == []


def test_get_downloaded_articles_returns_empty_when_no_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    import data_ingestion.wikipedia_storage as mod

    fake_pg = MagicMock(name="pg")
    fake_pg.enabled = True
    fake_pg.list_articles_with_local_path.return_value = []
    monkeypatch.setattr(mod, "get_pg_metadata_store", lambda: fake_pg)

    out = mod.WikipediaStorage().get_downloaded_articles()

    assert out == []


def test_get_downloaded_articles_skips_missing_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import data_ingestion.wikipedia_storage as mod

    missing = tmp_path / "missing.html"

    fake_pg = MagicMock(name="pg")
    fake_pg.enabled = True
    fake_pg.list_articles_with_local_path.return_value = [_Row("T", "/wiki/T", str(missing))]
    monkeypatch.setattr(mod, "get_pg_metadata_store", lambda: fake_pg)

    out = mod.WikipediaStorage().get_downloaded_articles()

    assert out == []


def test_get_downloaded_articles_reads_file_and_creates_scraper(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import data_ingestion.wikipedia_storage as mod

    fp = tmp_path / "t.html"
    fp.write_text("<html>ok</html>", encoding="utf-8")

    fake_pg = MagicMock(name="pg")
    fake_pg.enabled = True
    fake_pg.list_articles_with_local_path.return_value = [_Row("T", "/wiki/T", str(fp))]
    monkeypatch.setattr(mod, "get_pg_metadata_store", lambda: fake_pg)

    scraper = object()

    def _fake_scraper(html: str, title: str, url: str | None):
        assert html == "<html>ok</html>"
        assert title == "T"
        assert url == "/wiki/T"
        return scraper

    monkeypatch.setattr(mod, "WikipediaArticleScraper", _fake_scraper)

    out = mod.WikipediaStorage().get_downloaded_articles()

    assert out == [scraper]


def test_article_title_to_filename_unquotes_and_sanitizes() -> None:
    from data_ingestion.wikipedia_storage import WikipediaStorage

    s = WikipediaStorage()

    out = s.article_title_to_filename("Hello%20World/Bad:Name")

    assert out == "Hello_World_Bad_Name"


def test_image_title_to_filename_truncates_and_appends_hash_when_too_long(monkeypatch: pytest.MonkeyPatch) -> None:
    import data_ingestion.wikipedia_storage as mod

    # Make md5 stable and short for deterministic assertion.
    class _FakeMd5:
        def __init__(self, *_a, **_k):
            pass

        def hexdigest(self) -> str:
            return "0123456789abcdef"

    monkeypatch.setattr(mod.hashlib, "md5", lambda *_a, **_k: _FakeMd5())

    s = mod.WikipediaStorage()

    long = "a" * 400 + ".png"
    out = s.image_title_to_filename(long)

    assert len(out) <= 255
    assert out.endswith("_01234567.png")


def test_image_filepath_uses_images_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import data_ingestion.wikipedia_storage as mod

    monkeypatch.setattr(mod, "IMAGES_DIR", tmp_path)

    s = mod.WikipediaStorage()
    fp = s.image_filepath("Foo.png")

    assert fp == tmp_path / "Foo.png"


def test_image_exists_delegates_to_image_filepath(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import data_ingestion.wikipedia_storage as mod

    monkeypatch.setattr(mod, "IMAGES_DIR", tmp_path)

    s = mod.WikipediaStorage()

    existing = tmp_path / "x.png"
    existing.write_bytes(b"x")

    assert s.image_exists("x.png") is True


def test_save_article_to_file_writes_atomic(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import data_ingestion.wikipedia_storage as mod

    monkeypatch.setattr(mod, "ARTICLES_DIR", tmp_path)

    s = mod.WikipediaStorage()

    out_path = s.save_article_to_file("My Title", "<html/>")

    assert out_path.exists()
    assert out_path.read_text(encoding="utf-8") == "<html/>"


def test_extract_image_filename_delegates_to_wikipedia_image_get_filename(monkeypatch: pytest.MonkeyPatch) -> None:
    import data_ingestion.wikipedia_storage as mod

    called = {"url": None}

    class _FakeImage:
        def __init__(self, url: str):
            called["url"] = url

        def get_filename(self) -> str:
            return "X.png"

    monkeypatch.setattr(mod, "WikipediaImage", _FakeImage)

    s = mod.WikipediaStorage()

    assert s._extract_image_filename("http://example/X.png") == "X.png"
    assert called["url"] == "http://example/X.png"

