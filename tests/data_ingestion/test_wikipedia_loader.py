from __future__ import annotations

# Install import-time stubs *before* importing production modules.
from ._import_stubs import install as _install_import_stubs

_install_import_stubs()

from pathlib import Path
from unittest.mock import MagicMock

import pytest


def _mk_storage(*, article_exists: bool = False, image_exists: bool = False, tmp_path: Path | None = None):
    storage = MagicMock(name="storage")
    storage.article_exists.return_value = article_exists
    storage.image_exists.return_value = image_exists
    if tmp_path is not None:
        storage.image_filepath.return_value = tmp_path / "img.png"
        storage.save_article_to_file.return_value = tmp_path / "a.html"
    return storage


def _touch(p: Path, data: bytes = b"x") -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)


def test_load_categories_reads_non_empty_lines(tmp_path: Path) -> None:
    import data_ingestion.wikipedia_loader as mod

    fp = tmp_path / "cats.txt"
    fp.write_text("\n  \nCat1\nCat2\n", encoding="utf-8")

    assert mod.WikipediaLoader().load_categories(str(fp)) == ["Cat1", "Cat2"]


def test_get_all_articles_from_categories_skips_when_fetch_category_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    import data_ingestion.wikipedia_loader as mod

    client = MagicMock(name="client")
    client.fetch_category.return_value = None
    monkeypatch.setattr(mod, "_wikipedia_client", client)

    scraper = MagicMock(name="scraper")
    scraper.extract_articles_from_category.return_value = {"/wiki/A"}
    monkeypatch.setattr(mod, "WikipediaCategoryScraper", lambda html, category: scraper)

    pg = MagicMock(name="pg")
    monkeypatch.setattr(mod, "_postgres", pg)

    loader = mod.WikipediaLoader()
    out = loader.get_all_articles_from_categories(["Cat"])

    assert out == set()
    assert pg.upsert_article_title.call_count == 0
    assert pg.update_article_url.call_count == 0


def test_get_all_articles_from_categories_upserts_titles_and_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    import data_ingestion.wikipedia_loader as mod

    client = MagicMock(name="client")
    client.fetch_category.return_value = "<html/>"
    monkeypatch.setattr(mod, "_wikipedia_client", client)

    discovered = {"/wiki/A%20B", "/wiki/C"}
    scraper = MagicMock(name="scraper")
    scraper.extract_articles_from_category.return_value = discovered
    monkeypatch.setattr(mod, "WikipediaCategoryScraper", lambda html, category: scraper)

    pg = MagicMock(name="pg")
    monkeypatch.setattr(mod, "_postgres", pg)

    loader = mod.WikipediaLoader()
    out = loader.get_all_articles_from_categories(["Cat"])

    assert out == discovered
    assert pg.upsert_article_title.call_count == 2
    assert pg.update_article_url.call_count == 2


def test_fetch_pages_by_titles_skips_existing_articles(monkeypatch: pytest.MonkeyPatch) -> None:
    import data_ingestion.wikipedia_loader as mod

    storage = _mk_storage(article_exists=True)
    monkeypatch.setattr(mod, "_storage", storage)

    client = MagicMock(name="client")
    monkeypatch.setattr(mod, "_wikipedia_client", client)

    pg = MagicMock(name="pg")
    monkeypatch.setattr(mod, "_postgres", pg)

    monkeypatch.setattr(mod.time, "sleep", lambda *_a, **_k: None)

    loader = mod.WikipediaLoader()
    out = loader.fetch_pages_by_titles({"/wiki/A"})

    assert out == []
    assert client.fetch_article.call_count == 0
    assert storage.save_article_to_file.call_count == 0


def test_fetch_pages_by_titles_downloads_and_persists_local_path_and_quality(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import data_ingestion.wikipedia_loader as mod

    storage = _mk_storage(article_exists=False, tmp_path=tmp_path)
    file_path = storage.save_article_to_file.return_value
    _touch(file_path, b"<html>ok</html>")
    monkeypatch.setattr(mod, "_storage", storage)

    client = MagicMock(name="client")
    client.fetch_article.return_value = "<html>ok</html>"
    monkeypatch.setattr(mod, "_wikipedia_client", client)

    pg = MagicMock(name="pg")
    monkeypatch.setattr(mod, "_postgres", pg)

    # Avoid real ArticleScraper complexity; just emulate the methods used.
    fake_article = MagicMock(name="article")
    fake_article._get_visible_text.return_value = "x" * 10
    fake_article.count_citations.return_value = 3
    fake_article.has_problem_or_update_box.return_value = False
    fake_article.is_english_article.return_value = True
    monkeypatch.setattr(mod, "WikipediaArticleScraper", lambda html, title, url: fake_article)

    monkeypatch.setattr(mod.time, "sleep", lambda *_a, **_k: None)

    loader = mod.WikipediaLoader()
    out = loader.fetch_pages_by_titles({"/wiki/A"})

    assert out == ["<html>ok</html>"]
    assert storage.save_article_to_file.call_count == 1
    assert pg.update_article_local_path.call_count == 1
    assert pg.update_article_quality.call_count == 1


def test_fetch_pages_by_titles_does_not_persist_local_path_when_file_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import data_ingestion.wikipedia_loader as mod

    storage = _mk_storage(article_exists=False, tmp_path=tmp_path)
    # Do NOT create the file; simulates write failure / missing file.
    monkeypatch.setattr(mod, "_storage", storage)

    client = MagicMock(name="client")
    client.fetch_article.return_value = "<html>ok</html>"
    monkeypatch.setattr(mod, "_wikipedia_client", client)

    pg = MagicMock(name="pg")
    monkeypatch.setattr(mod, "_postgres", pg)

    monkeypatch.setattr(mod.time, "sleep", lambda *_a, **_k: None)

    loader = mod.WikipediaLoader()
    out = loader.fetch_pages_by_titles({"/wiki/A"})

    assert out == []
    assert pg.update_article_local_path.call_count == 0
    assert pg.update_article_quality.call_count == 0


def test_download_images_normalizes_url_and_persists_metadata_on_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import data_ingestion.wikipedia_loader as mod

    storage = _mk_storage(image_exists=False, tmp_path=tmp_path)
    filepath = storage.image_filepath.return_value
    monkeypatch.setattr(mod, "_storage", storage)

    client = MagicMock(name="client")

    def _download_image(_url: str, target: Path):
        _touch(target, b"img")

    client.download_image.side_effect = _download_image
    client.get_image_license.return_value = {"LicenseShortName": {"value": "CC"}}
    monkeypatch.setattr(mod, "_wikipedia_client", client)

    pg = MagicMock(name="pg")
    monkeypatch.setattr(mod, "_postgres", pg)

    monkeypatch.setattr(mod.time, "sleep", lambda *_a, **_k: None)

    img = mod.WikipediaImage("//upload.wikimedia.org/wikipedia/commons/a/a1/Foo.png")

    loader = mod.WikipediaLoader()
    loader.download_images([img])

    assert img.local_path == filepath
    assert pg.upsert_image_url.call_count == 1
    assert pg.update_image_metadata.call_count == 1
    assert client.download_image.call_count == 1


def test_download_images_skips_when_already_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    import data_ingestion.wikipedia_loader as mod

    storage = _mk_storage(image_exists=True)
    monkeypatch.setattr(mod, "_storage", storage)

    client = MagicMock(name="client")
    monkeypatch.setattr(mod, "_wikipedia_client", client)

    pg = MagicMock(name="pg")
    monkeypatch.setattr(mod, "_postgres", pg)

    monkeypatch.setattr(mod.time, "sleep", lambda *_a, **_k: None)

    img = mod.WikipediaImage("https://upload.wikimedia.org/wikipedia/commons/a/a1/Foo.png")

    loader = mod.WikipediaLoader()
    loader.download_images([img])

    assert client.download_image.call_count == 0
    assert pg.update_image_metadata.call_count == 0
