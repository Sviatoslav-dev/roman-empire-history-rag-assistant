from ._import_stubs import install as _install_import_stubs

_install_import_stubs()

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import data_ingestion.wikipedia_collector as mod


@pytest.fixture()
def collector_mocks(monkeypatch: pytest.MonkeyPatch):
    loader = MagicMock(name="loader")
    storage = MagicMock(name="storage")
    chunk_processor = MagicMock(name="chunk_processor")

    # Patch module-level article filter singleton to be controllable in tests.
    article_filter = MagicMock(name="article_filter")
    monkeypatch.setattr(mod, "_article_filter", article_filter)

    retriever = MagicMock(name="retriever")
    collector = mod.WikipediaCollector(loader, storage, chunk_processor, retriever=retriever)

    return collector, loader, storage, chunk_processor, retriever, article_filter


def test_collect_articles_happy_path_wires_everything(collector_mocks) -> None:
    collector, loader, storage, chunk_processor, retriever, article_filter = collector_mocks

    categories_file = "cats.txt"
    categories = ["Cat1", "Cat2"]
    article_titles = {"A", "B"}

    a1 = SimpleNamespace(title="A")
    a2 = SimpleNamespace(title="B")

    loader.load_categories.return_value = categories
    loader.get_all_articles_from_categories.return_value = article_titles
    storage.get_downloaded_articles.return_value = [a1, a2]
    article_filter.filter_articles.return_value = [a1]

    chunk_texts = ["t1", "t2"]
    chunk_metadata = [{"i": 1}, {"i": 2}]
    unique_images_by_url = {"https://img/1.png": {"image_id": "1"}}
    image_paths = ["/tmp/1.png"]
    image_metadata = [{"url": "https://img/1.png"}]
    image_ids = ["1"]
    links = [{"text_chunk_id": 0, "image_id": "1"}]

    chunk_processor.prepare_text_collection.return_value = (chunk_texts, chunk_metadata)
    chunk_processor.prepare_images_collection.return_value = (
        unique_images_by_url,
        image_paths,
        image_metadata,
        image_ids,
    )
    chunk_processor.prepare_link_collection.return_value = links

    out = collector.collect_articles(categories_file)

    assert out == [a1]

    loader.load_categories.assert_called_once_with(categories_file)
    loader.get_all_articles_from_categories.assert_called_once_with(categories)
    loader.fetch_pages_by_titles.assert_called_once_with(article_titles)

    storage.get_downloaded_articles.assert_called_once_with()
    article_filter.filter_articles.assert_called_once_with([a1, a2])

    chunk_processor.split_articles_into_chunks.assert_called_once_with([a1])
    chunk_processor.download_images.assert_called_once_with()
    chunk_processor.postprocess_images.assert_called_once_with()
    chunk_processor.filter_images_by_license.assert_called_once_with()

    chunk_processor.prepare_text_collection.assert_called_once_with()
    chunk_processor.prepare_images_collection.assert_called_once_with()
    chunk_processor.prepare_link_collection.assert_called_once_with(unique_images_by_url)

    retriever.add_text_chunks.assert_called_once_with(chunk_texts, chunk_metadata, ids=[0, 1])
    retriever.add_images.assert_called_once_with(image_paths, image_metadata, ids=['1'])
    retriever.add_chunk_image_links.assert_called_once_with(links, ids=[0])


def test_collect_articles_returns_empty_when_no_article_urls(collector_mocks, monkeypatch: pytest.MonkeyPatch) -> None:
    collector, loader, storage, chunk_processor, retriever, article_filter = collector_mocks

    loader.load_categories.return_value = ["Cat"]
    loader.get_all_articles_from_categories.return_value = {}

    warn = MagicMock()
    monkeypatch.setattr(mod.logger, "warning", warn)

    out = collector.collect_articles("cats.txt")

    assert out == []
    warn.assert_any_call("No article titles discovered from categories; aborting fetch.")

    loader.fetch_pages_by_titles.assert_not_called()
    storage.get_downloaded_articles.assert_not_called()
    article_filter.filter_articles.assert_not_called()
    chunk_processor.split_articles_into_chunks.assert_not_called()
    retriever.add_text_chunks.assert_not_called()


def test_collect_articles_logs_when_no_valid_categories_but_still_early_exits(collector_mocks, monkeypatch: pytest.MonkeyPatch) -> None:
    collector, loader, storage, chunk_processor, retriever, article_filter = collector_mocks

    loader.load_categories.return_value = []
    loader.get_all_articles_from_categories.return_value = {}

    warn = MagicMock()
    monkeypatch.setattr(mod.logger, "warning", warn)

    out = collector.collect_articles("cats.txt")

    assert out == []
    warn.assert_any_call("No valid categories provided in %s.", "cats.txt")
    warn.assert_any_call("No article titles discovered from categories; aborting fetch.")


def test_collect_articles_skips_add_images_when_no_image_paths(collector_mocks) -> None:
    collector, loader, storage, chunk_processor, retriever, article_filter = collector_mocks

    a1 = SimpleNamespace(title="A")

    loader.load_categories.return_value = ["Cat"]
    loader.get_all_articles_from_categories.return_value = {"A"}
    storage.get_downloaded_articles.return_value = [a1]
    article_filter.filter_articles.return_value = [a1]

    chunk_processor.prepare_text_collection.return_value = ([], [])
    chunk_processor.prepare_images_collection.return_value = ({}, [], [], [])
    chunk_processor.prepare_link_collection.return_value = []

    out = collector.collect_articles("cats.txt")

    assert out == [a1]
    retriever.add_text_chunks.assert_called_once_with([], [], ids=[])
    retriever.add_images.assert_not_called()
    retriever.add_chunk_image_links.assert_not_called()


def test_collect_articles_falls_back_when_image_ids_are_not_ints(collector_mocks) -> None:
    collector, loader, storage, chunk_processor, retriever, article_filter = collector_mocks

    a1 = SimpleNamespace(title="A")

    loader.load_categories.return_value = ["Cat"]
    loader.get_all_articles_from_categories.return_value = {"A"}
    storage.get_downloaded_articles.return_value = [a1]
    article_filter.filter_articles.return_value = [a1]

    chunk_processor.prepare_text_collection.return_value = ([], [])
    chunk_processor.prepare_images_collection.return_value = (
        {"https://img/x.png": {"image_id": "abc"}},
        ["/tmp/x.png"],
        [{"url": "https://img/x.png"}],
        ["abc"],
    )
    chunk_processor.prepare_link_collection.return_value = []

    out = collector.collect_articles("cats.txt")

    assert out == [a1]
    retriever.add_images.assert_called_once_with(["/tmp/x.png"], [{"url": "https://img/x.png"}], ids=["abc"])
