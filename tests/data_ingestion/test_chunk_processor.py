from __future__ import annotations

# Install import-time stubs *before* importing production modules.
from ._import_stubs import install as _install_import_stubs

_install_import_stubs()

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from data_ingestion.chunk_models import ArticleChunk, ChunkImageMention
from data_ingestion.chunk_processor import ChunkProcessor
from data_ingestion.wikipedia_image import WikipediaImage


def _mk_chunk(*, img_urls: list[str] | None = None) -> ArticleChunk:
    img_urls = img_urls or []
    mentions = [ChunkImageMention(WikipediaImage(url=u), caption="cap") for u in img_urls]
    return ArticleChunk(
        page_title="T",
        page_url="/wiki/T",
        section_title="Intro",
        section_path="Intro",
        section_level=1,
        text_parts=["Hello"],
        images=mentions,
    )


def test_split_articles_into_chunks_flattens_and_sets_self_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    loader = MagicMock()
    article_filter = MagicMock()
    metadata_store = MagicMock()

    processor = ChunkProcessor(loader=loader, article_filter=article_filter, metadata_store=metadata_store)

    # Two scraper-like objects with split_by_chunks
    a1 = MagicMock()
    a1.split_by_chunks.return_value = [
        ArticleChunk("T1", "/wiki/T1", "Intro", "Intro", 1, text_parts=["a"])
    ]

    a2 = MagicMock()
    a2.split_by_chunks.return_value = [
        ArticleChunk("T2", "/wiki/T2", "Intro", "Intro", 1, text_parts=["b"]),
        ArticleChunk("T2", "/wiki/T2", "History", "History", 2, text_parts=["c"]),
    ]

    chunks = processor.split_articles_into_chunks([a1, a2])

    assert len(chunks) == 3
    assert processor.chunks is chunks

    a1.split_by_chunks.assert_called_once_with(processor.MAX_CHUNK_SIZE)
    a2.split_by_chunks.assert_called_once_with(processor.MAX_CHUNK_SIZE)


def test_extend_chunks_appends_to_internal_list() -> None:
    processor = ChunkProcessor(loader=MagicMock(), article_filter=MagicMock(), metadata_store=MagicMock())

    c1 = ArticleChunk("T", "/wiki/T", "Intro", "Intro", 1, text_parts=["a"])
    c2 = ArticleChunk("T", "/wiki/T", "Intro", "Intro", 1, text_parts=["b"])

    processor.extend_chunks([c1])
    processor.extend_chunks([c2])

    assert processor.chunks == [c1, c2]


def test_download_images_flattens_mentions_and_calls_loader() -> None:
    loader = MagicMock()
    processor = ChunkProcessor(loader=loader, article_filter=MagicMock(), metadata_store=MagicMock())

    processor.chunks = [
        _mk_chunk(img_urls=["https://img/1.png", "https://img/2.png"]),
        _mk_chunk(img_urls=["https://img/3.png"]),
    ]

    processor.download_images()

    # Ensure we pass the WikipediaImage objects (not mentions)
    args, _kwargs = loader.download_images.call_args
    images = args[0]
    assert [i.url for i in images] == ["https://img/1.png", "https://img/2.png", "https://img/3.png"]


def test_postprocess_images_calls_images_preprocessor() -> None:
    pre = MagicMock()
    processor = ChunkProcessor(loader=MagicMock(), article_filter=MagicMock(), metadata_store=MagicMock(), images_preprocessor=pre)

    processor.postprocess_images()
    pre.convert_svgs_to_png.assert_called_once_with()


def test_filter_images_by_license_replaces_internal_chunks() -> None:
    article_filter = MagicMock()
    processor = ChunkProcessor(loader=MagicMock(), article_filter=article_filter, metadata_store=MagicMock())

    original = [_mk_chunk(img_urls=["https://img/a.png"])]
    filtered = [_mk_chunk(img_urls=[])]
    processor.chunks = original

    article_filter.filter_chunk_images.return_value = filtered

    processor.filter_images_by_license()

    article_filter.filter_chunk_images.assert_called_once_with(original)
    assert processor.chunks == filtered


def test_prepare_text_collection_delegates_to_payload_builder(monkeypatch: pytest.MonkeyPatch) -> None:
    processor = ChunkProcessor(loader=MagicMock(), article_filter=MagicMock(), metadata_store=MagicMock())
    processor.chunks = [ArticleChunk("T", "/wiki/T", "Intro", "Intro", 1, text_parts=["hello"], text="hello")]

    stub_out = (["txt"], [{"page_title": "T"}])

    import data_ingestion.chunk_processor as mod

    def _fake_builder(chunks):
        assert chunks is processor.chunks
        return stub_out

    monkeypatch.setattr(mod, "build_text_collection_payloads", _fake_builder)

    assert processor.prepare_text_collection() == stub_out


def test_prepare_images_collection_resolves_local_path_via_metadata_store(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    metadata_store = MagicMock()
    processor = ChunkProcessor(loader=MagicMock(), article_filter=MagicMock(), metadata_store=metadata_store)

    # One mention; build_images_collection_payloads will call local_path_by_url
    processor.chunks = [_mk_chunk(img_urls=["https://img/1.png"])]

    def _get_image_by_url(url: str):
        if url == "https://img/1.png":
            return SimpleNamespace(local_path=tmp_path / "1.png")
        return None

    metadata_store.get_image_by_url.side_effect = _get_image_by_url

    import data_ingestion.chunk_processor as mod

    def _fake_builder(chunks, *, local_path_by_url):
        assert chunks is processor.chunks
        assert local_path_by_url("https://img/1.png").endswith("1.png")
        assert local_path_by_url("https://img/missing.png") is None
        return ({}, [], [], [])

    monkeypatch.setattr(mod, "build_images_collection_payloads", _fake_builder)

    assert processor.prepare_images_collection() == ({}, [], [], [])
    metadata_store.get_image_by_url.assert_any_call("https://img/1.png")


def test_prepare_link_collection_delegates_to_payload_builder(monkeypatch: pytest.MonkeyPatch) -> None:
    processor = ChunkProcessor(loader=MagicMock(), article_filter=MagicMock(), metadata_store=MagicMock())
    processor.chunks = [ArticleChunk("T", "/wiki/T", "Intro", "Intro", 1, text_parts=["hello"], text="hello")]

    unique = {"https://img/1.png": {"image_id": "id1"}}
    expected = [{"text_chunk_id": 0, "image_id": "id1"}]

    import data_ingestion.chunk_processor as mod

    def _fake_builder(chunks, unique_images_by_url):
        assert chunks is processor.chunks
        assert unique_images_by_url is unique
        return expected

    monkeypatch.setattr(mod, "build_link_collection_payloads", _fake_builder)

    assert processor.prepare_link_collection(unique) == expected

