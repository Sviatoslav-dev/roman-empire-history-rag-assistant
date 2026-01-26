from __future__ import annotations

from pathlib import Path

from data_ingestion.chunk_models import ArticleChunk, ChunkImageMention
from data_ingestion.qdrant_payloads import (
    build_images_collection_payloads,
    build_link_collection_payloads,
    build_text_collection_payloads,
)
from data_ingestion.wikipedia_image import WikipediaImage


def _chunk(
    *,
    text: str = "t",
    page_title: str = "P",
    page_url: str = "u",
    section_title: str = "S",
    section_path: str = "S",
    section_level: int = 1,
    images=None,
) -> ArticleChunk:
    return ArticleChunk(
        page_title=page_title,
        page_url=page_url,
        section_title=section_title,
        section_path=section_path,
        section_level=section_level,
        images=list(images or []),
        text=text,
    )


def test_build_text_collection_payloads_preserves_order_and_fields():
    chunks = [
        _chunk(text="a", page_title="T1", page_url="U1", section_title="S1", section_path="P1", section_level=1),
        _chunk(text="b", page_title="T2", page_url="U2", section_title="S2", section_path="P2", section_level=2),
    ]

    texts, meta = build_text_collection_payloads(chunks)

    assert texts == ["a", "b"]
    assert meta == [
        {
            "page_title": "T1",
            "page_url": "U1",
            "section_title": "S1",
            "section_path": "P1",
            "section_level": 1,
        },
        {
            "page_title": "T2",
            "page_url": "U2",
            "section_title": "S2",
            "section_path": "P2",
            "section_level": 2,
        },
    ]


def test_build_images_collection_payloads_dedups_filters_and_is_deterministic():
    blank = WikipediaImage(url="https://upload.wikimedia.org/wikipedia/en/a/a9/Blank.png")
    img1 = WikipediaImage(url="https://upload.wikimedia.org/wikipedia/commons/a/a1/A.png")
    img2 = WikipediaImage(url="")

    chunks = [
        _chunk(
            images=[
                ChunkImageMention(image=blank, caption="ignored"),
                ChunkImageMention(image=img1, caption="cap1"),
                ChunkImageMention(image=img1, caption="dup"),
                ChunkImageMention(image=img2, caption="empty-url"),
            ]
        )
    ]

    # Only img1 is present in resolver
    def resolver(url: str):
        return {img1.url: str(Path("/tmp/A.png"))}.get(url)

    ids = iter(["id-1", "id-2"])
    uid, paths, meta, image_ids = build_images_collection_payloads(
        chunks,
        local_path_by_url=resolver,
        id_factory=lambda: next(ids),
    )

    assert uid == {
        img1.url: {"image_id": "id-1", "image_url": img1.url, "local_path": "/tmp/A.png"}
    }
    assert paths == ["/tmp/A.png"]
    assert meta == [{"image_url": img1.url, "local_path": "/tmp/A.png"}]
    assert image_ids == ["id-1"]


def test_build_images_collection_payloads_skips_missing_local_path():
    img = WikipediaImage(url="https://upload.wikimedia.org/wikipedia/commons/a/a1/A.png")
    chunks = [_chunk(images=[ChunkImageMention(image=img, caption="x")])]

    uid, paths, meta, image_ids = build_images_collection_payloads(
        chunks,
        local_path_by_url=lambda _url: None,
        id_factory=lambda: "id-1",
    )

    assert uid == {}
    assert paths == []
    assert meta == []
    assert image_ids == []


def test_build_link_collection_payloads_builds_links_and_strips_caption():
    img1 = WikipediaImage(url="https://upload.wikimedia.org/wikipedia/commons/a/a1/A.png")
    img2 = WikipediaImage(url="https://upload.wikimedia.org/wikipedia/commons/b/b2/B.png")

    chunks = [
        _chunk(
            text="a",
            page_title="T",
            page_url="U",
            section_title="S",
            section_path="P",
            section_level=3,
            images=[
                ChunkImageMention(image=img1, caption="  cap1  "),
                ChunkImageMention(image=img2, caption="cap2"),
            ],
        ),
        _chunk(
            text="b",
            page_title="T2",
            page_url="U2",
            section_title="S2",
            section_path="P2",
            section_level=1,
            images=[ChunkImageMention(image=img1, caption="cap3")],
        ),
    ]

    unique = {
        img1.url: {"image_id": "id-A", "image_url": img1.url, "local_path": "/tmp/A.png"},
        # img2 intentionally missing to verify it gets skipped
    }

    links = build_link_collection_payloads(chunks, unique)

    assert links == [
        {
            "text_chunk_id": 0,
            "image_id": "id-A",
            "caption": "cap1",
            "page_title": "T",
            "page_url": "U",
            "section_title": "S",
            "section_path": "P",
            "section_level": 3,
        },
        {
            "text_chunk_id": 1,
            "image_id": "id-A",
            "caption": "cap3",
            "page_title": "T2",
            "page_url": "U2",
            "section_title": "S2",
            "section_path": "P2",
            "section_level": 1,
        },
    ]


def test_build_link_collection_payloads_skips_blank_png_and_empty_url():
    blank = WikipediaImage(url="https://upload.wikimedia.org/wikipedia/en/a/a9/Blank.png")
    empty = WikipediaImage(url="")
    chunks = [_chunk(images=[ChunkImageMention(image=blank, caption="x"), ChunkImageMention(image=empty, caption="y")])]

    links = build_link_collection_payloads(chunks, unique_images_by_url={})

    assert links == []

