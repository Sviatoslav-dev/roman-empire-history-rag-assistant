from tests._import_stubs import install as _install_import_stubs

_install_import_stubs()

from data_ingestion.wikipedia_image import WikipediaImage


def test_convert_thumbnail_to_fullsize_converts_thumb_url() -> None:
    img = WikipediaImage(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a1/Foo.png/220px-Foo.png"
    )

    img.convert_thumbnail_to_fullsize()

    assert img.url == "https://upload.wikimedia.org/wikipedia/commons/a/a1/Foo.png"


def test_convert_thumbnail_to_fullsize_noop_when_not_thumb() -> None:
    url = "https://upload.wikimedia.org/wikipedia/commons/a/a1/Foo.png"
    img = WikipediaImage(url)

    img.convert_thumbnail_to_fullsize()

    assert img.url == url


def test_normalize_url_converts_thumb_and_keeps_query_free() -> None:
    img = WikipediaImage(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a1/Foo.png/220px-Foo.png?abc=1"
    )

    img.normalize_url()

    # normalize converts thumb -> fullsize but preserves query if present (urlparse keeps it)
    assert img.url.startswith("https://upload.wikimedia.org/wikipedia/commons/a/a1/Foo.png")


def test_normalize_url_adds_https_for_protocol_relative() -> None:
    img = WikipediaImage("//upload.wikimedia.org/wikipedia/commons/a/a1/Foo.png")

    img.normalize_url()

    assert img.url == "https://upload.wikimedia.org/wikipedia/commons/a/a1/Foo.png"


def test_normalize_url_prefixes_wikipedia_host_for_root_relative() -> None:
    img = WikipediaImage("/wiki/File:Foo.png")

    img.normalize_url(wikipedia_host="https://en.wikipedia.org")

    assert img.url == "https://en.wikipedia.org/wiki/File:Foo.png"


def test_normalize_url_removes_duplicate_last_segment() -> None:
    img = WikipediaImage("https://example.com/path/Foo.png/Foo.png")

    img.normalize_url()

    assert img.url == "https://example.com/path/Foo.png"


def test_normalize_url_duplicate_last_segment_ignores_query_in_comparison() -> None:
    img = WikipediaImage("https://example.com/path/Foo.png/Foo.png?x=1")

    img.normalize_url()

    assert img.url == "https://example.com/path/Foo.png"


def test_get_filename_decodes_percent_encoding() -> None:
    img = WikipediaImage("https://upload.wikimedia.org/wikipedia/commons/a/a1/Hello%20World.png")

    assert img.get_filename() == "Hello World.png"


def test_normalize_url_returns_self_for_chaining() -> None:
    img = WikipediaImage("//upload.wikimedia.org/wikipedia/commons/a/a1/Foo.png")

    out = img.normalize_url()

    assert out is img

