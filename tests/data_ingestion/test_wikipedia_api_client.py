from pathlib import Path

import pytest


class _FakePage:
    def __init__(self, url: str):
        self.url = url


class _FakeResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        text: str = "",
        json_data=None,
        headers=None,
        iter_chunks=None,
        raise_for_status_exc: Exception | None = None,
    ):
        self.status_code = status_code
        self.text = text
        self._json_data = json_data
        self.headers = headers or {}
        self._iter_chunks = iter_chunks or []
        self._raise_for_status_exc = raise_for_status_exc

    def raise_for_status(self):
        if self._raise_for_status_exc is not None:
            raise self._raise_for_status_exc

    def json(self):
        return self._json_data

    def iter_content(self, chunk_size: int):
        yield from self._iter_chunks


def test_fetch_article_success(monkeypatch):
    import data_ingestion.wikipedia_api_client as mod

    monkeypatch.setattr(mod.wikipedia, "page", lambda title, auto_suggest=False: _FakePage("http://example/page"))
    monkeypatch.setattr(mod.requests, "get", lambda url, headers=None: _FakeResponse(text="<html>ok</html>"))

    client = mod.WikipediaApiClient()
    html = client.fetch_article("Any")

    assert html == "<html>ok</html>"


def test_fetch_article_disambiguation_picks_first_option(monkeypatch):
    import data_ingestion.wikipedia_api_client as mod

    calls = {"titles": []}

    def fake_page(title, auto_suggest=False):
        calls["titles"].append(title)
        if title == "X":
            # The wikipedia library's DisambiguationError signature differs
            # between versions. We only rely on the `.options` attribute.
            e = mod.wikipedia.exceptions.DisambiguationError("X", ["Y", "Z"])
            e.options = ["Y", "Z"]
            raise e
        return _FakePage("http://example/y")

    monkeypatch.setattr(mod.wikipedia, "page", fake_page)
    monkeypatch.setattr(mod.requests, "get", lambda url, headers=None: _FakeResponse(text="<html>y</html>"))

    html = mod.WikipediaApiClient().fetch_article("X")

    assert html == "<html>y</html>"
    assert calls["titles"] == ["X", "Y"]


def test_fetch_article_empty_html_returns_none(monkeypatch):
    import data_ingestion.wikipedia_api_client as mod

    monkeypatch.setattr(mod.wikipedia, "page", lambda title, auto_suggest=False: _FakePage("http://example/page"))
    monkeypatch.setattr(mod.requests, "get", lambda url, headers=None: _FakeResponse(text=""))

    assert mod.WikipediaApiClient().fetch_article("Any") is None


def test_fetch_article_page_error_returns_none(monkeypatch):
    import data_ingestion.wikipedia_api_client as mod

    def fake_page(title, auto_suggest=False):
        raise mod.PageError(title)

    monkeypatch.setattr(mod.wikipedia, "page", fake_page)

    assert mod.WikipediaApiClient().fetch_article("Missing") is None


def test_fetch_category_success(monkeypatch):
    import data_ingestion.wikipedia_api_client as mod

    def fake_get(url, headers=None, timeout=None):
        assert "Category:Roman_Emperors" in url
        return _FakeResponse(text="<html>cat</html>")

    monkeypatch.setattr(mod.requests, "get", fake_get)

    assert mod.WikipediaApiClient().fetch_category("Roman Emperors") == "<html>cat</html>"


def test_fetch_category_request_exception_returns_none(monkeypatch):
    import data_ingestion.wikipedia_api_client as mod

    def fake_get(url, headers=None, timeout=None):
        raise mod.requests.RequestException("boom")

    monkeypatch.setattr(mod.requests, "get", fake_get)

    assert mod.WikipediaApiClient().fetch_category("Roman Emperors") is None


def test_get_image_license_parses_extmetadata(monkeypatch):
    import data_ingestion.wikipedia_api_client as mod

    data = {
        "query": {
            "pages": {
                "1": {"imageinfo": [{"extmetadata": {"LicenseShortName": {"value": "CC"}}}]}
            }
        }
    }

    def fake_get(url, params=None, headers=None, timeout=None):
        assert params["titles"].startswith("File:")
        return _FakeResponse(json_data=data)

    monkeypatch.setattr(mod.requests, "get", fake_get)

    meta = mod.WikipediaApiClient().get_image_license("A.png")
    assert meta == {"LicenseShortName": {"value": "CC"}}


def test_get_image_license_missing_imageinfo_returns_none(monkeypatch):
    import data_ingestion.wikipedia_api_client as mod

    data = {"query": {"pages": {"1": {}}}}

    monkeypatch.setattr(mod.requests, "get", lambda *a, **k: _FakeResponse(json_data=data))

    assert mod.WikipediaApiClient().get_image_license("A.png") is None


def test_download_image_429_retries_and_writes_file(tmp_path, monkeypatch):
    import data_ingestion.wikipedia_api_client as mod

    target = tmp_path / "img.png"
    tmp_created: dict[str, str | None] = {"name": None}

    first = _FakeResponse(status_code=429, headers={"Retry-After": "0"})
    second = _FakeResponse(
        status_code=200,
        headers={"content-type": "image/png", "content-length": "3"},
        iter_chunks=[b"abc"],
    )
    responses = iter([first, second])

    def fake_get(*args, **kwargs):
        return next(responses)

    monkeypatch.setattr(mod.requests, "get", fake_get)
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)

    # Use a real temp file on disk but in tmp_path
    def fake_mkstemp(prefix: str, dir: str):
        p = Path(dir) / f"{prefix}.tmp"
        tmp_created["name"] = str(p)
        fd = mod.os.open(p, mod.os.O_CREAT | mod.os.O_RDWR)
        return fd, str(p)

    monkeypatch.setattr(mod.tempfile, "mkstemp", fake_mkstemp)

    # no-op fsync for speed
    monkeypatch.setattr(mod.os, "fsync", lambda _fd: None)

    result = mod.WikipediaApiClient().download_image("http://example/img", target)

    assert result == str(target)
    assert target.read_bytes() == b"abc"


def test_download_image_non_image_returns_response(tmp_path, monkeypatch):
    import data_ingestion.wikipedia_api_client as mod

    target = tmp_path / "x.bin"

    resp = _FakeResponse(status_code=200, headers={"content-type": "text/html", "content-length": "0"})
    monkeypatch.setattr(mod.requests, "get", lambda *a, **k: resp)

    out = mod.WikipediaApiClient().download_image("http://example/x", target)

    assert out is resp
    assert not target.exists()


def test_download_image_write_failure_cleans_tmp(tmp_path, monkeypatch):
    import data_ingestion.wikipedia_api_client as mod

    target = tmp_path / "img.png"

    resp = _FakeResponse(
        status_code=200,
        headers={"content-type": "image/png", "content-length": "3"},
        iter_chunks=[b"abc"],
    )
    monkeypatch.setattr(mod.requests, "get", lambda *a, **k: resp)

    # create temp file
    def fake_mkstemp(prefix: str, dir: str):
        p = Path(dir) / f"{prefix}.tmp"
        fd = mod.os.open(p, mod.os.O_CREAT | mod.os.O_RDWR)
        return fd, str(p)

    monkeypatch.setattr(mod.tempfile, "mkstemp", fake_mkstemp)

    # Make replace fail to force cleanup path
    monkeypatch.setattr(mod.os, "replace", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("replace failed")))

    # Track unlink calls
    unlinked = {"paths": []}

    def fake_unlink(p):
        unlinked["paths"].append(str(p))

    monkeypatch.setattr(mod.os, "unlink", fake_unlink)
    monkeypatch.setattr(mod.os, "fsync", lambda _fd: None)

    with pytest.raises(RuntimeError, match="replace failed"):
        mod.WikipediaApiClient().download_image("http://example/img", target)

    assert unlinked["paths"], "tmp file should be unlinked on failure"
