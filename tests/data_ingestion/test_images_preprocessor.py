from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest


@dataclass(frozen=True)
class _Row:
    url: str
    local_path: str
    extension: str | None = ".svg"


class _FakePg:
    def __init__(self, rows: list[_Row]):
        self._rows = rows
        self.updated: list[dict] = []

    def list_svg_images(self):
        return list(self._rows)

    def update_image_metadata(self, *, url: str, local_path: str, extension: str):
        self.updated.append({"url": url, "local_path": local_path, "extension": extension})


class _FakeCairoSVG:
    def __init__(self):
        self.calls: list[dict] = []

    def svg2png(self, *, url: str, write_to: str):
        self.calls.append({"url": url, "write_to": write_to})
        # simulate successful conversion by writing something to the output
        Path(write_to).write_bytes(b"png")


@pytest.fixture()
def fake_cairosvg(monkeypatch):
    """Patch `data_ingestion.images_preprocessor.cairosvg` with a tiny fake."""
    from data_ingestion import images_preprocessor as mod

    fake = _FakeCairoSVG()
    monkeypatch.setattr(mod, "cairosvg", fake)
    return fake


def test_convert_one_svg_row_missing_file(tmp_path, monkeypatch, fake_cairosvg):
    from data_ingestion.images_preprocessor import ImagesPreprocessor
    from data_ingestion import images_preprocessor as mod

    fake_pg = _FakePg(rows=[])
    monkeypatch.setattr(mod, "_postgres", fake_pg)

    missing = tmp_path / "missing.svg"
    ok = ImagesPreprocessor()._convert_one_svg_row(
        _Row(url="http://example/svg", local_path=str(missing)),
        remove_original=False,
    )

    assert ok is False
    assert fake_cairosvg.calls == []
    assert fake_pg.updated == []


def test_convert_one_svg_row_idempotent_existing_png_updates_db(tmp_path, monkeypatch, fake_cairosvg):
    from data_ingestion.images_preprocessor import ImagesPreprocessor
    from data_ingestion import images_preprocessor as mod

    svg = tmp_path / "a.svg"
    svg.write_text("<svg></svg>")

    png = tmp_path / "a.png"
    png.write_bytes(b"already")

    fake_pg = _FakePg(rows=[])
    monkeypatch.setattr(mod, "_postgres", fake_pg)

    ok = ImagesPreprocessor()._convert_one_svg_row(
        _Row(url="http://example/a.svg", local_path=str(svg)),
        remove_original=False,
    )

    # idempotent path: it updates DB but doesn't count as newly converted
    assert ok is False
    assert fake_cairosvg.calls == []
    assert fake_pg.updated == [
        {"url": "http://example/a.svg", "local_path": str(png), "extension": ".png"}
    ]
    assert svg.exists()
    assert png.exists()


def test_convert_one_svg_row_success_converts_and_optionally_removes_source(tmp_path, monkeypatch, fake_cairosvg):
    from data_ingestion.images_preprocessor import ImagesPreprocessor
    from data_ingestion import images_preprocessor as mod

    svg = tmp_path / "b.svg"
    svg.write_text("<svg></svg>")

    fake_pg = _FakePg(rows=[])
    monkeypatch.setattr(mod, "_postgres", fake_pg)

    ok = ImagesPreprocessor()._convert_one_svg_row(
        _Row(url="http://example/b.svg", local_path=str(svg)),
        remove_original=True,
    )

    png = tmp_path / "b.png"
    assert ok is True
    assert fake_cairosvg.calls == [{"url": str(svg), "write_to": str(png)}]
    assert fake_pg.updated == [
        {"url": "http://example/b.svg", "local_path": str(png), "extension": ".png"}
    ]
    assert png.exists() and png.stat().st_size > 0
    assert not svg.exists()


def test_convert_svgs_to_png_counts_only_new_conversions(tmp_path, monkeypatch, fake_cairosvg):
    from data_ingestion.images_preprocessor import ImagesPreprocessor
    from data_ingestion import images_preprocessor as mod

    svg1 = tmp_path / "c.svg"
    svg1.write_text("<svg></svg>")

    svg2 = tmp_path / "d.svg"
    svg2.write_text("<svg></svg>")
    (tmp_path / "d.png").write_bytes(b"already")

    rows = [
        _Row(url="http://example/c.svg", local_path=str(svg1)),
        _Row(url="http://example/d.svg", local_path=str(svg2)),
    ]
    fake_pg = _FakePg(rows=rows)
    monkeypatch.setattr(mod, "_postgres", fake_pg)

    converted = ImagesPreprocessor().convert_svgs_to_png(remove_original=False)

    # c.svg converts; d.svg is idempotent => total 1
    assert converted == 1
    assert (tmp_path / "c.png").exists()
    assert (tmp_path / "d.png").exists()
    assert len(fake_pg.updated) == 2

