"""Import-time stubs for unit tests.

Why this file exists:
Some production modules open DB connections / create singletons at IMPORT time.
Pytest fixtures (even autouse) run *after* the test module is imported, so they
can't prevent those side effects.

We keep these stubs in a single place and activate them via `tests/__init__.py`
so they are installed as soon as the `tests` package is imported.

If you need real integrations (testing PgMetadataStore/WikipediaArticleFilter
against a real DB, etc.), disable these stubs by setting:

    DISABLE_IMPORT_STUBS=1

before running pytest.
"""

import os
import sys
import types


def _disabled() -> bool:
    return os.getenv("DISABLE_IMPORT_STUBS", "").strip() in {"1", "true", "True", "yes", "YES"}


def install() -> None:
    """Install stub modules into sys.modules.

    Call this at the very top of a unit test module *before* importing the
    production modules that have import-time side effects.

    To force real imports (e.g. for integration tests), set DISABLE_IMPORT_STUBS=1.
    """
    if _disabled():
        return

    # ----- cairosvg (external dependency requiring native cairo) -----
    # CairoSVG imports cairocffi which needs system cairo installed. For unit
    # tests we only need the `svg2png()` API and we patch it anyway.
    if "cairosvg" not in sys.modules:
        fake_cairosvg = types.ModuleType("cairosvg")

        def svg2png(*args, **kwargs):  # pragma: no cover
            raise RuntimeError("cairosvg.svg2png stub called unexpectedly")

        fake_cairosvg.svg2png = svg2png
        sys.modules["cairosvg"] = fake_cairosvg

    # ----- pg_metadata_store (prevent import-time DB connections) -----
    # Many ingestion modules call get_pg_metadata_store() at import time. In unit
    # tests, we replace the module with a lightweight stub exposing the same API.
    # Tests that need the real implementation can set DISABLE_IMPORT_STUBS=1.
    if "data_ingestion.pg_metadata_store" not in sys.modules:
        fake_pg = types.ModuleType("data_ingestion.pg_metadata_store")

        class ArticleQuality:  # pragma: no cover
            def __init__(
                self,
                symbols_number: int,
                citations_number: int,
                has_problem_or_update_box: bool,
                is_english_available: bool,
            ) -> None:
                self.symbols_number = symbols_number
                self.citations_number = citations_number
                self.has_problem_or_update_box = has_problem_or_update_box
                self.is_english_available = is_english_available

        class IngestionArticleRow:  # pragma: no cover
            def __init__(self, title: str, url: str | None, local_path: str) -> None:
                self.title = title
                self.url = url
                self.local_path = local_path

        class IngestionImageRow:  # pragma: no cover
            def __init__(self, url: str, local_path: str, extension: str | None) -> None:
                self.url = url
                self.local_path = local_path
                self.extension = extension

        class IngestionImageRowWithMetadata:  # pragma: no cover
            def __init__(
                self,
                url: str,
                licence: str | None,
                local_path: str | None,
                filename: str | None,
                extension: str | None,
            ) -> None:
                self.url = url
                self.licence = licence
                self.local_path = local_path
                self.filename = filename
                self.extension = extension

        class PgMetadataStore:  # pragma: no cover
            def get_image_by_url(self, url: str):
                return None

            def upsert_image(self, *args, **kwargs):
                return None

            def upsert_article(self, *args, **kwargs):
                return None

            def upsert_article_title(self, *args, **kwargs):
                return None

            def update_article_url(self, *args, **kwargs):
                return None

            def update_image_metadata(self, *args, **kwargs):
                return None

            def list_svg_images(self, *args, **kwargs):
                return []

        def get_pg_metadata_store():  # pragma: no cover
            return PgMetadataStore()

        fake_pg.ArticleQuality = ArticleQuality
        fake_pg.IngestionArticleRow = IngestionArticleRow
        fake_pg.IngestionImageRow = IngestionImageRow
        fake_pg.IngestionImageRowWithMetadata = IngestionImageRowWithMetadata
        fake_pg.PgMetadataStore = PgMetadataStore
        fake_pg.get_pg_metadata_store = get_pg_metadata_store

        sys.modules["data_ingestion.pg_metadata_store"] = fake_pg

    # NOTE:
    # We intentionally do NOT stub `data_ingestion.wikipedia_article_filter`.
    # That module is pure enough for unit tests as long as pg_metadata_store is
    # stubbed (it creates its _postgres singleton at import time).
