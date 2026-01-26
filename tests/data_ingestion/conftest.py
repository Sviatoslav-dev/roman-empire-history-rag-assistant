"""Pytest configuration.

This conftest installs import-time stubs to prevent unit tests from triggering:
- Postgres connections (created by ingestion singletons at import time)
- optional native deps imports (CairoSVG/cairo)

Disable by setting DISABLE_IMPORT_STUBS=1.
"""


def pytest_configure(config):
    from ._import_stubs import install

    install()
