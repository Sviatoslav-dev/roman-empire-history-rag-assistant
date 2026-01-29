import importlib
import sys

import pytest


def _import_real_pg_metadata_store():
    """Import the real data_ingestion.pg_metadata_store even if stubs are installed."""
    sys.modules.pop("data_ingestion.pg_metadata_store", None)
    return importlib.import_module("data_ingestion.pg_metadata_store")


class _FakeCursor:
    def __init__(self, execute_side_effect=None):
        self.executed = []
        self._execute_side_effect = execute_side_effect

    def execute(self, sql, params=None):
        if self._execute_side_effect is not None:
            raise self._execute_side_effect
        self.executed.append((sql, params))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeConn:
    def __init__(self, cursor: _FakeCursor):
        self._cursor = cursor
        self.cursor_calls = 0

    def cursor(self):
        self.cursor_calls += 1
        return self._cursor

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_ensure_schema_executes_ddl(monkeypatch):
    mod = _import_real_pg_metadata_store()
    PgMetadataStore = mod.PgMetadataStore

    store = PgMetadataStore.__new__(PgMetadataStore)
    store._dsn = "postgresql://ignored"
    store._enabled = True

    cur = _FakeCursor()
    conn = _FakeConn(cur)

    monkeypatch.setattr(store, "_connect", lambda: conn)

    store._ensure_schema()

    assert conn.cursor_calls == 1
    assert len(cur.executed) == 1

    ddl_sql, ddl_params = cur.executed[0]
    assert ddl_params is None

    ddl_text = str(ddl_sql).lower()
    assert "create table if not exists ingestion_article" in ddl_text
    assert "create table if not exists ingestion_image" in ddl_text


def test_ensure_schema_logs_and_reraises_on_error(monkeypatch):
    mod = _import_real_pg_metadata_store()
    PgMetadataStore = mod.PgMetadataStore

    store = PgMetadataStore.__new__(PgMetadataStore)
    store._dsn = "postgresql://ignored"
    store._enabled = True

    err = RuntimeError("boom")
    cur = _FakeCursor(execute_side_effect=err)
    conn = _FakeConn(cur)

    monkeypatch.setattr(store, "_connect", lambda: conn)

    calls = {"exception": 0}

    def _fake_exception(msg, *args, **kwargs):
        calls["exception"] += 1

    monkeypatch.setattr(mod.logger, "exception", _fake_exception)

    with pytest.raises(RuntimeError, match="boom"):
        store._ensure_schema()

    assert calls["exception"] == 1
