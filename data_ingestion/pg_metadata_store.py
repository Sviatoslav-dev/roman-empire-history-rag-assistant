from __future__ import annotations

from dotenv import load_dotenv

"""PostgreSQL metadata store for ingestion.

This module stores *intermediate ingestion metadata* (what we downloaded, where we saved it,
quality metrics) and is intentionally separate from Qdrant, which stores retrieval vectors.

Design goals:
- incremental updates (create record when title is known, update when URL/path/metrics become known)
- idempotent upserts
- very small API surface that ingestion code can call without carrying DB logic around

Configuration:
- set `POSTGRES_DSN`, e.g. postgresql://user:pass@localhost:5432/roman_ingestion

Tables are created automatically on first use (can be replaced by migrations later).
"""

from dataclasses import dataclass
import os
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.sql import SQL

from logger import get_logger

logger = get_logger(__name__)

load_dotenv()
DEFAULT_DSN = os.getenv("POSTGRES_DSN")


@dataclass(frozen=True)
class ArticleQuality:
    symbols_number: int
    citations_number: int
    has_problem_or_update_box: bool
    is_english_available: bool


@dataclass(frozen=True)
class IngestionArticleRow:
    title: str
    url: str | None
    local_path: str


@dataclass(frozen=True)
class IngestionImageRow:
    url: str
    local_path: str
    extension: str | None


@dataclass(frozen=True)
class IngestionImageRowWithMetadata:
    url: str
    licence: str | None
    local_path: str | None
    filename: str | None
    extension: str | None


class PgMetadataStore:
    """PostgreSQL-backed metadata store used during ingestion."""

    def __init__(
        self,
        dsn: str | None = None,
        *,
        enabled: bool | None = True,
    ) -> None:
        """Create the store.

        Args:
            dsn: PostgreSQL DSN. Defaults to POSTGRES_DSN env var.
            enabled: If None, auto-enables when dsn is present. If True, requires dsn.

        Raises:
            ValueError: When enabled is True but DSN is missing.
        """
        self._dsn = dsn or DEFAULT_DSN
        if enabled is None:
            enabled = bool(self._dsn)
        self._enabled = enabled

        if self._enabled and not self._dsn:
            raise ValueError("PostgreSQL enabled but POSTGRES_DSN is empty")

        if self._enabled:
            self._ensure_schema()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _connect(self) -> psycopg.Connection[Any]:
        """Open a PostgreSQL connection.

        Uses autocommit because ingestion updates are small, independent upserts.
        """
        if not self._dsn:
            # This should be guarded by __init__, but keep mypy and callers safe.
            raise ValueError("POSTGRES_DSN is empty")

        # autocommit keeps calls small and safe for incremental updates
        return psycopg.connect(self._dsn, autocommit=True, row_factory=dict_row)

    def _ensure_schema(self) -> None:
        """Create ingestion tables/indexes if they don't exist.

        Called automatically when store is enabled.

        Raises:
            Exception: Re-raises DB errors after logging.
        """
        ddl = SQL(
            """
        create table if not exists ingestion_article (
            id bigserial primary key,
            title text not null unique,
            url text,
            local_path text,
            symbols_number integer,
            citations_number integer,
            has_problem_or_update_box boolean,
            is_english_available boolean,
            created_at timestamptz not null default now(),
            updated_at timestamptz not null default now()
        );

        create table if not exists ingestion_image (
            id bigserial primary key,
            url text not null unique,
            licence text,
            local_path text,
            filename text,
            extension text,
            created_at timestamptz not null default now(),
            updated_at timestamptz not null default now()
        );

        create index if not exists idx_ingestion_article_url on ingestion_article(url);
        create index if not exists idx_ingestion_image_filename on ingestion_image(filename);
        """
        )
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(ddl)
        except Exception:
            logger.exception("Failed to ensure PostgreSQL schema")
            raise

    def _execute(self, sql: SQL, params: tuple) -> None:
        """Execute a statement if enabled.

        This is used by upsert/update methods to centralize the `enabled` guard.
        """
        if not self._enabled:
            return
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)

    # ------------------------- Article incremental upserts -------------------------

    def upsert_article_title(self, title: str) -> None:
        """Create the record as soon as we learn the title."""
        sql = SQL(
            """
        insert into ingestion_article(title)
        values (%s)
        on conflict (title) do update
        set updated_at = now();
        """
        )
        self._execute(sql, (title,))

    def update_article_url(self, title: str, url: str) -> None:
        """Upsert an article URL for a given title."""
        sql = SQL(
            """
        insert into ingestion_article(title, url)
        values (%s, %s)
        on conflict (title) do update
        set url = excluded.url,
            updated_at = now();
        """
        )
        self._execute(sql, (title, url))

    def update_article_local_path(self, title: str, local_path: str) -> None:
        """Upsert the local HTML file path for a given title."""
        sql = SQL(
            """
        insert into ingestion_article(title, local_path)
        values (%s, %s)
        on conflict (title) do update
        set local_path = excluded.local_path,
            updated_at = now();
        """
        )
        self._execute(sql, (title, local_path))

    def update_article_quality(self, title: str, quality: ArticleQuality) -> None:
        """Upsert quality metrics for a given title."""
        sql = SQL(
            """
        insert into ingestion_article(
            title, symbols_number, citations_number,
            has_problem_or_update_box, is_english_available
        )
        values (%s, %s, %s, %s, %s)
        on conflict (title) do update
        set symbols_number = excluded.symbols_number,
            citations_number = excluded.citations_number,
            has_problem_or_update_box = excluded.has_problem_or_update_box,
            is_english_available = excluded.is_english_available,
            updated_at = now();
        """
        )
        self._execute(
            sql,
            (
                title,
                int(quality.symbols_number),
                int(quality.citations_number),
                bool(quality.has_problem_or_update_box),
                bool(quality.is_english_available),
            ),
        )

    # ------------------------- Image incremental upserts -------------------------

    def upsert_image_url(self, url: str) -> None:
        """Create an image record as soon as we learn the URL."""
        sql = SQL(
            """
        insert into ingestion_image(url)
        values (%s)
        on conflict (url) do update
        set updated_at = now();
        """
        )
        self._execute(sql, (url,))

    def update_image_metadata(
        self,
        *,
        url: str,
        licence: str | None = None,
        local_path: str | None = None,
        filename: str | None = None,
        extension: str | None = None,
    ) -> None:
        """Upsert image metadata.

        Fields are merged using COALESCE, so passing None won't overwrite existing
        values in the database.

        Args:
            url: Image URL (primary key).
            licence: License short name (best-effort).
            local_path: Local path on disk.
            filename: Stored filename.
            extension: File extension.
        """
        sql = SQL(
            """
        insert into ingestion_image(url, licence, local_path, filename, extension)
        values (%s, %s, %s, %s, %s)
        on conflict (url) do update
        set licence = coalesce(excluded.licence, ingestion_image.licence),
            local_path = coalesce(excluded.local_path, ingestion_image.local_path),
            filename = coalesce(excluded.filename, ingestion_image.filename),
            extension = coalesce(excluded.extension, ingestion_image.extension),
            updated_at = now();
        """
        )
        self._execute(sql, (url, licence, local_path, filename, extension))

    def image_has_licence(self, url: str) -> bool:
        """Return True if we already persisted a licence value for this image URL."""
        if not self._enabled:
            return False

        sql = SQL(
            "select 1 from ingestion_image where url = %s and licence is not null and licence <> '' limit 1"
        )
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (url,))
                    return cur.fetchone() is not None
        except Exception:
            logger.exception("Failed to check existing image licence in PostgreSQL: %s", url)
            return False

    def list_articles_with_local_path(self) -> list[IngestionArticleRow]:
        """Return ingestion articles that have a local HTML file path.

        This is primarily used by WikipediaStorage.get_downloaded_articles to
        reconstruct scrapers from persisted HTML.
        """
        if not self._enabled:
            return []

        sql = SQL(
            """
        select title, url, local_path
        from ingestion_article
        where local_path is not null and local_path <> ''
        order by title asc
        """
        )

        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql)
                    rows = cur.fetchall() or []
                    result: list[IngestionArticleRow] = []
                    for r in rows:
                        local_path = (r.get("local_path") or "").strip()
                        title = (r.get("title") or "").strip()
                        if not title or not local_path:
                            continue
                        result.append(
                            IngestionArticleRow(
                                title=title,
                                url=(r.get("url") or None),
                                local_path=local_path,
                            )
                        )
                    return result
        except Exception:
            logger.exception("Failed to list ingestion articles from PostgreSQL")
            return []

    def get_article_quality(self, title: str) -> ArticleQuality | None:
        """Fetch quality metrics for an article title, if present."""
        if not self._enabled:
            return None

        sql = SQL(
            """
        select symbols_number, citations_number, has_problem_or_update_box, is_english_available
        from ingestion_article
        where title = %s
        limit 1
        """
        )

        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (title,))
                    row = cur.fetchone()
                    if not row:
                        return None

                    if row.get("symbols_number") is None or row.get("citations_number") is None:
                        return None

                    return ArticleQuality(
                        symbols_number=int(row["symbols_number"]),
                        citations_number=int(row["citations_number"]),
                        has_problem_or_update_box=bool(row.get("has_problem_or_update_box") or False),
                        is_english_available=bool(row.get("is_english_available") or False),
                    )
        except Exception:
            logger.exception("Failed to fetch article quality from PostgreSQL: %s", title)
            return None

    def get_image_by_url(self, url: str) -> IngestionImageRowWithMetadata | None:
        """Fetch an image record from `ingestion_image` by URL."""
        if not self._enabled:
            return None

        sql = SQL(
            """
        select url, licence, local_path, filename, extension
        from ingestion_image
        where url = %s
        limit 1
        """
        )

        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (url,))
                    row = cur.fetchone()
                    if not row:
                        return None

                    return IngestionImageRowWithMetadata(
                        url=(row.get("url") or url).strip(),
                        licence=((row.get("licence") or "").strip() or None),
                        local_path=((row.get("local_path") or "").strip() or None),
                        filename=((row.get("filename") or "").strip() or None),
                        extension=((row.get("extension") or "").strip() or None),
                    )
        except Exception:
            logger.exception("Failed to fetch image metadata from PostgreSQL: %s", url)
            return None

    def list_svg_images(self) -> list[IngestionImageRow]:
        """Return image rows that represent downloaded SVG files on disk."""
        if not self._enabled:
            return []

        sql = SQL(
            """
        select url, local_path, extension
        from ingestion_image
        where local_path is not null
          and local_path <> ''
          and (
            lower(coalesce(extension, '')) = '.svg'
            or lower(right(local_path, 4)) = '.svg'
          )
        order by url asc
        """
        )

        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql)
                    rows = cur.fetchall() or []
                    result: list[IngestionImageRow] = []
                    for r in rows:
                        url = (r.get("url") or "").strip()
                        local_path = (r.get("local_path") or "").strip()
                        if not url or not local_path:
                            continue
                        result.append(
                            IngestionImageRow(
                                url=url,
                                local_path=local_path,
                                extension=(r.get("extension") or None),
                            )
                        )
                    return result
        except Exception:
            logger.exception("Failed to list SVG images from PostgreSQL")
            return []


# Lazy singleton used by ingestion code.
_store: PgMetadataStore | None = None


def get_pg_metadata_store() -> PgMetadataStore:
    """Return a process-wide singleton PgMetadataStore.

    Ingestion code uses this to avoid passing the store through every call.
    """
    global _store
    if _store is None:
        _store = PgMetadataStore()
    return _store
