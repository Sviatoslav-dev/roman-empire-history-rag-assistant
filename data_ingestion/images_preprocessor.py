from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from data_ingestion.pg_metadata_store import get_pg_metadata_store, IngestionImageRow
from logger import get_logger
import cairosvg

logger = get_logger(__name__)
load_dotenv()

_postgres = get_pg_metadata_store()


@dataclass(frozen=True)
class ImagesPreprocessor:
    """Post-processes downloaded images."""

    def _safe_unlink(self, path: Path) -> None:
        try:
            path.unlink()
        except Exception as exc:
            logger.warning("Failed to remove file %s: %s", path, exc)

    def _iter_svg_rows(self) -> list[IngestionImageRow]:
        return list(_postgres.list_svg_images())

    def _convert_one_svg_row(self, row: IngestionImageRow, *, remove_original: bool) -> bool:
        svg_path = Path(row.local_path)
        if not svg_path.exists() or not svg_path.is_file():
            logger.warning("SVG file missing on disk (url=%s): %s", row.url, svg_path)
            return False

        png_path = svg_path.with_suffix(".png")

        try:
            # Idempotency: if already converted, just update DB and go on.
            if png_path.exists() and png_path.stat().st_size > 0:
                try:
                    _postgres.update_image_metadata(url=row.url, local_path=str(png_path), extension=".png")
                except Exception:
                    logger.exception("Failed to update DB for already-converted png (url=%s)", row.url)
                return False

            cairosvg.svg2png(url=str(svg_path), write_to=str(png_path))

            if not (png_path.exists() and png_path.stat().st_size > 0):
                # If conversion produced nothing usable, clean up
                if png_path.exists():
                    self._safe_unlink(png_path)
                return False

            try:
                _postgres.update_image_metadata(url=row.url, local_path=str(png_path), extension=".png")
            except Exception:
                logger.exception("Failed to update DB after svg->png conversion (url=%s)", row.url)

            if remove_original:
                self._safe_unlink(svg_path)

            return True
        except Exception as conv_err:
            logger.warning("Failed to convert SVG %s to PNG: %s", svg_path, conv_err)
            if png_path.exists():
                try:
                    self._safe_unlink(png_path)
                except Exception:
                    pass
            return False

    def convert_svgs_to_png(self, *, remove_original: bool = False) -> int:
        """Convert downloaded SVG images (tracked in Postgres) to PNG.

        Algorithm:
        - Query `ingestion_image` for rows that represent SVGs (via PgMetadataStore).
        - For each row: read SVG from `local_path`, convert to `.png` next to it.
        - Update DB record with new `local_path` and `extension`.

        Returns:
            Number of successfully converted SVG files.
        """
        converted = 0
        for row in self._iter_svg_rows():
            if self._convert_one_svg_row(row, remove_original=remove_original):
                converted += 1
        return converted
