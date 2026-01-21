from __future__ import annotations

from dataclasses import dataclass

from dotenv import load_dotenv

from data_ingestion.pg_metadata_store import get_pg_metadata_store
from logger import get_logger
import cairosvg

logger = get_logger(__name__)
load_dotenv()

_meta = get_pg_metadata_store()


@dataclass(frozen=True)
class ImagesPreprocessor:
    """Post-processes downloaded images.

    Responsibilities (kept intentionally small and non-invasive):
    - Convert SVG files to PNG (optionally remove the original SVG).
    - Validate raster images (JPG/JPEG/PNG/GIF/WEBP/TIFF) and remove broken files.

    Notes:
    - SVG conversion requires the optional dependency `cairosvg`.
    - Raster validation uses Pillow (`PIL`).
    """

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
        for row in _meta.list_svg_images():
            from pathlib import Path

            svg_path = Path(row.local_path)
            if not svg_path.exists() or not svg_path.is_file():
                logger.warning("SVG file missing on disk (url=%s): %s", row.url, svg_path)
                continue

            png_path = svg_path.with_suffix(".png")

            try:
                # Idempotency: if already converted, just update DB and go on.
                if png_path.exists() and png_path.stat().st_size > 0:
                    try:
                        _meta.update_image_metadata(url=row.url, local_path=str(png_path), extension=".png")
                    except Exception:
                        logger.exception("Failed to update DB for already-converted png (url=%s)", row.url)
                    continue

                cairosvg.svg2png(url=str(svg_path), write_to=str(png_path))

                if png_path.exists() and png_path.stat().st_size > 0:
                    converted += 1
                    try:
                        _meta.update_image_metadata(url=row.url, local_path=str(png_path), extension=".png")
                    except Exception:
                        logger.exception("Failed to update DB after svg->png conversion (url=%s)", row.url)

                    if remove_original:
                        try:
                            svg_path.unlink()
                        except Exception as unlink_err:
                            logger.warning("Failed to remove original SVG %s: %s", svg_path, unlink_err)
                else:
                    # If conversion produced nothing usable, clean up
                    try:
                        if png_path.exists():
                            png_path.unlink()
                    except Exception:
                        pass
            except Exception as conv_err:
                logger.warning("Failed to convert SVG %s to PNG: %s", svg_path, conv_err)
                try:
                    if png_path.exists():
                        png_path.unlink()
                except Exception:
                    pass

        return converted
