from __future__ import annotations

from dataclasses import dataclass
from typing import List

from dotenv import load_dotenv

from data_ingestion.pg_metadata_store import get_pg_metadata_store
from data_ingestion.wikipedia_image import WikipediaImage
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


    # def _iter_files(self) -> Iterable[Path]:
    #     if not self.images_dir.exists():
    #         logger.warning("Images directory does not exist: %s", self.images_dir)
    #         return []
    #     return (p for p in self.images_dir.rglob("*") if p.is_file())

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

    # def remove_broken_raster_images(self) -> int:
    #     """Detect and remove broken raster images via Pillow.
    #
    #     Returns:
    #         Number of removed files.
    #     """
    #     try:
    #         from PIL import Image  # type: ignore
    #     except Exception as e:
    #         logger.warning(
    #             "Raster validation skipped (Pillow not installed). Install pillow to enable it. Error: %s",
    #             e,
    #         )
    #         return 0
    #
    #     removed = 0
    #     raster_exts = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".tif", ".tiff"}
    #
    #     result = []
    #     for image in self.images:
    #         if image.local_path.suffix.lower() not in raster_exts:
    #             continue
    #
    #         try:
    #             # verify() is fast and catches truncation/corruption, but doesn't decode pixels.
    #             with Image.open(image.local_path) as img:
    #                 img.verify()
    #
    #             # Extra safety for some edge cases: reopen and force basic load.
    #             with Image.open(image.local_path) as img:
    #                 img.load()
    #
    #         except Exception as e:
    #             logger.warning("Broken image detected, removing %s: %s", p, e)
    #             try:
    #                 image.local_path.unlink()
    #                 removed += 1
    #             except Exception as unlink_err:
    #                 logger.error("Failed to remove broken image %s: %s", p, unlink_err)
    #
    #     return removed
