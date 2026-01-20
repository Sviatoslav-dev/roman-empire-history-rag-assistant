from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from data_ingestion.wikipedia_image import WikipediaImage
from logger import get_logger

logger = get_logger(__name__)


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

    images: List[WikipediaImage]

    # def _iter_files(self) -> Iterable[Path]:
    #     if not self.images_dir.exists():
    #         logger.warning("Images directory does not exist: %s", self.images_dir)
    #         return []
    #     return (p for p in self.images_dir.rglob("*") if p.is_file())

    def convert_svgs_to_png(self, *, remove_original: bool = False) -> int:
        """Convert all .svg files under images_dir to .png.

        Returns:
            Number of successfully converted SVG files.
        """
        try:
            import cairosvg  # type: ignore
        except Exception as e:
            logger.warning(
                "SVG conversion skipped (optional dependency missing). Install cairosvg to enable it. Error: %s",
                e,
            )
            return 0

        converted = 0
        for image in self.images:
            if image.local_path.suffix.lower() != ".svg":
                continue

            svg_path = image.local_path
            png_path = image.local_path.with_suffix(".png")
            try:
                # If already converted, skip (keeps idempotency and avoids rework)
                if png_path.exists() and png_path.stat().st_size > 0:
                    image.local_path = png_path
                    continue

                cairosvg.svg2png(url=str(image.local_path), write_to=str(png_path))

                if png_path.exists() and png_path.stat().st_size > 0:
                    image.local_path = png_path
                    converted += 1
                    if remove_original:
                        try:
                            svg_path.unlink()
                        except Exception as unlink_err:
                            logger.warning("Failed to remove original SVG %s: %s", image.local_path, unlink_err)
                else:
                    # If conversion produced nothing usable, clean up
                    try:
                        if png_path.exists():
                            png_path.unlink()
                    except Exception:
                        pass
            except Exception as conv_err:
                logger.warning("Failed to convert SVG %s to PNG: %s", image.local_path, conv_err)
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

    def run_all(self) -> None:
        """Run all preprocessing steps."""
        converted = self.convert_svgs_to_png(remove_original=False)
        # removed = self.remove_broken_raster_images()
        # logger.info(
        #     "Images preprocessing done. SVG->PNG converted: %d, broken raster removed: %d",
        #     converted,
        #     removed,
        # )

