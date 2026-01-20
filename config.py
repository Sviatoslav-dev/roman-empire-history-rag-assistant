from pathlib import Path
import os

from dotenv import load_dotenv

# Load environment variables from a .env file (if present).
# Keep this module minimal to avoid import cycles when other modules import ARTICLES_DIR.
load_dotenv()

# Path to the directory where article HTML files are stored.
ARTICLES_DIR = Path(os.getenv("ARTICLES_DIR", "data_ingestion/data/articles"))

__all__ = ["ARTICLES_DIR"]

