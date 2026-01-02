"""
Post-installation setup script for RAG prototype.
Downloads required NLTK data files.
"""

import nltk
from loguru import logger


def download_nltk_data():
    """Download required NLTK data packages."""
    logger.info("Downloading NLTK punkt tokenizer...")
    try:
        # NLTK 3.8+ uses punkt_tab instead of punkt
        nltk.download('punkt_tab', quiet=True)
        logger.success("✅ NLTK punkt_tab tokenizer downloaded successfully")
        logger.info(f"Data location: {nltk.data.path}")
    except Exception as e:
        logger.error(f"Failed to download NLTK data: {e}")
        raise


if __name__ == "__main__":
    download_nltk_data()
