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
        nltk.download('punkt', quiet=True)
        logger.success("Successfully downloaded NLTK punkt tokenizer")
    except Exception as e:
        logger.error(f"Failed to download NLTK data: {e}")
        raise


if __name__ == "__main__":
    download_nltk_data()
