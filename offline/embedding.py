"""
OpenAI Embedding Module

This module handles converting text chunks into vector embeddings using OpenAI's API.
Uses text-embedding-3-large model with 3072 dimensions for highest quality.
Includes retry logic and batch processing for efficiency.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from loguru import logger
from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential_jitter,
    retry_if_exception_type,
)

from config import get_config
from offline.chunking import Chunk


@dataclass
class EmbeddedChunk:
    """A chunk with its vector embedding."""
    
    chunk: Chunk
    embedding: List[float]
    model: str
    dimensions: int
    
    def __post_init__(self):
        """Validate embedding dimensions."""
        if len(self.embedding) != self.dimensions:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.dimensions}, "
                f"got {len(self.embedding)}"
            )


class TextEmbedder:
    """Handles embedding text chunks using OpenAI's API."""
    
    def __init__(self):
        """Initialize the embedder with OpenAI client and configuration."""
        self.config = get_config()
        self.client = OpenAI(api_key=self.config.openai_api_key)
        
        # Use embedding model and dimensions from configuration
        self.model = self.config.openai_embedding_model
        self.dimensions = self.config.openai_embedding_dimensions
        
        # OpenAI allows up to 2048 texts per batch for embeddings
        self.batch_size = 100  # Conservative batch size for safety
        
        logger.info(
            f"TextEmbedder initialized: model={self.model}, "
            f"dimensions={self.dimensions}, batch_size={self.batch_size}"
        )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=1, max=10, jitter=2),
        retry=retry_if_exception_type((Exception,)),
        reraise=True,
    )
    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a batch of texts using OpenAI API with retry logic.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors (each is a list of floats)
            
        Raises:
            Exception: If API call fails after retries
        """
        if not texts:
            return []
        
        logger.debug(f"Embedding batch of {len(texts)} texts")
        
        try:
            response = self.client.embeddings.create(
                input=texts,
                model=self.model,
                dimensions=self.dimensions,
            )
            
            # Extract embeddings in correct order
            embeddings = [item.embedding for item in response.data]
            
            logger.debug(f"Successfully embedded {len(embeddings)} texts")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error embedding texts: {e}")
            raise
    
    def embed_chunks(
        self,
        chunks: List[Chunk],
        show_progress: bool = True,
    ) -> List[EmbeddedChunk]:
        """
        Embed a list of chunks, processing in batches.
        
        Args:
            chunks: List of Chunk objects to embed
            show_progress: Whether to log progress updates
            
        Returns:
            List of EmbeddedChunk objects with embeddings
            
        Raises:
            Exception: If embedding fails after retries
        """
        if not chunks:
            logger.warning("No chunks provided for embedding")
            return []
        
        logger.info(f"Starting embedding process for {len(chunks)} chunks")
        
        embedded_chunks: List[EmbeddedChunk] = []
        total_batches = (len(chunks) + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(0, len(chunks), self.batch_size):
            batch_end = min(batch_idx + self.batch_size, len(chunks))
            batch_chunks = chunks[batch_idx:batch_end]
            batch_num = batch_idx // self.batch_size + 1
            
            if show_progress:
                logger.info(
                    f"Processing batch {batch_num}/{total_batches} "
                    f"({len(batch_chunks)} chunks)"
                )
            
            # Extract text from chunks
            texts = [chunk.text for chunk in batch_chunks]
            
            # Get embeddings for this batch
            embeddings = self._embed_texts(texts)
            
            # Create EmbeddedChunk objects
            for chunk, embedding in zip(batch_chunks, embeddings):
                embedded_chunk = EmbeddedChunk(
                    chunk=chunk,
                    embedding=embedding,
                    model=self.model,
                    dimensions=self.dimensions,
                )
                embedded_chunks.append(embedded_chunk)
        
        logger.success(
            f"Successfully embedded all {len(embedded_chunks)} chunks"
        )
        
        return embedded_chunks
    
    def embed_file(
        self,
        file_path: Path,
        show_progress: bool = True,
    ) -> List[EmbeddedChunk]:
        """
        Convenience method to chunk and embed a file in one step.
        
        Args:
            file_path: Path to the text file to process
            show_progress: Whether to log progress updates
            
        Returns:
            List of EmbeddedChunk objects
            
        Raises:
            Exception: If chunking or embedding fails
        """
        from offline.chunking import chunk_document
        
        logger.info(f"Processing file: {file_path}")
        
        # First, chunk the document
        chunks = chunk_document(file_path)
        
        # Then, embed the chunks
        embedded_chunks = self.embed_chunks(chunks, show_progress=show_progress)
        
        return embedded_chunks


def embed_chunks(chunks: List[Chunk]) -> List[EmbeddedChunk]:
    """
    Convenience function to embed a list of chunks.
    
    Args:
        chunks: List of Chunk objects to embed
        
    Returns:
        List of EmbeddedChunk objects with embeddings
    """
    embedder = TextEmbedder()
    return embedder.embed_chunks(chunks)


def embed_document(file_path: Path) -> List[EmbeddedChunk]:
    """
    Convenience function to chunk and embed a document file.
    
    Args:
        file_path: Path to the text file to process
        
    Returns:
        List of EmbeddedChunk objects
    """
    embedder = TextEmbedder()
    return embedder.embed_file(file_path)
