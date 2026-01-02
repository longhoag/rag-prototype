"""
Pinecone Indexing Module

This module handles uploading vector embeddings to Pinecone for similarity search.
Includes batch processing, retry logic, and metadata management.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any

from loguru import logger
from pinecone import Pinecone, ServerlessSpec
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential_jitter,
    retry_if_exception_type,
)

from config import get_config
from offline.embedding import EmbeddedChunk


@dataclass
class IndexedChunk:
    """Represents a chunk that has been indexed in Pinecone."""
    
    chunk_id: str
    vector_id: str
    text_preview: str  # First 100 chars for logging
    token_count: int
    
    @classmethod
    def from_embedded_chunk(cls, embedded_chunk: EmbeddedChunk, vector_id: str) -> "IndexedChunk":
        """Create an IndexedChunk from an EmbeddedChunk."""
        chunk = embedded_chunk.chunk
        text_preview = chunk.text[:100] + "..." if len(chunk.text) > 100 else chunk.text
        
        return cls(
            chunk_id=str(chunk.chunk_id),
            vector_id=vector_id,
            text_preview=text_preview,
            token_count=chunk.token_count,
        )


class PineconeIndexer:
    """Handles uploading embeddings to Pinecone vector database."""
    
    def __init__(self):
        """Initialize Pinecone client and index connection."""
        self.config = get_config()
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=self.config.pinecone_api_key)
        self.index_name = self.config.pinecone_index_name
        
        # Batch size for uploads (Pinecone can handle large batches)
        self.batch_size = 100
        
        logger.info(f"PineconeIndexer initialized: index_name={self.index_name}")
    
    def _ensure_index_exists(self, dimensions: int) -> None:
        """
        Ensure the Pinecone index exists, create if it doesn't.
        
        Args:
            dimensions: Dimension of the vectors to be stored
        """
        existing_indexes = [index.name for index in self.pc.list_indexes()]
        
        if self.index_name not in existing_indexes:
            logger.info(f"Creating new Pinecone index: {self.index_name}")
            
            self.pc.create_index(
                name=self.index_name,
                dimension=dimensions,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region=self.config.pinecone_environment,
                ),
            )
            
            logger.success(f"Created index: {self.index_name}")
        else:
            logger.info(f"Index already exists: {self.index_name}")
    
    def _get_index(self):
        """Get reference to the Pinecone index."""
        return self.pc.Index(self.index_name)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=1, max=10, jitter=2),
        retry=retry_if_exception_type((Exception,)),
        reraise=True,
    )
    def _upsert_vectors(
        self,
        vectors: List[tuple],
    ) -> Dict[str, Any]:
        """
        Upload a batch of vectors to Pinecone with retry logic.
        
        Args:
            vectors: List of (id, embedding, metadata) tuples
            
        Returns:
            Response from Pinecone upsert operation
            
        Raises:
            Exception: If upload fails after retries
        """
        index = self._get_index()
        
        try:
            response = index.upsert(vectors=vectors)
            logger.debug(f"Successfully upserted {len(vectors)} vectors")
            return response
            
        except Exception as e:
            logger.error(f"Error upserting vectors: {e}")
            raise
    
    def index_embedded_chunks(
        self,
        embedded_chunks: List[EmbeddedChunk],
        document_source: Optional[str] = None,
        show_progress: bool = True,
    ) -> List[IndexedChunk]:
        """
        Upload embedded chunks to Pinecone with metadata.
        
        Args:
            embedded_chunks: List of EmbeddedChunk objects to index
            document_source: Optional source identifier for the document
            show_progress: Whether to log progress updates
            
        Returns:
            List of IndexedChunk objects representing indexed chunks
            
        Raises:
            Exception: If indexing fails after retries
        """
        if not embedded_chunks:
            logger.warning("No embedded chunks provided for indexing")
            return []
        
        # Ensure index exists with correct dimensions
        dimensions = embedded_chunks[0].dimensions
        self._ensure_index_exists(dimensions)
        
        logger.info(
            f"Starting indexing process for {len(embedded_chunks)} embedded chunks"
        )
        
        indexed_chunks: List[IndexedChunk] = []
        total_batches = (len(embedded_chunks) + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(0, len(embedded_chunks), self.batch_size):
            batch_end = min(batch_idx + self.batch_size, len(embedded_chunks))
            batch = embedded_chunks[batch_idx:batch_end]
            batch_num = batch_idx // self.batch_size + 1
            
            if show_progress:
                logger.info(
                    f"Uploading batch {batch_num}/{total_batches} "
                    f"({len(batch)} vectors)"
                )
            
            # Prepare vectors for Pinecone
            vectors = []
            for embedded_chunk in batch:
                chunk = embedded_chunk.chunk
                
                # Create unique vector ID (using chunk_id as string)
                vector_id = str(chunk.chunk_id)
                
                # Prepare metadata
                metadata = {
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "token_count": chunk.token_count,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                    "model": embedded_chunk.model,
                }
                
                if document_source:
                    metadata["source"] = document_source
                
                # Pinecone format: (id, embedding, metadata)
                vectors.append((vector_id, embedded_chunk.embedding, metadata))
                
                # Track indexed chunk
                indexed_chunk = IndexedChunk.from_embedded_chunk(
                    embedded_chunk, vector_id
                )
                indexed_chunks.append(indexed_chunk)
            
            # Upload batch to Pinecone
            self._upsert_vectors(vectors)
        
        logger.success(
            f"Successfully indexed all {len(indexed_chunks)} chunks to Pinecone"
        )
        
        return indexed_chunks
    
    def index_document(
        self,
        file_path: Path,
        show_progress: bool = True,
    ) -> List[IndexedChunk]:
        """
        Convenience method to chunk, embed, and index a document file.
        
        Args:
            file_path: Path to the text file to process
            show_progress: Whether to log progress updates
            
        Returns:
            List of IndexedChunk objects
            
        Raises:
            Exception: If any step fails
        """
        from offline.embedding import embed_document
        
        logger.info(f"Processing and indexing document: {file_path}")
        
        # Chunk and embed the document
        embedded_chunks = embed_document(file_path)
        
        # Index the embeddings
        document_source = str(file_path.name)
        indexed_chunks = self.index_embedded_chunks(
            embedded_chunks,
            document_source=document_source,
            show_progress=show_progress,
        )
        
        return indexed_chunks
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the Pinecone index.
        
        Returns:
            Dictionary with index statistics
        """
        index = self._get_index()
        stats = index.describe_index_stats()
        
        logger.info(f"Index stats: {stats}")
        return stats


def index_embedded_chunks(
    embedded_chunks: List[EmbeddedChunk],
    document_source: Optional[str] = None,
) -> List[IndexedChunk]:
    """
    Convenience function to index a list of embedded chunks.
    
    Args:
        embedded_chunks: List of EmbeddedChunk objects to index
        document_source: Optional source identifier for the document
        
    Returns:
        List of IndexedChunk objects
    """
    indexer = PineconeIndexer()
    return indexer.index_embedded_chunks(embedded_chunks, document_source)


def index_document(file_path: Path) -> List[IndexedChunk]:
    """
    Convenience function to chunk, embed, and index a document file.
    
    Args:
        file_path: Path to the text file to process
        
    Returns:
        List of IndexedChunk objects
    """
    indexer = PineconeIndexer()
    return indexer.index_document(file_path)
