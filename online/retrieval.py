"""
Vector retrieval module for querying Pinecone and retrieving relevant chunks.

This module uses query embeddings to search the Pinecone vector database
and return the most relevant document chunks with their similarity scores.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from loguru import logger
from pinecone import Pinecone
from tenacity import retry, stop_after_attempt, wait_exponential

from config import get_config


@dataclass
class RetrievalResult:
    """
    Represents a single retrieved chunk with its metadata and relevance score.
    
    Attributes:
        chunk_id: Unique identifier for the chunk
        text: The actual text content of the chunk
        score: Similarity score from Pinecone (higher = more relevant)
        metadata: Additional metadata (source file, position, etc.)
    """
    chunk_id: str
    text: str
    score: float
    metadata: Dict[str, Any]
    
    def __repr__(self) -> str:
        """Short representation showing ID and score."""
        text_preview = self.text[:100] + "..." if len(self.text) > 100 else self.text
        return (
            f"RetrievalResult(id={self.chunk_id}, score={self.score:.4f}, "
            f"text='{text_preview}')"
        )


class VectorRetriever:
    """
    Handles vector similarity search in Pinecone.
    
    Uses query embeddings to find the most relevant document chunks
    stored in the Pinecone index.
    """
    
    def __init__(self):
        """Initialize Pinecone connection and load configuration."""
        self.config = get_config()
        
        # Initialize Pinecone client
        self.pc = Pinecone(api_key=self.config.pinecone_api_key)
        self.index = self.pc.Index(self.config.pinecone_index_name)
        
        # Retrieval parameters from config
        self.top_k = self.config.top_k
        self.min_score = self.config.retrieval_min_score
        
        logger.info(
            f"VectorRetriever initialized with index '{self.config.pinecone_index_name}', "
            f"top_k={self.top_k}, min_score={self.min_score}"
        )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _query_pinecone(
        self,
        embedding: List[float],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """
        Query Pinecone index with retry logic.
        
        Args:
            embedding: Query embedding vector
            top_k: Number of results to return
            
        Returns:
            List of matches from Pinecone
            
        Raises:
            Exception: If query fails after retries
        """
        logger.debug(f"Querying Pinecone with embedding of {len(embedding)} dimensions")
        
        response = self.index.query(
            vector=embedding,
            top_k=top_k,
            include_metadata=True,
        )
        
        matches = response.get('matches', [])
        logger.debug(f"Pinecone returned {len(matches)} matches")
        
        return matches
    
    def _filter_by_score(
        self,
        matches: List[Dict[str, Any]],
        min_score: float,
    ) -> List[Dict[str, Any]]:
        """
        Filter matches by minimum similarity score.
        
        Args:
            matches: Raw matches from Pinecone
            min_score: Minimum score threshold
            
        Returns:
            Filtered list of matches
        """
        filtered = [m for m in matches if m.get('score', 0) >= min_score]
        
        if len(filtered) < len(matches):
            logger.debug(
                f"Filtered {len(matches) - len(filtered)} matches "
                f"below threshold {min_score}"
            )
        
        return filtered
    
    def _parse_matches(
        self,
        matches: List[Dict[str, Any]],
    ) -> List[RetrievalResult]:
        """
        Parse Pinecone matches into RetrievalResult objects.
        
        Args:
            matches: List of match dictionaries from Pinecone
            
        Returns:
            List of RetrievalResult objects
        """
        results = []
        
        for match in matches:
            chunk_id = match.get('id', 'unknown')
            score = match.get('score', 0.0)
            metadata = match.get('metadata', {})
            
            # Extract text from metadata
            text = metadata.get('text', '')
            
            if not text:
                logger.warning(f"Chunk {chunk_id} has no text in metadata")
                continue
            
            result = RetrievalResult(
                chunk_id=chunk_id,
                text=text,
                score=score,
                metadata=metadata,
            )
            results.append(result)
        
        return results
    
    def retrieve(
        self,
        query_embedding: List[float],
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant chunks using query embedding.
        
        Args:
            query_embedding: The embedded query vector
            top_k: Number of results to return (uses config default if None)
            min_score: Minimum similarity score (uses config default if None)
            
        Returns:
            List of RetrievalResult objects, sorted by relevance (highest first)
            
        Raises:
            ValueError: If query_embedding is invalid
            Exception: If Pinecone query fails after retries
            
        Example:
            >>> retriever = VectorRetriever()
            >>> embedding = [0.1, 0.2, ...]  # 3072-dim vector
            >>> results = retriever.retrieve(embedding)
            >>> for result in results:
            ...     print(f"Score: {result.score}, Text: {result.text[:50]}")
        """
        # Validate embedding
        if not query_embedding:
            raise ValueError("Query embedding cannot be empty")
        
        if len(query_embedding) != 3072:
            raise ValueError(
                f"Expected embedding dimension 3072, got {len(query_embedding)}"
            )
        
        # Use config defaults if not specified
        k = top_k if top_k is not None else self.top_k
        threshold = min_score if min_score is not None else self.min_score
        
        logger.info(f"Retrieving top {k} chunks with min_score {threshold}")
        
        # Query Pinecone
        matches = self._query_pinecone(query_embedding, k)
        
        if not matches:
            logger.warning("No matches found in Pinecone")
            return []
        
        # Filter by score
        filtered_matches = self._filter_by_score(matches, threshold)
        
        if not filtered_matches:
            logger.warning(
                f"All {len(matches)} matches filtered out by threshold {threshold}"
            )
            return []
        
        # Parse into RetrievalResult objects
        results = self._parse_matches(filtered_matches)
        
        logger.success(
            f"Retrieved {len(results)} chunks "
            f"(scores: {results[0].score:.4f} to {results[-1].score:.4f})"
        )
        
        return results


# Convenience functions for quick usage

def retrieve(
    query_embedding: List[float],
    top_k: Optional[int] = None,
    min_score: Optional[float] = None,
) -> List[RetrievalResult]:
    """
    Convenience function to retrieve chunks without explicitly creating a retriever.
    
    Args:
        query_embedding: The embedded query vector
        top_k: Number of results to return
        min_score: Minimum similarity score threshold
        
    Returns:
        List of RetrievalResult objects
        
    Example:
        >>> from online.retrieval import retrieve
        >>> embedding = [0.1, 0.2, ...]  # from query processor
        >>> results = retrieve(embedding, top_k=5)
    """
    retriever = VectorRetriever()
    return retriever.retrieve(query_embedding, top_k, min_score)
