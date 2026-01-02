"""
Query Processing Module

This module handles processing and embedding user queries for RAG retrieval.
Includes query validation, preprocessing, and embedding generation.
"""

from dataclasses import dataclass
from typing import List, Optional

from loguru import logger

from config import get_config
from offline.embedding import TextEmbedder


@dataclass
class ProcessedQuery:
    """Represents a processed and embedded query."""
    
    original_query: str
    processed_query: str
    embedding: List[float]
    dimensions: int
    model: str
    
    def __repr__(self) -> str:
        preview = self.processed_query[:100] + "..." if len(self.processed_query) > 100 else self.processed_query
        return (
            f"ProcessedQuery(query='{preview}', "
            f"dimensions={self.dimensions}, model='{self.model}')"
        )


class QueryProcessor:
    """Handles query preprocessing and embedding."""
    
    def __init__(self):
        """Initialize query processor with embedder."""
        self.config = get_config()
        self.embedder = TextEmbedder()
        
        logger.info(
            f"QueryProcessor initialized: model={self.embedder.model}, "
            f"dimensions={self.embedder.dimensions}"
        )
    
    def _validate_query(self, query: str) -> None:
        """
        Validate query input.
        
        Args:
            query: User query string
            
        Raises:
            ValueError: If query is invalid
        """
        if not query:
            raise ValueError("Query cannot be empty")
        
        if not isinstance(query, str):
            raise ValueError(f"Query must be a string, got {type(query)}")
        
        # Check minimum length (at least one word)
        if len(query.strip()) < 2:
            raise ValueError("Query is too short (minimum 2 characters)")
        
        # Check maximum length (reasonable upper bound)
        max_query_length = 1000  # characters
        if len(query) > max_query_length:
            logger.warning(
                f"Query length ({len(query)}) exceeds recommended maximum "
                f"({max_query_length}). Truncating."
            )
    
    def _preprocess_query(self, query: str) -> str:
        """
        Preprocess query for embedding.
        
        Args:
            query: Raw user query
            
        Returns:
            Preprocessed query string
        """
        # Strip leading/trailing whitespace
        processed = query.strip()
        
        # Replace multiple spaces with single space
        processed = " ".join(processed.split())
        
        # Truncate if too long (keep first 1000 characters)
        max_length = 1000
        if len(processed) > max_length:
            processed = processed[:max_length].strip()
            logger.warning(f"Query truncated to {max_length} characters")
        
        logger.debug(f"Preprocessed query: '{processed[:100]}...'")
        
        return processed
    
    def process_query(self, query: str) -> ProcessedQuery:
        """
        Process and embed a single query.
        
        Args:
            query: User query string
            
        Returns:
            ProcessedQuery object with embedding
            
        Raises:
            ValueError: If query is invalid
            Exception: If embedding fails
        """
        logger.info(f"Processing query: '{query[:100]}...'")
        
        # Validate
        self._validate_query(query)
        
        # Preprocess
        processed_query = self._preprocess_query(query)
        
        # Embed
        logger.debug("Generating query embedding...")
        embedding = self.embedder.embed_text(processed_query)
        
        # Create result
        result = ProcessedQuery(
            original_query=query,
            processed_query=processed_query,
            embedding=embedding,
            dimensions=len(embedding),
            model=self.embedder.model,
        )
        
        logger.success(
            f"Query processed successfully: {result.dimensions}D embedding generated"
        )
        
        return result
    
    def process_queries(self, queries: List[str]) -> List[ProcessedQuery]:
        """
        Process and embed multiple queries in batch.
        
        Args:
            queries: List of query strings
            
        Returns:
            List of ProcessedQuery objects
            
        Raises:
            ValueError: If any query is invalid
            Exception: If embedding fails
        """
        if not queries:
            logger.warning("Empty query list provided")
            return []
        
        logger.info(f"Processing {len(queries)} queries...")
        
        # Validate all queries first
        for i, query in enumerate(queries):
            try:
                self._validate_query(query)
            except ValueError as e:
                raise ValueError(f"Query {i} is invalid: {e}")
        
        # Preprocess
        processed_queries = [self._preprocess_query(q) for q in queries]
        
        # Batch embed
        logger.debug(f"Generating embeddings for {len(queries)} queries...")
        embeddings = self.embedder.embed_texts(processed_queries)
        
        # Create results
        results = []
        for original, processed, embedding in zip(queries, processed_queries, embeddings):
            result = ProcessedQuery(
                original_query=original,
                processed_query=processed,
                embedding=embedding,
                dimensions=len(embedding),
                model=self.embedder.model,
            )
            results.append(result)
        
        logger.success(f"Successfully processed {len(results)} queries")
        
        return results


def process_query(query: str) -> ProcessedQuery:
    """
    Convenience function to process and embed a single query.
    
    Args:
        query: User query string
        
    Returns:
        ProcessedQuery object with embedding
        
    Example:
        >>> processed = process_query("What is Santiago's Personal Legend?")
        >>> print(f"Embedding dimensions: {processed.dimensions}")
        Embedding dimensions: 3072
    """
    processor = QueryProcessor()
    return processor.process_query(query)


def process_queries(queries: List[str]) -> List[ProcessedQuery]:
    """
    Convenience function to process and embed multiple queries.
    
    Args:
        queries: List of query strings
        
    Returns:
        List of ProcessedQuery objects
        
    Example:
        >>> queries = ["What is alchemy?", "Who is Fatima?"]
        >>> processed = process_queries(queries)
        >>> print(f"Processed {len(processed)} queries")
        Processed 2 queries
    """
    processor = QueryProcessor()
    return processor.process_queries(queries)
