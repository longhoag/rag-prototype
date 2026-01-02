"""
Tests for the query processing module.
"""

import pytest
from unittest.mock import Mock, patch

from online.query import (
    ProcessedQuery,
    QueryProcessor,
    process_query,
    process_queries,
)


class TestProcessedQuery:
    """Tests for ProcessedQuery dataclass."""
    
    def test_processed_query_creation(self):
        """Test creating a ProcessedQuery object."""
        query = ProcessedQuery(
            original_query="What is alchemy?",
            processed_query="What is alchemy?",
            embedding=[0.1] * 3072,
            dimensions=3072,
            model="text-embedding-3-large",
        )
        
        assert query.original_query == "What is alchemy?"
        assert query.processed_query == "What is alchemy?"
        assert len(query.embedding) == 3072
        assert query.dimensions == 3072
        assert query.model == "text-embedding-3-large"
    
    def test_processed_query_repr_short(self):
        """Test __repr__ for short query."""
        query = ProcessedQuery(
            original_query="Short query",
            processed_query="Short query",
            embedding=[0.1] * 100,
            dimensions=100,
            model="test-model",
        )
        
        repr_str = repr(query)
        assert "Short query" in repr_str
        assert "dimensions=100" in repr_str
        assert "test-model" in repr_str
    
    def test_processed_query_repr_long(self):
        """Test __repr__ for long query (should be truncated)."""
        long_query = "a" * 150
        query = ProcessedQuery(
            original_query=long_query,
            processed_query=long_query,
            embedding=[0.1] * 100,
            dimensions=100,
            model="test-model",
        )
        
        repr_str = repr(query)
        assert "..." in repr_str  # Should be truncated


class TestQueryProcessor:
    """Tests for QueryProcessor class."""
    
    def test_initialization(self):
        """Test QueryProcessor initialization."""
        processor = QueryProcessor()
        
        assert processor.config is not None
        assert processor.embedder is not None
        assert processor.embedder.model == "text-embedding-3-large"
        assert processor.embedder.dimensions == 3072
    
    def test_validate_query_valid(self):
        """Test validation with valid queries."""
        processor = QueryProcessor()
        
        # These should not raise exceptions
        processor._validate_query("What is alchemy?")
        processor._validate_query("Short")
        processor._validate_query("a" * 1000)  # Max length
    
    def test_validate_query_empty(self):
        """Test validation with empty query."""
        processor = QueryProcessor()
        
        with pytest.raises(ValueError, match="Query cannot be empty"):
            processor._validate_query("")
        
        with pytest.raises(ValueError, match="Query cannot be empty"):
            processor._validate_query(None)
    
    def test_validate_query_wrong_type(self):
        """Test validation with wrong type."""
        processor = QueryProcessor()
        
        with pytest.raises(ValueError, match="Query must be a string"):
            processor._validate_query(123)
        
        with pytest.raises(ValueError, match="Query must be a string"):
            processor._validate_query(['list', 'query'])
    
    def test_validate_query_too_short(self):
        """Test validation with too short query."""
        processor = QueryProcessor()
        
        with pytest.raises(ValueError, match="Query is too short"):
            processor._validate_query("a")
        
        with pytest.raises(ValueError, match="Query is too short"):
            processor._validate_query(" ")
    
    def test_validate_query_too_long(self):
        """Test validation with very long query (should warn but not fail)."""
        processor = QueryProcessor()
        
        long_query = "a" * 1500
        # Should not raise exception, just warn
        processor._validate_query(long_query)
    
    def test_preprocess_query_basic(self):
        """Test basic query preprocessing."""
        processor = QueryProcessor()
        
        # Test whitespace stripping
        assert processor._preprocess_query("  query  ") == "query"
        
        # Test multiple spaces
        assert processor._preprocess_query("query   with    spaces") == "query with spaces"
        
        # Test combination
        assert processor._preprocess_query("  multiple   spaces  ") == "multiple spaces"
    
    def test_preprocess_query_truncation(self):
        """Test query truncation for very long queries."""
        processor = QueryProcessor()
        
        long_query = "a" * 1500
        processed = processor._preprocess_query(long_query)
        
        assert len(processed) == 1000
        assert processed == "a" * 1000
    
    def test_preprocess_query_newlines(self):
        """Test preprocessing with newlines and tabs."""
        processor = QueryProcessor()
        
        query = "query\nwith\nnewlines\tand\ttabs"
        processed = processor._preprocess_query(query)
        
        # Should normalize to single spaces
        assert processed == "query with newlines and tabs"
    
    @patch('online.query.TextEmbedder')
    def test_process_query_success(self, mock_embedder_class):
        """Test successful single query processing."""
        # Setup mock
        mock_embedder = Mock()
        mock_embedder.model = "text-embedding-3-large"
        mock_embedder.dimensions = 3072
        mock_embedder.embed_text.return_value = [0.1] * 3072
        mock_embedder_class.return_value = mock_embedder
        
        processor = QueryProcessor()
        result = processor.process_query("What is Santiago's Personal Legend?")
        
        assert isinstance(result, ProcessedQuery)
        assert result.original_query == "What is Santiago's Personal Legend?"
        assert result.processed_query == "What is Santiago's Personal Legend?"
        assert len(result.embedding) == 3072
        assert result.dimensions == 3072
        assert result.model == "text-embedding-3-large"
        
        # Verify embedder was called
        mock_embedder.embed_text.assert_called_once_with("What is Santiago's Personal Legend?")
    
    @patch('online.query.TextEmbedder')
    def test_process_query_with_preprocessing(self, mock_embedder_class):
        """Test query processing with preprocessing."""
        # Setup mock
        mock_embedder = Mock()
        mock_embedder.model = "test-model"
        mock_embedder.dimensions = 100
        mock_embedder.embed_text.return_value = [0.1] * 100
        mock_embedder_class.return_value = mock_embedder
        
        processor = QueryProcessor()
        result = processor.process_query("  query   with    spaces  ")
        
        assert result.original_query == "  query   with    spaces  "
        assert result.processed_query == "query with spaces"
        
        # Verify embedder was called with preprocessed query
        mock_embedder.embed_text.assert_called_once_with("query with spaces")
    
    def test_process_query_invalid(self):
        """Test query processing with invalid input."""
        processor = QueryProcessor()
        
        with pytest.raises(ValueError, match="Query cannot be empty"):
            processor.process_query("")
        
        with pytest.raises(ValueError, match="Query is too short"):
            processor.process_query("x")
    
    @patch('online.query.TextEmbedder')
    def test_process_queries_batch(self, mock_embedder_class):
        """Test batch query processing."""
        # Setup mock
        mock_embedder = Mock()
        mock_embedder.model = "text-embedding-3-large"
        mock_embedder.dimensions = 3072
        mock_embedder.embed_texts.return_value = [
            [0.1] * 3072,
            [0.2] * 3072,
            [0.3] * 3072,
        ]
        mock_embedder_class.return_value = mock_embedder
        
        processor = QueryProcessor()
        queries = [
            "What is alchemy?",
            "Who is Fatima?",
            "Where are the Pyramids?",
        ]
        results = processor.process_queries(queries)
        
        assert len(results) == 3
        assert all(isinstance(r, ProcessedQuery) for r in results)
        assert results[0].original_query == "What is alchemy?"
        assert results[1].original_query == "Who is Fatima?"
        assert results[2].original_query == "Where are the Pyramids?"
        
        # Verify batch embedder was called
        mock_embedder.embed_texts.assert_called_once()
        call_args = mock_embedder.embed_texts.call_args[0][0]
        assert call_args == queries
    
    @patch('online.query.TextEmbedder')
    def test_process_queries_empty_list(self, mock_embedder_class):
        """Test batch processing with empty list."""
        mock_embedder = Mock()
        mock_embedder_class.return_value = mock_embedder
        
        processor = QueryProcessor()
        results = processor.process_queries([])
        
        assert results == []
        # Should not call embedder
        mock_embedder.embed_texts.assert_not_called()
    
    def test_process_queries_invalid_query(self):
        """Test batch processing with invalid query in list."""
        processor = QueryProcessor()
        
        queries = [
            "Valid query",
            "",  # Invalid - empty
            "Another valid query",
        ]
        
        with pytest.raises(ValueError, match="Query 1 is invalid"):
            processor.process_queries(queries)
    
    @patch('online.query.TextEmbedder')
    def test_process_queries_with_preprocessing(self, mock_embedder_class):
        """Test batch processing with preprocessing."""
        # Setup mock
        mock_embedder = Mock()
        mock_embedder.model = "test-model"
        mock_embedder.dimensions = 100
        mock_embedder.embed_texts.return_value = [
            [0.1] * 100,
            [0.2] * 100,
        ]
        mock_embedder_class.return_value = mock_embedder
        
        processor = QueryProcessor()
        queries = [
            "  query   one  ",
            "query   two  ",
        ]
        results = processor.process_queries(queries)
        
        assert results[0].original_query == "  query   one  "
        assert results[0].processed_query == "query one"
        assert results[1].original_query == "query   two  "
        assert results[1].processed_query == "query two"
        
        # Verify preprocessed queries were passed to embedder
        call_args = mock_embedder.embed_texts.call_args[0][0]
        assert call_args == ["query one", "query two"]


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    @patch('online.query.TextEmbedder')
    def test_process_query_function(self, mock_embedder_class):
        """Test process_query convenience function."""
        # Setup mock
        mock_embedder = Mock()
        mock_embedder.model = "text-embedding-3-large"
        mock_embedder.dimensions = 3072
        mock_embedder.embed_text.return_value = [0.1] * 3072
        mock_embedder_class.return_value = mock_embedder
        
        result = process_query("Test query")
        
        assert isinstance(result, ProcessedQuery)
        assert result.original_query == "Test query"
        assert len(result.embedding) == 3072
    
    @patch('online.query.TextEmbedder')
    def test_process_queries_function(self, mock_embedder_class):
        """Test process_queries convenience function."""
        # Setup mock
        mock_embedder = Mock()
        mock_embedder.model = "text-embedding-3-large"
        mock_embedder.dimensions = 3072
        mock_embedder.embed_texts.return_value = [
            [0.1] * 3072,
            [0.2] * 3072,
        ]
        mock_embedder_class.return_value = mock_embedder
        
        queries = ["Query 1", "Query 2"]
        results = process_queries(queries)
        
        assert len(results) == 2
        assert all(isinstance(r, ProcessedQuery) for r in results)
        assert results[0].original_query == "Query 1"
        assert results[1].original_query == "Query 2"


class TestIntegration:
    """Integration tests with real embeddings (if API key is available)."""
    
    @pytest.mark.skip(reason="Integration test - requires OpenAI API key")
    def test_real_query_processing(self):
        """Test query processing with real OpenAI API."""
        processor = QueryProcessor()
        
        query = "What is Santiago's Personal Legend?"
        result = processor.process_query(query)
        
        assert result.dimensions == 3072
        assert result.model == "text-embedding-3-large"
        assert len(result.embedding) == 3072
        # Verify embeddings are normalized (roughly unit length)
        magnitude = sum(x**2 for x in result.embedding) ** 0.5
        assert 0.9 < magnitude < 1.1
    
    @pytest.mark.skip(reason="Integration test - requires OpenAI API key")
    def test_real_batch_processing(self):
        """Test batch processing with real OpenAI API."""
        processor = QueryProcessor()
        
        queries = [
            "What is alchemy?",
            "Who is Fatima?",
            "Where are the Pyramids?",
        ]
        results = processor.process_queries(queries)
        
        assert len(results) == 3
        assert all(r.dimensions == 3072 for r in results)
        assert all(len(r.embedding) == 3072 for r in results)
