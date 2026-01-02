"""
Comprehensive tests for the retrieval module.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from online.retrieval import RetrievalResult, VectorRetriever, retrieve


class TestRetrievalResult:
    """Test the RetrievalResult dataclass."""
    
    def test_retrieval_result_creation(self):
        """Test creating a RetrievalResult."""
        result = RetrievalResult(
            chunk_id="chunk_001",
            text="This is test text",
            score=0.95,
            metadata={"source": "test.txt", "page": 1}
        )
        
        assert result.chunk_id == "chunk_001"
        assert result.text == "This is test text"
        assert result.score == 0.95
        assert result.metadata == {"source": "test.txt", "page": 1}
    
    def test_retrieval_result_repr_short_text(self):
        """Test __repr__ with short text."""
        result = RetrievalResult(
            chunk_id="chunk_001",
            text="Short text",
            score=0.85,
            metadata={}
        )
        
        repr_str = repr(result)
        assert "chunk_001" in repr_str
        assert "0.8500" in repr_str
        assert "Short text" in repr_str
        assert "..." not in repr_str
    
    def test_retrieval_result_repr_long_text(self):
        """Test __repr__ with long text (should truncate)."""
        long_text = "a" * 150
        result = RetrievalResult(
            chunk_id="chunk_002",
            text=long_text,
            score=0.92,
            metadata={}
        )
        
        repr_str = repr(result)
        assert "chunk_002" in repr_str
        assert "0.9200" in repr_str
        assert "..." in repr_str
        # Text should be truncated to 100 chars + "..."
        assert long_text[:100] in repr_str


class TestVectorRetriever:
    """Test the VectorRetriever class."""
    
    @patch('online.retrieval.Pinecone')
    @patch('online.retrieval.get_config')
    def test_initialization(self, mock_get_config, mock_pinecone):
        """Test VectorRetriever initialization."""
        # Setup mock config
        mock_config = Mock()
        mock_config.pinecone_api_key = "test-api-key"
        mock_config.pinecone_index = "test-index"
        mock_config.top_k = 10
        mock_config.retrieval_min_score = 0.7
        mock_get_config.return_value = mock_config
        
        # Setup mock Pinecone
        mock_pc_instance = Mock()
        mock_index = Mock()
        mock_pc_instance.Index.return_value = mock_index
        mock_pinecone.return_value = mock_pc_instance
        
        # Create retriever
        retriever = VectorRetriever()
        
        # Verify initialization
        assert retriever.config == mock_config
        assert retriever.top_k == 10
        assert retriever.min_score == 0.7
        mock_pinecone.assert_called_once_with(api_key="test-api-key")
        mock_pc_instance.Index.assert_called_once_with("test-index")
    
    @patch('online.retrieval.Pinecone')
    @patch('online.retrieval.get_config')
    def test_query_pinecone_success(self, mock_get_config, mock_pinecone):
        """Test successful Pinecone query."""
        # Setup mocks
        mock_config = Mock()
        mock_config.pinecone_api_key = "test-key"
        mock_config.pinecone_index = "test-index"
        mock_config.top_k = 5
        mock_config.retrieval_min_score = 0.7
        mock_get_config.return_value = mock_config
        
        mock_index = Mock()
        mock_index.query.return_value = {
            'matches': [
                {
                    'id': 'chunk_001',
                    'score': 0.95,
                    'metadata': {'text': 'First match'}
                },
                {
                    'id': 'chunk_002',
                    'score': 0.88,
                    'metadata': {'text': 'Second match'}
                }
            ]
        }
        
        mock_pc_instance = Mock()
        mock_pc_instance.Index.return_value = mock_index
        mock_pinecone.return_value = mock_pc_instance
        
        # Create retriever and query
        retriever = VectorRetriever()
        embedding = [0.1] * 3072
        
        matches = retriever._query_pinecone(embedding, 5)
        
        # Verify
        assert len(matches) == 2
        assert matches[0]['id'] == 'chunk_001'
        assert matches[0]['score'] == 0.95
        mock_index.query.assert_called_once_with(
            vector=embedding,
            top_k=5,
            include_metadata=True
        )
    
    @patch('online.retrieval.Pinecone')
    @patch('online.retrieval.get_config')
    def test_query_pinecone_no_matches(self, mock_get_config, mock_pinecone):
        """Test Pinecone query with no matches."""
        # Setup mocks
        mock_config = Mock()
        mock_config.pinecone_api_key = "test-key"
        mock_config.pinecone_index = "test-index"
        mock_config.top_k = 5
        mock_config.retrieval_min_score = 0.7
        mock_get_config.return_value = mock_config
        
        mock_index = Mock()
        mock_index.query.return_value = {'matches': []}
        
        mock_pc_instance = Mock()
        mock_pc_instance.Index.return_value = mock_index
        mock_pinecone.return_value = mock_pc_instance
        
        # Create retriever and query
        retriever = VectorRetriever()
        embedding = [0.1] * 3072
        
        matches = retriever._query_pinecone(embedding, 5)
        
        # Verify
        assert matches == []
    
    @patch('online.retrieval.Pinecone')
    @patch('online.retrieval.get_config')
    def test_filter_by_score(self, mock_get_config, mock_pinecone):
        """Test filtering matches by score threshold."""
        # Setup mocks
        mock_config = Mock()
        mock_config.pinecone_api_key = "test-key"
        mock_config.pinecone_index = "test-index"
        mock_config.top_k = 5
        mock_config.retrieval_min_score = 0.7
        mock_get_config.return_value = mock_config
        
        mock_pc_instance = Mock()
        mock_pc_instance.Index.return_value = Mock()
        mock_pinecone.return_value = mock_pc_instance
        
        # Create retriever
        retriever = VectorRetriever()
        
        # Test data
        matches = [
            {'id': 'chunk_001', 'score': 0.95},
            {'id': 'chunk_002', 'score': 0.85},
            {'id': 'chunk_003', 'score': 0.65},  # Below threshold
            {'id': 'chunk_004', 'score': 0.55},  # Below threshold
        ]
        
        # Filter with threshold 0.7
        filtered = retriever._filter_by_score(matches, 0.7)
        
        # Verify
        assert len(filtered) == 2
        assert filtered[0]['id'] == 'chunk_001'
        assert filtered[1]['id'] == 'chunk_002'
    
    @patch('online.retrieval.Pinecone')
    @patch('online.retrieval.get_config')
    def test_filter_by_score_all_filtered(self, mock_get_config, mock_pinecone):
        """Test when all matches are filtered out."""
        # Setup mocks
        mock_config = Mock()
        mock_config.pinecone_api_key = "test-key"
        mock_config.pinecone_index = "test-index"
        mock_config.top_k = 5
        mock_config.retrieval_min_score = 0.7
        mock_get_config.return_value = mock_config
        
        mock_pc_instance = Mock()
        mock_pc_instance.Index.return_value = Mock()
        mock_pinecone.return_value = mock_pc_instance
        
        # Create retriever
        retriever = VectorRetriever()
        
        # Test data with all scores below threshold
        matches = [
            {'id': 'chunk_001', 'score': 0.65},
            {'id': 'chunk_002', 'score': 0.55},
        ]
        
        # Filter with high threshold
        filtered = retriever._filter_by_score(matches, 0.7)
        
        # Verify
        assert filtered == []
    
    @patch('online.retrieval.Pinecone')
    @patch('online.retrieval.get_config')
    def test_parse_matches(self, mock_get_config, mock_pinecone):
        """Test parsing Pinecone matches into RetrievalResult objects."""
        # Setup mocks
        mock_config = Mock()
        mock_config.pinecone_api_key = "test-key"
        mock_config.pinecone_index = "test-index"
        mock_config.top_k = 5
        mock_config.retrieval_min_score = 0.7
        mock_get_config.return_value = mock_config
        
        mock_pc_instance = Mock()
        mock_pc_instance.Index.return_value = Mock()
        mock_pinecone.return_value = mock_pc_instance
        
        # Create retriever
        retriever = VectorRetriever()
        
        # Test data
        matches = [
            {
                'id': 'chunk_001',
                'score': 0.95,
                'metadata': {
                    'text': 'First chunk text',
                    'source': 'test.txt',
                    'page': 1
                }
            },
            {
                'id': 'chunk_002',
                'score': 0.88,
                'metadata': {
                    'text': 'Second chunk text',
                    'source': 'test.txt',
                    'page': 2
                }
            }
        ]
        
        # Parse matches
        results = retriever._parse_matches(matches)
        
        # Verify
        assert len(results) == 2
        assert isinstance(results[0], RetrievalResult)
        assert results[0].chunk_id == 'chunk_001'
        assert results[0].text == 'First chunk text'
        assert results[0].score == 0.95
        assert results[0].metadata['source'] == 'test.txt'
    
    @patch('online.retrieval.Pinecone')
    @patch('online.retrieval.get_config')
    def test_parse_matches_missing_text(self, mock_get_config, mock_pinecone):
        """Test parsing matches when text is missing from metadata."""
        # Setup mocks
        mock_config = Mock()
        mock_config.pinecone_api_key = "test-key"
        mock_config.pinecone_index = "test-index"
        mock_config.top_k = 5
        mock_config.retrieval_min_score = 0.7
        mock_get_config.return_value = mock_config
        
        mock_pc_instance = Mock()
        mock_pc_instance.Index.return_value = Mock()
        mock_pinecone.return_value = mock_pc_instance
        
        # Create retriever
        retriever = VectorRetriever()
        
        # Test data with missing text
        matches = [
            {
                'id': 'chunk_001',
                'score': 0.95,
                'metadata': {'text': 'Has text'}
            },
            {
                'id': 'chunk_002',
                'score': 0.88,
                'metadata': {}  # No text field
            }
        ]
        
        # Parse matches
        results = retriever._parse_matches(matches)
        
        # Verify - should skip chunk without text
        assert len(results) == 1
        assert results[0].chunk_id == 'chunk_001'
    
    @patch('online.retrieval.Pinecone')
    @patch('online.retrieval.get_config')
    def test_retrieve_success(self, mock_get_config, mock_pinecone):
        """Test successful retrieval."""
        # Setup mocks
        mock_config = Mock()
        mock_config.pinecone_api_key = "test-key"
        mock_config.pinecone_index = "test-index"
        mock_config.top_k = 5
        mock_config.retrieval_min_score = 0.7
        mock_get_config.return_value = mock_config
        
        mock_index = Mock()
        mock_index.query.return_value = {
            'matches': [
                {
                    'id': 'chunk_001',
                    'score': 0.95,
                    'metadata': {'text': 'First match'}
                },
                {
                    'id': 'chunk_002',
                    'score': 0.88,
                    'metadata': {'text': 'Second match'}
                },
                {
                    'id': 'chunk_003',
                    'score': 0.65,  # Below threshold
                    'metadata': {'text': 'Third match'}
                }
            ]
        }
        
        mock_pc_instance = Mock()
        mock_pc_instance.Index.return_value = mock_index
        mock_pinecone.return_value = mock_pc_instance
        
        # Create retriever and retrieve
        retriever = VectorRetriever()
        embedding = [0.1] * 3072
        
        results = retriever.retrieve(embedding)
        
        # Verify - should filter out the one below threshold
        assert len(results) == 2
        assert results[0].chunk_id == 'chunk_001'
        assert results[0].score == 0.95
        assert results[1].chunk_id == 'chunk_002'
        assert results[1].score == 0.88
    
    @patch('online.retrieval.Pinecone')
    @patch('online.retrieval.get_config')
    def test_retrieve_with_custom_params(self, mock_get_config, mock_pinecone):
        """Test retrieval with custom top_k and min_score."""
        # Setup mocks
        mock_config = Mock()
        mock_config.pinecone_api_key = "test-key"
        mock_config.pinecone_index = "test-index"
        mock_config.top_k = 5
        mock_config.retrieval_min_score = 0.7
        mock_get_config.return_value = mock_config
        
        mock_index = Mock()
        mock_index.query.return_value = {
            'matches': [
                {
                    'id': 'chunk_001',
                    'score': 0.95,
                    'metadata': {'text': 'Match'}
                }
            ]
        }
        
        mock_pc_instance = Mock()
        mock_pc_instance.Index.return_value = mock_index
        mock_pinecone.return_value = mock_pc_instance
        
        # Create retriever and retrieve with custom params
        retriever = VectorRetriever()
        embedding = [0.1] * 3072
        
        results = retriever.retrieve(embedding, top_k=3, min_score=0.8)
        
        # Verify custom params were used
        mock_index.query.assert_called_once()
        call_args = mock_index.query.call_args
        assert call_args[1]['top_k'] == 3
    
    @patch('online.retrieval.Pinecone')
    @patch('online.retrieval.get_config')
    def test_retrieve_empty_embedding(self, mock_get_config, mock_pinecone):
        """Test retrieval with empty embedding."""
        # Setup mocks
        mock_config = Mock()
        mock_config.pinecone_api_key = "test-key"
        mock_config.pinecone_index = "test-index"
        mock_config.top_k = 5
        mock_config.retrieval_min_score = 0.7
        mock_get_config.return_value = mock_config
        
        mock_pc_instance = Mock()
        mock_pc_instance.Index.return_value = Mock()
        mock_pinecone.return_value = mock_pc_instance
        
        # Create retriever
        retriever = VectorRetriever()
        
        # Try to retrieve with empty embedding
        with pytest.raises(ValueError, match="Query embedding cannot be empty"):
            retriever.retrieve([])
    
    @patch('online.retrieval.Pinecone')
    @patch('online.retrieval.get_config')
    def test_retrieve_wrong_dimensions(self, mock_get_config, mock_pinecone):
        """Test retrieval with wrong embedding dimensions."""
        # Setup mocks
        mock_config = Mock()
        mock_config.pinecone_api_key = "test-key"
        mock_config.pinecone_index = "test-index"
        mock_config.top_k = 5
        mock_config.retrieval_min_score = 0.7
        mock_get_config.return_value = mock_config
        
        mock_pc_instance = Mock()
        mock_pc_instance.Index.return_value = Mock()
        mock_pinecone.return_value = mock_pc_instance
        
        # Create retriever
        retriever = VectorRetriever()
        
        # Try to retrieve with wrong dimensions
        wrong_embedding = [0.1] * 1536  # Wrong dimension
        
        with pytest.raises(ValueError, match="Expected embedding dimension 3072"):
            retriever.retrieve(wrong_embedding)
    
    @patch('online.retrieval.Pinecone')
    @patch('online.retrieval.get_config')
    def test_retrieve_no_matches(self, mock_get_config, mock_pinecone):
        """Test retrieval when Pinecone returns no matches."""
        # Setup mocks
        mock_config = Mock()
        mock_config.pinecone_api_key = "test-key"
        mock_config.pinecone_index = "test-index"
        mock_config.top_k = 5
        mock_config.retrieval_min_score = 0.7
        mock_get_config.return_value = mock_config
        
        mock_index = Mock()
        mock_index.query.return_value = {'matches': []}
        
        mock_pc_instance = Mock()
        mock_pc_instance.Index.return_value = mock_index
        mock_pinecone.return_value = mock_pc_instance
        
        # Create retriever and retrieve
        retriever = VectorRetriever()
        embedding = [0.1] * 3072
        
        results = retriever.retrieve(embedding)
        
        # Verify
        assert results == []
    
    @patch('online.retrieval.Pinecone')
    @patch('online.retrieval.get_config')
    def test_retrieve_all_filtered_by_score(self, mock_get_config, mock_pinecone):
        """Test retrieval when all matches are filtered by score."""
        # Setup mocks
        mock_config = Mock()
        mock_config.pinecone_api_key = "test-key"
        mock_config.pinecone_index = "test-index"
        mock_config.top_k = 5
        mock_config.retrieval_min_score = 0.9
        mock_get_config.return_value = mock_config
        
        mock_index = Mock()
        mock_index.query.return_value = {
            'matches': [
                {
                    'id': 'chunk_001',
                    'score': 0.75,  # Below threshold
                    'metadata': {'text': 'Match'}
                }
            ]
        }
        
        mock_pc_instance = Mock()
        mock_pc_instance.Index.return_value = mock_index
        mock_pinecone.return_value = mock_pc_instance
        
        # Create retriever and retrieve
        retriever = VectorRetriever()
        embedding = [0.1] * 3072
        
        results = retriever.retrieve(embedding)
        
        # Verify
        assert results == []


class TestConvenienceFunction:
    """Test the convenience retrieve() function."""
    
    @patch('online.retrieval.VectorRetriever')
    def test_retrieve_convenience_function(self, mock_retriever_class):
        """Test the convenience retrieve() function."""
        # Setup mock
        mock_retriever_instance = Mock()
        mock_retriever_instance.retrieve.return_value = [
            RetrievalResult(
                chunk_id="chunk_001",
                text="Test text",
                score=0.95,
                metadata={}
            )
        ]
        mock_retriever_class.return_value = mock_retriever_instance
        
        # Use convenience function
        embedding = [0.1] * 3072
        results = retrieve(embedding, top_k=5, min_score=0.8)
        
        # Verify
        mock_retriever_class.assert_called_once()
        mock_retriever_instance.retrieve.assert_called_once_with(
            embedding, 5, 0.8
        )
        assert len(results) == 1
        assert results[0].chunk_id == "chunk_001"


class TestIntegration:
    """Integration tests with real Pinecone (skipped by default)."""
    
    @pytest.mark.skip(reason="Requires real Pinecone connection and API key")
    def test_real_pinecone_query(self):
        """Test with real Pinecone connection."""
        from online.query import process_query
        
        # Process a real query
        query = "What is Santiago's Personal Legend?"
        processed = process_query(query)
        
        # Retrieve with real Pinecone
        retriever = VectorRetriever()
        results = retriever.retrieve(processed.embedding)
        
        # Verify
        assert len(results) > 0
        assert all(isinstance(r, RetrievalResult) for r in results)
        assert all(r.score >= 0.7 for r in results)
        print(f"\nRetrieved {len(results)} results:")
        for i, result in enumerate(results, 1):
            print(f"{i}. Score: {result.score:.4f}, Text: {result.text[:100]}...")
    
    @pytest.mark.skip(reason="Requires real Pinecone connection and API key")
    def test_end_to_end_query_and_retrieve(self):
        """Test end-to-end: query processing -> embedding -> retrieval."""
        from online.query import process_query
        
        # Process query
        query = "Who is Fatima?"
        processed = process_query(query)
        
        assert processed.embedding is not None
        assert len(processed.embedding) == 3072
        
        # Retrieve results
        results = retrieve(processed.embedding)
        
        # Verify results
        assert len(results) > 0
        print(f"\nQuery: {query}")
        print(f"Retrieved {len(results)} results:")
        for result in results:
            print(f"  - Score: {result.score:.4f}")
            print(f"    Text: {result.text[:200]}...")
