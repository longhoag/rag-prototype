"""
Tests for the generation module.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass

from online.generation import (
    GeneratedResponse,
    ResponseGenerator,
    generate_response,
)
from online.retrieval import RetrievalResult


# Test Fixtures

@pytest.fixture
def mock_config():
    """Mock configuration."""
    config = Mock()
    config.openai_api_key = "test-api-key"
    config.openai_chat_model = "gpt-4o"
    return config


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client."""
    client = Mock()
    return client


@pytest.fixture
def sample_retrieval_results():
    """Sample retrieval results for testing."""
    return [
        RetrievalResult(
            chunk_id="chunk_001",
            text="Santiago is a shepherd boy who dreams of finding treasure.",
            score=0.95,
            metadata={"source": "document.txt"},
        ),
        RetrievalResult(
            chunk_id="chunk_002",
            text="The alchemist teaches Santiago about the Soul of the World.",
            score=0.92,
            metadata={"source": "document.txt"},
        ),
        RetrievalResult(
            chunk_id="chunk_003",
            text="Fatima is the woman Santiago falls in love with at the oasis.",
            score=0.88,
            metadata={"source": "document.txt"},
        ),
    ]


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response."""
    response = Mock()
    response.choices = [Mock()]
    response.choices[0].message.content = "Santiago's Personal Legend is to find his treasure at the pyramids."
    response.usage.prompt_tokens = 150
    response.usage.completion_tokens = 20
    response.usage.total_tokens = 170
    return response


# Test GeneratedResponse Dataclass

class TestGeneratedResponse:
    """Tests for GeneratedResponse dataclass."""
    
    def test_creation(self):
        """Test creating a GeneratedResponse."""
        response = GeneratedResponse(
            answer="This is an answer.",
            sources=["chunk_001", "chunk_002"],
            model="gpt-4o",
            token_usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        )
        
        assert response.answer == "This is an answer."
        assert response.sources == ["chunk_001", "chunk_002"]
        assert response.model == "gpt-4o"
        assert response.token_usage["total_tokens"] == 120
    
    def test_repr_short_answer(self):
        """Test __repr__ with short answer."""
        response = GeneratedResponse(
            answer="Short answer.",
            sources=["chunk_001"],
            model="gpt-4o",
            token_usage={"prompt_tokens": 30, "completion_tokens": 20, "total_tokens": 50},
        )
        
        repr_str = repr(response)
        assert "Short answer." in repr_str
        assert "1 chunks" in repr_str
        assert "tokens=50" in repr_str
    
    def test_repr_long_answer(self):
        """Test __repr__ with long answer (should be truncated)."""
        long_answer = "A" * 150
        response = GeneratedResponse(
            answer=long_answer,
            sources=["chunk_001", "chunk_002"],
            model="gpt-4o",
            token_usage={"total_tokens": 200},
        )
        
        repr_str = repr(response)
        assert len(repr_str) < len(long_answer) + 100  # Should be truncated
        assert "..." in repr_str
        assert "2 chunks" in repr_str


# Test ResponseGenerator Class

class TestResponseGenerator:
    """Tests for ResponseGenerator class."""
    
    def test_initialization(self, mock_config):
        """Test ResponseGenerator initialization."""
        with patch('online.generation.get_config', return_value=mock_config):
            with patch('online.generation.OpenAI') as mock_openai:
                generator = ResponseGenerator()
                
                assert generator.config == mock_config
                assert generator.model == "gpt-4o"
                mock_openai.assert_called_once_with(api_key="test-api-key")
    
    def test_construct_context_empty(self, mock_config):
        """Test constructing context with no results."""
        with patch('online.generation.get_config', return_value=mock_config):
            with patch('online.generation.OpenAI'):
                generator = ResponseGenerator()
                context = generator._construct_context([])
                
                assert context == "No relevant context found."
    
    def test_construct_context_single_result(self, mock_config, sample_retrieval_results):
        """Test constructing context with single result."""
        with patch('online.generation.get_config', return_value=mock_config):
            with patch('online.generation.OpenAI'):
                generator = ResponseGenerator()
                context = generator._construct_context([sample_retrieval_results[0]])
                
                assert "[Source 1]" in context
                assert "Santiago is a shepherd boy" in context
    
    def test_construct_context_multiple_results(self, mock_config, sample_retrieval_results):
        """Test constructing context with multiple results."""
        with patch('online.generation.get_config', return_value=mock_config):
            with patch('online.generation.OpenAI'):
                generator = ResponseGenerator()
                context = generator._construct_context(sample_retrieval_results)
                
                assert "[Source 1]" in context
                assert "[Source 2]" in context
                assert "[Source 3]" in context
                assert "Santiago is a shepherd boy" in context
                assert "alchemist teaches Santiago" in context
                assert "Fatima is the woman" in context
    
    def test_construct_system_message(self, mock_config):
        """Test system message construction."""
        with patch('online.generation.get_config', return_value=mock_config):
            with patch('online.generation.OpenAI'):
                generator = ResponseGenerator()
                system_msg = generator._construct_system_message()
                
                assert "helpful AI assistant" in system_msg
                assert "provided context" in system_msg
                assert "[Source N]" in system_msg
    
    def test_construct_user_message(self, mock_config):
        """Test user message construction."""
        with patch('online.generation.get_config', return_value=mock_config):
            with patch('online.generation.OpenAI'):
                generator = ResponseGenerator()
                context = "Some context here."
                query = "What is X?"
                
                user_msg = generator._construct_user_message(query, context)
                
                assert "Context:" in user_msg
                assert "Some context here." in user_msg
                assert "Question: What is X?" in user_msg
                assert "Cite your sources" in user_msg
    
    def test_call_openai_success(self, mock_config, mock_openai_response):
        """Test successful OpenAI API call."""
        with patch('online.generation.get_config', return_value=mock_config):
            mock_client = Mock()
            mock_client.chat.completions.create.return_value = mock_openai_response
            
            with patch('online.generation.OpenAI', return_value=mock_client):
                generator = ResponseGenerator()
                messages = [{"role": "system", "content": "Test"}]
                
                response = generator._call_openai(messages)
                
                assert response == mock_openai_response
                mock_client.chat.completions.create.assert_called_once()
    
    def test_call_openai_with_parameters(self, mock_config, mock_openai_response):
        """Test OpenAI API call with custom parameters."""
        with patch('online.generation.get_config', return_value=mock_config):
            mock_client = Mock()
            mock_client.chat.completions.create.return_value = mock_openai_response
            
            with patch('online.generation.OpenAI', return_value=mock_client):
                generator = ResponseGenerator()
                messages = [{"role": "system", "content": "Test"}]
                
                generator._call_openai(messages, temperature=0.5, max_tokens=100)
                
                call_args = mock_client.chat.completions.create.call_args
                assert call_args.kwargs['temperature'] == 0.5
                assert call_args.kwargs['max_tokens'] == 100
    
    def test_call_openai_retry_on_failure(self, mock_config):
        """Test OpenAI API call retries on failure."""
        with patch('online.generation.get_config', return_value=mock_config):
            mock_client = Mock()
            mock_client.chat.completions.create.side_effect = [
                Exception("API Error"),
                Exception("API Error"),
                Mock(choices=[Mock(message=Mock(content="Success"))]),
            ]
            
            with patch('online.generation.OpenAI', return_value=mock_client):
                generator = ResponseGenerator()
                messages = [{"role": "system", "content": "Test"}]
                
                # Should succeed on third attempt
                response = generator._call_openai(messages)
                assert mock_client.chat.completions.create.call_count == 3
    
    def test_generate_empty_query(self, mock_config):
        """Test generate with empty query raises ValueError."""
        with patch('online.generation.get_config', return_value=mock_config):
            with patch('online.generation.OpenAI'):
                generator = ResponseGenerator()
                
                with pytest.raises(ValueError, match="Query cannot be empty"):
                    generator.generate("", [])
    
    def test_generate_whitespace_query(self, mock_config):
        """Test generate with whitespace-only query raises ValueError."""
        with patch('online.generation.get_config', return_value=mock_config):
            with patch('online.generation.OpenAI'):
                generator = ResponseGenerator()
                
                with pytest.raises(ValueError, match="Query cannot be empty"):
                    generator.generate("   ", [])
    
    def test_generate_success(self, mock_config, sample_retrieval_results, mock_openai_response):
        """Test successful response generation."""
        with patch('online.generation.get_config', return_value=mock_config):
            mock_client = Mock()
            mock_client.chat.completions.create.return_value = mock_openai_response
            
            with patch('online.generation.OpenAI', return_value=mock_client):
                generator = ResponseGenerator()
                query = "What is Santiago's Personal Legend?"
                
                response = generator.generate(query, sample_retrieval_results)
                
                assert isinstance(response, GeneratedResponse)
                assert "Santiago's Personal Legend" in response.answer
                assert len(response.sources) == 3
                assert response.sources == ["chunk_001", "chunk_002", "chunk_003"]
                assert response.model == "gpt-4o"
                assert response.token_usage["prompt_tokens"] == 150
                assert response.token_usage["completion_tokens"] == 20
                assert response.token_usage["total_tokens"] == 170
    
    def test_generate_with_custom_temperature(self, mock_config, sample_retrieval_results, mock_openai_response):
        """Test generation with custom temperature."""
        with patch('online.generation.get_config', return_value=mock_config):
            mock_client = Mock()
            mock_client.chat.completions.create.return_value = mock_openai_response
            
            with patch('online.generation.OpenAI', return_value=mock_client):
                generator = ResponseGenerator()
                query = "What is X?"
                
                generator.generate(query, sample_retrieval_results, temperature=0.3)
                
                call_args = mock_client.chat.completions.create.call_args
                assert call_args.kwargs['temperature'] == 0.3
    
    def test_generate_with_max_tokens(self, mock_config, sample_retrieval_results, mock_openai_response):
        """Test generation with max_tokens limit."""
        with patch('online.generation.get_config', return_value=mock_config):
            mock_client = Mock()
            mock_client.chat.completions.create.return_value = mock_openai_response
            
            with patch('online.generation.OpenAI', return_value=mock_client):
                generator = ResponseGenerator()
                query = "What is X?"
                
                generator.generate(query, sample_retrieval_results, max_tokens=200)
                
                call_args = mock_client.chat.completions.create.call_args
                assert call_args.kwargs['max_tokens'] == 200
    
    def test_generate_empty_results(self, mock_config, mock_openai_response):
        """Test generation with no retrieved results."""
        with patch('online.generation.get_config', return_value=mock_config):
            mock_client = Mock()
            mock_client.chat.completions.create.return_value = mock_openai_response
            
            with patch('online.generation.OpenAI', return_value=mock_client):
                generator = ResponseGenerator()
                query = "What is X?"
                
                response = generator.generate(query, [])
                
                # Should still work, but with "No relevant context found" message
                assert isinstance(response, GeneratedResponse)
                assert len(response.sources) == 0
                
                # Check that context included the "no context" message
                call_args = mock_client.chat.completions.create.call_args
                messages = call_args.kwargs['messages']
                user_message = messages[1]['content']
                assert "No relevant context found" in user_message


# Test Convenience Function

class TestConvenienceFunction:
    """Tests for generate_response convenience function."""
    
    def test_generate_response(self, mock_config, sample_retrieval_results, mock_openai_response):
        """Test convenience function."""
        with patch('online.generation.get_config', return_value=mock_config):
            mock_client = Mock()
            mock_client.chat.completions.create.return_value = mock_openai_response
            
            with patch('online.generation.OpenAI', return_value=mock_client):
                query = "What is Santiago's Personal Legend?"
                
                response = generate_response(query, sample_retrieval_results)
                
                assert isinstance(response, GeneratedResponse)
                assert "Santiago's Personal Legend" in response.answer
                assert len(response.sources) == 3


# Integration Tests (Optional - Skip by Default)

class TestIntegration:
    """Integration tests with real OpenAI API (skipped by default)."""
    
    @pytest.mark.skip(reason="Requires real OpenAI API key and costs money")
    def test_real_generation(self):
        """Test with real OpenAI API."""
        from config import get_config
        
        config = get_config()
        generator = ResponseGenerator()
        
        # Create sample retrieval results
        results = [
            RetrievalResult(
                chunk_id="test_001",
                text="The Alchemist is a novel by Paulo Coelho about a shepherd named Santiago.",
                score=0.95,
                metadata={},
            ),
        ]
        
        response = generator.generate("What is The Alchemist about?", results)
        
        assert isinstance(response, GeneratedResponse)
        assert len(response.answer) > 0
        assert response.token_usage["total_tokens"] > 0
    
    @pytest.mark.skip(reason="Requires real OpenAI API key and costs money")
    def test_real_generation_no_context(self):
        """Test with real API but no context."""
        generator = ResponseGenerator()
        
        response = generator.generate("What is X?", [])
        
        assert isinstance(response, GeneratedResponse)
        assert len(response.answer) > 0
