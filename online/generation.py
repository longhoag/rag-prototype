"""
Generation module for creating AI responses using GPT with retrieved context.

This module constructs prompts from user queries and retrieved document chunks,
then generates contextual answers using OpenAI's GPT models.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from loguru import logger
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from config import get_config
from online.retrieval import RetrievalResult


@dataclass
class GeneratedResponse:
    """
    Represents a generated response with metadata.
    
    Attributes:
        answer: The generated answer text
        sources: List of source chunk IDs used to generate the answer
        model: The model used for generation
        token_usage: Token usage statistics (prompt, completion, total)
    """
    answer: str
    sources: List[str]
    model: str
    token_usage: Dict[str, int]
    
    def __repr__(self) -> str:
        """Short representation."""
        answer_preview = self.answer[:100] + "..." if len(self.answer) > 100 else self.answer
        return (
            f"GeneratedResponse(answer='{answer_preview}', "
            f"sources={len(self.sources)} chunks, "
            f"tokens={self.token_usage.get('total_tokens', 0)})"
        )


class ResponseGenerator:
    """
    Generates contextual responses using GPT models.
    
    Takes a user query and retrieved document chunks, constructs a prompt,
    and generates an answer using OpenAI's chat completion API.
    """
    
    def __init__(self):
        """Initialize OpenAI client and load configuration."""
        self.config = get_config()
        self.client = OpenAI(api_key=self.config.openai_api_key)
        self.model = self.config.openai_chat_model
        
        logger.info(f"ResponseGenerator initialized with model '{self.model}'")
    
    def _construct_context(self, results: List[RetrievalResult]) -> str:
        """
        Construct context string from retrieved results.
        
        Args:
            results: List of retrieved chunks
            
        Returns:
            Formatted context string with source citations
        """
        if not results:
            return "No relevant context found."
        
        context_parts = []
        for i, result in enumerate(results, 1):
            # Format each chunk with source citation
            chunk_text = f"[Source {i}]\n{result.text}\n"
            context_parts.append(chunk_text)
        
        context = "\n".join(context_parts)
        logger.debug(f"Constructed context from {len(results)} chunks ({len(context)} chars)")
        
        return context
    
    def _construct_system_message(self) -> str:
        """
        Construct the system message for the LLM.
        
        Returns:
            System message string
        """
        return (
            "You are a helpful AI assistant that answers questions based on provided context. "
            "Your answers should be accurate, concise, and directly address the user's question. "
            "Always cite your sources using [Source N] notation when referencing specific information. "
            "If the context doesn't contain enough information to answer the question, "
            "say so clearly and explain what information is missing."
        )
    
    def _construct_user_message(self, query: str, context: str) -> str:
        """
        Construct the user message with query and context.
        
        Args:
            query: The user's question
            context: The retrieved context
            
        Returns:
            Formatted user message
        """
        return (
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Please answer the question based on the context provided above. "
            f"Cite your sources using [Source N] notation."
        )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _call_openai(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Call OpenAI API with retry logic.
        
        Note: GPT-5 models don't support temperature (locked to 1.0) and use
        max_completion_tokens instead of max_tokens.
        
        Args:
            messages: List of message dictionaries
            temperature: Sampling temperature (0-2, ignored for GPT-5)
            max_tokens: Maximum tokens to generate (converted to max_completion_tokens for GPT-5)
            
        Returns:
            API response dictionary
            
        Raises:
            Exception: If API call fails after retries
        """
        logger.debug(f"Calling OpenAI API with model {self.model}")
        
        # GPT-5 has different parameter requirements
        is_gpt5 = self.model.startswith("gpt-5")
        
        params = {
            "model": self.model,
            "messages": messages,
        }
        
        # GPT-5 doesn't support temperature (locked to 1.0)
        if not is_gpt5:
            params["temperature"] = temperature
        
        # GPT-5 uses max_completion_tokens instead of max_tokens
        if max_tokens is not None:
            if is_gpt5:
                params["max_completion_tokens"] = max_tokens
            else:
                params["max_tokens"] = max_tokens
        
        response = self.client.chat.completions.create(**params)
        
        logger.debug("OpenAI API call successful")
        return response
    
    def generate(
        self,
        query: str,
        retrieved_chunks: List[RetrievalResult],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> GeneratedResponse:
        """
        Generate a response based on query and retrieved context.
        
        Note: For GPT-5 models, temperature is ignored (locked to 1.0) and
        max_tokens is automatically converted to max_completion_tokens.
        
        Args:
            query: The user's question
            retrieved_chunks: List of retrieved document chunks
            temperature: Sampling temperature (0-2, default 0.7, ignored for GPT-5)
            max_tokens: Maximum tokens to generate (None for model default)
            
        Returns:
            GeneratedResponse with answer and metadata
            
        Raises:
            ValueError: If query is empty
            Exception: If generation fails after retries
            
        Example:
            >>> generator = ResponseGenerator()
            >>> results = [...]  # Retrieved chunks
            >>> response = generator.generate("What is X?", results)
            >>> print(response.answer)
        """
        # Validate input
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        logger.info(f"Generating response for query: {query[:100]}...")
        
        # Construct context from retrieved chunks
        context = self._construct_context(retrieved_chunks)
        
        # Build messages
        messages = [
            {"role": "system", "content": self._construct_system_message()},
            {"role": "user", "content": self._construct_user_message(query, context)},
        ]
        
        # Call OpenAI API
        response = self._call_openai(messages, temperature, max_tokens)
        
        # Extract response data
        answer = response.choices[0].message.content
        source_ids = [chunk.chunk_id for chunk in retrieved_chunks]
        
        # Get token usage
        token_usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
        
        logger.success(
            f"Generated response ({token_usage['completion_tokens']} tokens) "
            f"using {len(source_ids)} sources"
        )
        
        return GeneratedResponse(
            answer=answer,
            sources=source_ids,
            model=self.model,
            token_usage=token_usage,
        )


# Convenience function for quick usage

def generate_response(
    query: str,
    retrieved_chunks: List[RetrievalResult],
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
) -> GeneratedResponse:
    """
    Convenience function to generate a response without explicitly creating a generator.
    
    Note: For GPT-5 models, temperature is ignored (locked to 1.0).
    
    Args:
        query: The user's question
        retrieved_chunks: List of retrieved document chunks
        temperature: Sampling temperature (0-2, ignored for GPT-5)
        max_tokens: Maximum tokens to generate
        
    Returns:
        GeneratedResponse object
        
    Example:
        >>> from online.generation import generate_response
        >>> results = [...]  # from retrieval
        >>> response = generate_response("What is X?", results)
        >>> print(response.answer)
    """
    generator = ResponseGenerator()
    return generator.generate(query, retrieved_chunks, temperature, max_tokens)
