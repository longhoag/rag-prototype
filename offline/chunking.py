"""
Chunking module for splitting text into semantically coherent chunks.
Uses sentence-based splitting with token counting.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List

import nltk
import tiktoken
from loguru import logger

from config import get_config


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    
    text: str
    token_count: int
    chunk_id: int
    start_char: int
    end_char: int


class TextChunker:
    """
    Handles text chunking with sentence-based splitting and token counting.
    
    Uses NLTK for sentence boundary detection and tiktoken for token counting.
    Creates chunks between CHUNK_MIN_TOKENS and CHUNK_MAX_TOKENS with overlap.
    """
    
    def __init__(self):
        """Initialize the chunker with configuration and tokenizer."""
        self.config = get_config()
        self.encoding = tiktoken.get_encoding("cl100k_base")
        logger.info(
            f"TextChunker initialized: {self.config.chunk_min_tokens}-{self.config.chunk_max_tokens} tokens, "
            f"overlap {self.config.chunk_overlap_min}-{self.config.chunk_overlap_max} tokens"
        )
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using tiktoken.
        
        Args:
            text: Text to count tokens for.
            
        Returns:
            Number of tokens.
        """
        return len(self.encoding.encode(text))
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using NLTK.
        
        Args:
            text: Text to split.
            
        Returns:
            List of sentences.
        """
        try:
            sentences = nltk.sent_tokenize(text)
            logger.debug(f"Split text into {len(sentences)} sentences")
            return sentences
        except LookupError:
            logger.error(
                "NLTK punkt tokenizer not found. "
                "Run: poetry run python scripts/setup.py"
            )
            raise
    
    def create_chunks(self, text: str) -> List[Chunk]:
        """
        Create chunks from text using sentence-based splitting.
        
        Algorithm:
        1. Split text into sentences using NLTK
        2. Combine sentences until reaching CHUNK_MIN_TOKENS to CHUNK_MAX_TOKENS
        3. Add overlap from previous chunk (CHUNK_OVERLAP_MIN to CHUNK_OVERLAP_MAX tokens)
        4. Ensure semantic coherence by keeping sentences intact
        
        Args:
            text: Text to chunk.
            
        Returns:
            List of Chunk objects.
        """
        logger.info("Starting text chunking...")
        
        # Split into sentences
        sentences = self.split_into_sentences(text)
        
        if not sentences:
            logger.warning("No sentences found in text")
            return []
        
        chunks = []
        current_chunk_sentences = []
        current_tokens = 0
        char_position = 0
        
        # Track previous chunk's text for overlap
        previous_chunk_text = ""
        
        for i, sentence in enumerate(sentences):
            sentence_tokens = self.count_tokens(sentence)
            
            # If adding this sentence would exceed max tokens, finalize current chunk
            if current_tokens + sentence_tokens > self.config.chunk_max_tokens and current_chunk_sentences:
                chunk_text = " ".join(current_chunk_sentences)
                chunk_token_count = self.count_tokens(chunk_text)
                
                # Create chunk
                chunk = Chunk(
                    text=chunk_text,
                    token_count=chunk_token_count,
                    chunk_id=len(chunks),
                    start_char=char_position,
                    end_char=char_position + len(chunk_text)
                )
                chunks.append(chunk)
                
                logger.debug(
                    f"Chunk {chunk.chunk_id}: {chunk.token_count} tokens, "
                    f"{len(current_chunk_sentences)} sentences"
                )
                
                # Prepare for next chunk with overlap
                previous_chunk_text = chunk_text
                
                # Calculate overlap: find sentences from previous chunk to include
                overlap_sentences = []
                overlap_tokens = 0
                
                # Work backwards through previous chunk's sentences to build overlap
                for prev_sentence in reversed(current_chunk_sentences):
                    sentence_tok = self.count_tokens(prev_sentence)
                    if overlap_tokens + sentence_tok <= self.config.chunk_overlap_max:
                        overlap_sentences.insert(0, prev_sentence)
                        overlap_tokens += sentence_tok
                    else:
                        break
                
                # Only use overlap if it meets minimum requirement
                if overlap_tokens >= self.config.chunk_overlap_min:
                    current_chunk_sentences = overlap_sentences
                    current_tokens = overlap_tokens
                    logger.debug(f"Added overlap: {overlap_tokens} tokens")
                else:
                    current_chunk_sentences = []
                    current_tokens = 0
                
                char_position += len(chunk_text) + 1  # +1 for space
            
            # Add current sentence to chunk
            current_chunk_sentences.append(sentence)
            current_tokens += sentence_tokens
            
            # If we've reached minimum tokens and this is the last sentence, finalize
            if i == len(sentences) - 1 and current_tokens >= self.config.chunk_min_tokens:
                chunk_text = " ".join(current_chunk_sentences)
                chunk_token_count = self.count_tokens(chunk_text)
                
                chunk = Chunk(
                    text=chunk_text,
                    token_count=chunk_token_count,
                    chunk_id=len(chunks),
                    start_char=char_position,
                    end_char=char_position + len(chunk_text)
                )
                chunks.append(chunk)
                
                logger.debug(
                    f"Chunk {chunk.chunk_id} (final): {chunk.token_count} tokens, "
                    f"{len(current_chunk_sentences)} sentences"
                )
        
        # Handle remaining sentences if they don't meet minimum
        if current_chunk_sentences and len(chunks) > 0:
            # Add remaining sentences to last chunk rather than creating tiny chunk
            last_chunk = chunks[-1]
            remaining_text = " ".join(current_chunk_sentences)
            combined_text = last_chunk.text + " " + remaining_text
            combined_tokens = self.count_tokens(combined_text)
            
            if combined_tokens <= self.config.chunk_max_tokens * 1.2:  # Allow 20% overflow for final chunk
                chunks[-1] = Chunk(
                    text=combined_text,
                    token_count=combined_tokens,
                    chunk_id=last_chunk.chunk_id,
                    start_char=last_chunk.start_char,
                    end_char=last_chunk.end_char + len(remaining_text) + 1
                )
                logger.debug(f"Merged remaining sentences into final chunk: {combined_tokens} tokens")
            else:
                # Create new chunk if remaining text is substantial
                chunk_text = remaining_text
                chunk_token_count = self.count_tokens(chunk_text)
                
                chunk = Chunk(
                    text=chunk_text,
                    token_count=chunk_token_count,
                    chunk_id=len(chunks),
                    start_char=char_position,
                    end_char=char_position + len(chunk_text)
                )
                chunks.append(chunk)
                logger.debug(f"Chunk {chunk.chunk_id} (remaining): {chunk_token_count} tokens")
        
        logger.success(f"Created {len(chunks)} chunks from {len(sentences)} sentences")
        
        # Log statistics
        if chunks:
            token_counts = [c.token_count for c in chunks]
            avg_tokens = sum(token_counts) / len(token_counts)
            min_tokens = min(token_counts)
            max_tokens = max(token_counts)
            
            logger.info(
                f"Chunk statistics: avg={avg_tokens:.0f} tokens, "
                f"min={min_tokens}, max={max_tokens}"
            )
        
        return chunks
    
    def chunk_file(self, file_path: str) -> List[Chunk]:
        """
        Read a text file and create chunks from its content.
        
        Args:
            file_path: Path to text file.
            
        Returns:
            List of Chunk objects.
        """
        path = Path(file_path)
        
        if not path.exists():
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"Reading file: {file_path}")
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            logger.info(f"File loaded: {len(text)} characters")
            return self.create_chunks(text)
            
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            raise


def chunk_document(document_path: str = "document.txt") -> List[Chunk]:
    """
    Convenience function to chunk a document.
    
    Args:
        document_path: Path to document file.
        
    Returns:
        List of Chunk objects.
    """
    chunker = TextChunker()
    return chunker.chunk_file(document_path)
