"""
Configuration module for RAG prototype.
Loads and validates environment variables from .env file.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from loguru import logger


@dataclass
class Config:
    """Configuration class for RAG pipeline."""
    
    # OpenAI Configuration
    openai_api_key: str
    openai_embedding_model: str
    openai_embedding_dimensions: int
    openai_chat_model: str
    
    # Pinecone Configuration
    pinecone_api_key: str
    pinecone_environment: str
    pinecone_index_name: str
    
    # RAG Pipeline Configuration
    chunk_min_tokens: int
    chunk_max_tokens: int
    chunk_overlap_min: int
    chunk_overlap_max: int
    
    # Retrieval Configuration
    top_k: int
    retrieval_min_score: float
    
    @classmethod
    def from_env(cls, env_path: Optional[str] = None) -> "Config":
        """
        Load configuration from environment variables.
        
        Args:
            env_path: Optional path to .env file. If None, searches for .env in current directory.
            
        Returns:
            Config instance with loaded values.
            
        Raises:
            ValueError: If required environment variables are missing or invalid.
        """
        # Load .env file
        if env_path:
            load_dotenv(env_path)
        else:
            # Search for .env in current directory and parent directories
            env_file = Path(".env")
            if env_file.exists():
                load_dotenv(env_file)
                logger.info(f"Loaded environment variables from {env_file.absolute()}")
            else:
                logger.warning("No .env file found. Using system environment variables.")
        
        # Required string variables
        required_vars = {
            "OPENAI_API_KEY": "OpenAI API key",
            "PINECONE_API_KEY": "Pinecone API key",
            "PINECONE_ENVIRONMENT": "Pinecone environment",
        }
        
        missing_vars = []
        for var, description in required_vars.items():
            if not os.getenv(var):
                missing_vars.append(f"{var} ({description})")
        
        if missing_vars:
            error_msg = f"Missing required environment variables:\n  - " + "\n  - ".join(missing_vars)
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Load and validate configuration
        try:
            config = cls(
                # OpenAI
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                openai_embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"),
                openai_embedding_dimensions=int(os.getenv("OPENAI_EMBEDDING_DIMENSIONS", "3072")),
                openai_chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-5"),
                
                # Pinecone
                pinecone_api_key=os.getenv("PINECONE_API_KEY"),
                pinecone_environment=os.getenv("PINECONE_ENVIRONMENT"),
                pinecone_index_name=os.getenv("PINECONE_INDEX_NAME", "rag-alchemist"),
                
                # RAG Pipeline
                chunk_min_tokens=int(os.getenv("CHUNK_MIN_TOKENS", "400")),
                chunk_max_tokens=int(os.getenv("CHUNK_MAX_TOKENS", "800")),
                chunk_overlap_min=int(os.getenv("CHUNK_OVERLAP_MIN", "80")),
                chunk_overlap_max=int(os.getenv("CHUNK_OVERLAP_MAX", "150")),
                
                # Retrieval
                top_k=int(os.getenv("TOP_K", "10")),
                retrieval_min_score=float(os.getenv("RETRIEVAL_MIN_SCORE", "0.7")),
            )
            
            # Validate ranges
            if config.chunk_min_tokens >= config.chunk_max_tokens:
                raise ValueError(f"CHUNK_MIN_TOKENS ({config.chunk_min_tokens}) must be less than CHUNK_MAX_TOKENS ({config.chunk_max_tokens})")
            
            if config.chunk_overlap_min >= config.chunk_overlap_max:
                raise ValueError(f"CHUNK_OVERLAP_MIN ({config.chunk_overlap_min}) must be less than CHUNK_OVERLAP_MAX ({config.chunk_overlap_max})")
            
            if config.chunk_overlap_max >= config.chunk_min_tokens:
                raise ValueError(f"CHUNK_OVERLAP_MAX ({config.chunk_overlap_max}) must be less than CHUNK_MIN_TOKENS ({config.chunk_min_tokens})")
            
            if config.top_k <= 0:
                raise ValueError(f"TOP_K must be positive, got {config.top_k}")
            
            if not (0.0 <= config.retrieval_min_score <= 1.0):
                raise ValueError(f"RETRIEVAL_MIN_SCORE must be between 0.0 and 1.0, got {config.retrieval_min_score}")
            
            logger.success("Configuration loaded and validated successfully")
            return config
            
        except ValueError as e:
            logger.error(f"Invalid configuration value: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise


# Global config instance - load once on import
_config: Optional[Config] = None


def get_config() -> Config:
    """
    Get the global configuration instance.
    Loads configuration on first call.
    
    Returns:
        Config instance.
    """
    global _config
    if _config is None:
        _config = Config.from_env()
    return _config


def reload_config(env_path: Optional[str] = None) -> Config:
    """
    Reload configuration from environment variables.
    Useful for testing or when .env file changes.
    
    Args:
        env_path: Optional path to .env file.
        
    Returns:
        New Config instance.
    """
    global _config
    _config = Config.from_env(env_path)
    return _config
