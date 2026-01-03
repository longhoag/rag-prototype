#!/usr/bin/env python3
"""
Online pipeline orchestrator for RAG query execution.

This script runs the complete query pipeline:
1. Process and validate user query
2. Embed query using OpenAI
3. Retrieve relevant chunks from Pinecone
4. Generate answer using GPT with retrieved context

Usage:
    # Single query mode
    poetry run python scripts/run_query.py "What is Santiago's Personal Legend?"
    
    # Interactive mode (multiple queries)
    poetry run python scripts/run_query.py
    
    # With custom retrieval parameters
    poetry run python scripts/run_query.py --top-k 5 --min-score 0.8 "What is X?"
"""

import sys
import time
import argparse
from pathlib import Path
from typing import Optional
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import get_config
from online.query import QueryProcessor
from online.retrieval import VectorRetriever
from online.generation import ResponseGenerator


class QueryPipeline:
    """
    Orchestrates the complete online RAG pipeline.
    """
    
    def __init__(self, top_k: Optional[int] = None, min_score: Optional[float] = None):
        """
        Initialize pipeline components.
        
        Args:
            top_k: Override config top_k for retrieval
            min_score: Override config min_score for retrieval
        """
        logger.info("Initializing query pipeline...")
        
        self.config = get_config()
        self.query_processor = QueryProcessor()
        self.retriever = VectorRetriever()
        self.generator = ResponseGenerator()
        
        # Override retrieval parameters if provided
        self.top_k = top_k if top_k is not None else self.config.top_k
        self.min_score = min_score if min_score is not None else self.config.retrieval_min_score
        
        logger.success(
            f"Pipeline ready (top_k={self.top_k}, min_score={self.min_score:.2f})"
        )
    
    def run(
        self,
        query: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        verbose: bool = False,
    ) -> dict:
        """
        Execute the complete query pipeline.
        
        Args:
            query: User's question
            temperature: GPT temperature (ignored for GPT-5)
            max_tokens: Maximum tokens to generate
            verbose: Show detailed progress
            
        Returns:
            Dictionary with results and metadata
        """
        start_time = time.time()
        
        if verbose:
            logger.info(f"Query: {query}")
            logger.info("\n" + "="*80)
            logger.info(f"QUERY: {query}")
            logger.info("="*80 + "\n")
        
        # Step 1: Process and embed query
        if verbose:
            logger.info("Step 1/3: Processing query...")
        
        query_start = time.time()
        processed_query = self.query_processor.process(query)
        query_time = time.time() - query_start
        
        if verbose:
            logger.info(f"Query embedded ({processed_query.embedding_dimensions} dimensions)")
            logger.success(f"✓ Query processed in {query_time:.2f}s")
        
        # Step 2: Retrieve relevant chunks
        if verbose:
            logger.info("Step 2/3: Retrieving relevant chunks...")
        
        retrieval_start = time.time()
        retrieved_chunks = self.retriever.retrieve(
            query_embedding=processed_query.embedding,
            top_k=self.top_k,
            min_score=self.min_score,
        )
        retrieval_time = time.time() - retrieval_start
        
        if verbose:
            logger.info(f"Retrieved {len(retrieved_chunks)} chunks")
            logger.success(f"✓ Retrieved {len(retrieved_chunks)} chunks in {retrieval_time:.2f}s\n")
            
            if retrieved_chunks:
                logger.info("Retrieved Chunks:")
                logger.info("-" * 80)
                for i, chunk in enumerate(retrieved_chunks, 1):
                    text_preview = chunk.text[:100] + "..." if len(chunk.text) > 100 else chunk.text
                    logger.info(f"[{i}] {chunk.chunk_id} (score: {chunk.score:.4f})")
                    logger.info(f"    {text_preview}\n")
        
        # Step 3: Generate answer
        if verbose:
            logger.info("Step 3/3: Generating answer...")
        
        generation_start = time.time()
        response = self.generator.generate(
            query=query,
            retrieved_chunks=retrieved_chunks,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        generation_time = time.time() - generation_start
        
        total_time = time.time() - start_time
        
        if verbose:
            logger.success(f"Answer generated ({response.token_usage['completion_tokens']} tokens)")
            logger.success(f"✓ Generated in {generation_time:.2f}s\n")
        
        # Display results
        logger.info("="*80)
        logger.info("ANSWER:")
        logger.info("="*80)
        logger.info(response.answer)
        logger.info("\n" + "="*80)
        logger.info("METADATA:")
        logger.info("="*80)
        logger.info(f"Model:            {response.model}")
        logger.info(f"Sources:          {len(response.sources)} chunks")
        logger.info(f"Prompt tokens:    {response.token_usage['prompt_tokens']}")
        logger.info(f"Response tokens:  {response.token_usage['completion_tokens']}")
        logger.info(f"Total tokens:     {response.token_usage['total_tokens']}")
        logger.info(f"Total time:       {total_time:.2f}s")
        
        if retrieved_chunks:
            logger.info(f"\nSource Chunk IDs:")
            for i, chunk_id in enumerate(response.sources, 1):
                score = next(c.score for c in retrieved_chunks if c.chunk_id == chunk_id)
                logger.info(f"  [{i}] {chunk_id} (similarity: {score:.4f})")
        
        logger.info("="*80 + "\n")
        
        # Return results for programmatic use
        return {
            "query": query,
            "answer": response.answer,
            "sources": response.sources,
            "model": response.model,
            "token_usage": response.token_usage,
            "retrieved_chunks": len(retrieved_chunks),
            "timing": {
                "query_processing": query_time,
                "retrieval": retrieval_time,
                "generation": generation_time,
                "total": total_time,
            },
        }


def interactive_mode(pipeline: QueryPipeline, args):
    """
    Run pipeline in interactive mode for multiple queries.
    
    Args:
        pipeline: QueryPipeline instance
        args: Command line arguments
    """
    logger.info("\n" + "="*80)
    logger.info("RAG Query Pipeline - Interactive Mode")
    logger.info("="*80)
    logger.info("Enter your questions (or 'quit' to exit)")
    logger.info("Settings:")
    logger.info(f"  - Top K: {pipeline.top_k}")
    logger.info(f"  - Min Score: {pipeline.min_score:.2f}")
    logger.info(f"  - Temperature: {args.temperature}")
    logger.info(f"  - Model: {pipeline.config.openai_chat_model}")
    logger.info("="*80 + "\n")
    
    query_count = 0
    total_tokens = 0
    
    while True:
        try:
            # Get user input
            query = input("\n🔍 Your question: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q']:
                logger.info("\nExiting...")
                break
            
            # Run pipeline
            result = pipeline.run(
                query=query,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                verbose=args.verbose,
            )
            
            query_count += 1
            total_tokens += result["token_usage"]["total_tokens"]
            
        except KeyboardInterrupt:
            logger.info("\n\nInterrupted. Exiting...")
            break
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            logger.error(f"\n❌ Error: {e}\n")
            continue
    
    # Show session statistics
    if query_count > 0:
        logger.info("\n" + "="*80)
        logger.info("SESSION STATISTICS")
        logger.info("="*80)
        logger.info(f"Queries processed: {query_count}")
        logger.info(f"Total tokens used: {total_tokens}")
        logger.info(f"Avg tokens/query:  {total_tokens / query_count:.1f}")
        logger.info("="*80 + "\n")


def main():
    """Main entry point for the query pipeline script."""
    parser = argparse.ArgumentParser(
        description="Run RAG query pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single query
  python scripts/run_query.py "What is Santiago's Personal Legend?"
  
  # Interactive mode
  python scripts/run_query.py
  
  # Custom retrieval parameters
  python scripts/run_query.py --top-k 5 --min-score 0.8 "What is X?"
  
  # Verbose output
  python scripts/run_query.py -v "Who is Fatima?"
        """
    )
    
    parser.add_argument(
        "query",
        nargs="?",
        help="Query to execute (if not provided, enters interactive mode)"
    )
    parser.add_argument(
        "-k", "--top-k",
        type=int,
        help="Number of chunks to retrieve (overrides config)"
    )
    parser.add_argument(
        "-s", "--min-score",
        type=float,
        help="Minimum similarity score (overrides config)"
    )
    parser.add_argument(
        "-t", "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for generation (default: 0.7, ignored for GPT-5)"
    )
    parser.add_argument(
        "-m", "--max-tokens",
        type=int,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed progress"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize pipeline
        pipeline = QueryPipeline(
            top_k=args.top_k,
            min_score=args.min_score,
        )
        
        # Single query mode or interactive mode
        if args.query:
            # Single query
            pipeline.run(
                query=args.query,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                verbose=args.verbose,
            )
        else:
            # Interactive mode
            interactive_mode(pipeline, args)
    
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        logger.error(f"\n❌ Pipeline failed: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
