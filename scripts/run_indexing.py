#!/usr/bin/env python3
"""
Offline Pipeline Orchestrator

Runs the complete offline pipeline: chunking → embedding → indexing.
Processes the document and uploads all vectors to Pinecone.
"""

import sys
from pathlib import Path

from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import get_config
from offline.chunking import chunk_file
from offline.embedding import embed_chunks
from offline.indexing import index_embedded_chunks


def run_offline_pipeline(document_path: Path) -> None:
    """
    Execute the complete offline pipeline for a document.
    
    Args:
        document_path: Path to the document to process
    """
    logger.info("=" * 70)
    logger.info("OFFLINE PIPELINE: Starting full document processing")
    logger.info("=" * 70)
    
    # Load configuration
    config = get_config()
    logger.info(f"Document: {document_path}")
    logger.info(f"Embedding Model: {config.openai_embedding_model}")
    logger.info(f"Pinecone Index: {config.pinecone_index_name}")
    logger.info("")
    
    # Step 1: Chunking
    logger.info("=" * 70)
    logger.info("STEP 1: Chunking Document")
    logger.info("=" * 70)
    
    chunks = chunk_file(document_path)
    
    if not chunks:
        logger.error("No chunks created from document!")
        return
    
    total_tokens = sum(chunk.token_count for chunk in chunks)
    avg_tokens = total_tokens / len(chunks)
    
    logger.success(f"✅ Created {len(chunks)} chunks")
    logger.info(f"   Total tokens: {total_tokens:,}")
    logger.info(f"   Average tokens per chunk: {avg_tokens:.1f}")
    logger.info(f"   Token range: {min(c.token_count for c in chunks)} - {max(c.token_count for c in chunks)}")
    logger.info("")
    
    # Step 2: Embedding
    logger.info("=" * 70)
    logger.info("STEP 2: Generating Embeddings")
    logger.info("=" * 70)
    
    embedded_chunks = embed_chunks(chunks)
    
    if not embedded_chunks:
        logger.error("No embeddings created!")
        return
    
    logger.success(f"✅ Generated {len(embedded_chunks)} embeddings")
    logger.info(f"   Dimensions: {embedded_chunks[0].dimensions}")
    logger.info(f"   Model: {embedded_chunks[0].model}")
    logger.info("")
    
    # Step 3: Indexing
    logger.info("=" * 70)
    logger.info("STEP 3: Uploading to Pinecone")
    logger.info("=" * 70)
    
    document_source = document_path.name
    indexed_chunks = index_embedded_chunks(
        embedded_chunks,
        document_source=document_source,
    )
    
    if not indexed_chunks:
        logger.error("No chunks indexed!")
        return
    
    logger.success(f"✅ Indexed {len(indexed_chunks)} chunks to Pinecone")
    logger.info("")
    
    # Final Summary
    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETE ✅")
    logger.info("=" * 70)
    logger.info(f"Document: {document_path.name}")
    logger.info(f"Total Chunks: {len(chunks)}")
    logger.info(f"Total Tokens: {total_tokens:,}")
    logger.info(f"Vector Dimension: {embedded_chunks[0].dimensions}")
    logger.info(f"Pinecone Index: {config.pinecone_index_name}")
    logger.info("")
    logger.success("The document is now ready for querying!")
    logger.info("=" * 70)


def main():
    """Main entry point for the offline pipeline script."""
    # Default to document.txt in project root
    project_root = Path(__file__).parent.parent
    document_path = project_root / "document.txt"
    
    if not document_path.exists():
        logger.error(f"Document not found: {document_path}")
        sys.exit(1)
    
    try:
        run_offline_pipeline(document_path)
    except KeyboardInterrupt:
        logger.warning("\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Pipeline failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
