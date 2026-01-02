"""
Test script for Pinecone indexing module.
Tests vector upload, metadata handling, and batch processing.
"""

from pathlib import Path
import sys

from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from offline.chunking import chunk_document
from offline.embedding import embed_chunks
from offline.indexing import PineconeIndexer, index_embedded_chunks


def test_indexing_basic():
    """Test basic indexing with a small sample."""
    logger.info("=" * 60)
    logger.info("TEST: Basic Indexing Functionality")
    logger.info("=" * 60)
    
    # Create chunks from first part of document
    document_path = project_root / "document.txt"
    all_chunks = chunk_document(document_path)
    
    # Use first 3 chunks for testing
    test_chunks = all_chunks[:3]
    logger.info(f"Using {len(test_chunks)} chunks for testing")
    
    # Embed the chunks
    embedded_chunks = embed_chunks(test_chunks)
    logger.info(f"Embedded {len(embedded_chunks)} chunks")
    
    # Index the embeddings
    indexer = PineconeIndexer()
    indexed_chunks = indexer.index_embedded_chunks(
        embedded_chunks,
        document_source="document.txt",
        show_progress=True,
    )
    
    # Validate results
    assert len(indexed_chunks) == len(embedded_chunks), "Mismatch in indexed chunk count"
    
    for i, indexed in enumerate(indexed_chunks):
        assert indexed.chunk_id == str(embedded_chunks[i].chunk.chunk_id)
        assert indexed.vector_id == str(embedded_chunks[i].chunk.chunk_id)
        assert indexed.token_count == embedded_chunks[i].chunk.token_count
        logger.info(
            f"✓ Chunk {i}: ID={indexed.chunk_id}, "
            f"tokens={indexed.token_count}, "
            f"preview='{indexed.text_preview[:50]}...'"
        )
    
    logger.success("✅ Basic indexing test PASSED")
    return indexed_chunks


def test_index_stats():
    """Test retrieving index statistics."""
    logger.info("=" * 60)
    logger.info("TEST: Index Statistics")
    logger.info("=" * 60)
    
    indexer = PineconeIndexer()
    stats = indexer.get_index_stats()
    
    logger.info(f"Index statistics:")
    logger.info(f"  - Total vectors: {stats.get('total_vector_count', 'N/A')}")
    logger.info(f"  - Dimension: {stats.get('dimension', 'N/A')}")
    logger.info(f"  - Namespaces: {stats.get('namespaces', {})}")
    
    logger.success("✅ Index stats test PASSED")
    return stats


def test_batch_processing():
    """Test batch processing with more chunks."""
    logger.info("=" * 60)
    logger.info("TEST: Batch Processing")
    logger.info("=" * 60)
    
    # Create chunks
    document_path = project_root / "document.txt"
    all_chunks = chunk_document(document_path)
    
    # Use first 10 chunks for batch testing
    test_chunks = all_chunks[:10]
    logger.info(f"Testing batch upload with {len(test_chunks)} chunks")
    
    # Embed chunks
    embedded_chunks = embed_chunks(test_chunks)
    
    # Index with batch processing
    indexed_chunks = index_embedded_chunks(
        embedded_chunks,
        document_source="document.txt",
    )
    
    assert len(indexed_chunks) == len(test_chunks), "Batch processing count mismatch"
    
    logger.success(f"✅ Batch processing test PASSED ({len(indexed_chunks)} chunks)")
    return indexed_chunks


def test_indexer_class():
    """Test PineconeIndexer class methods."""
    logger.info("=" * 60)
    logger.info("TEST: PineconeIndexer Class")
    logger.info("=" * 60)
    
    # Initialize indexer
    indexer = PineconeIndexer()
    
    logger.info(f"✓ Index name: {indexer.index_name}")
    logger.info(f"✓ Batch size: {indexer.batch_size}")
    
    # Test index stats
    stats = indexer.get_index_stats()
    assert stats is not None, "Failed to get index stats"
    
    logger.success("✅ PineconeIndexer class test PASSED")


def print_indexing_summary(indexed_chunks):
    """Print summary of indexed chunks."""
    logger.info("=" * 60)
    logger.info("INDEXING SUMMARY")
    logger.info("=" * 60)
    
    total_chunks = len(indexed_chunks)
    total_tokens = sum(chunk.token_count for chunk in indexed_chunks)
    avg_tokens = total_tokens / total_chunks if total_chunks > 0 else 0
    
    logger.info(f"Total chunks indexed: {total_chunks}")
    logger.info(f"Total tokens: {total_tokens:,}")
    logger.info(f"Average tokens per chunk: {avg_tokens:.1f}")
    
    logger.info("\nFirst 3 indexed chunks:")
    for i, chunk in enumerate(indexed_chunks[:3]):
        logger.info(
            f"  {i+1}. ID={chunk.vector_id}, tokens={chunk.token_count}, "
            f"preview='{chunk.text_preview[:40]}...'"
        )


def main():
    """Run all indexing tests."""
    logger.info("Starting Pinecone indexing tests...")
    logger.info("")
    
    try:
        # Test 1: Basic indexing
        indexed_basic = test_indexing_basic()
        logger.info("")
        
        # Test 2: Index stats
        test_index_stats()
        logger.info("")
        
        # Test 3: Batch processing
        indexed_batch = test_batch_processing()
        logger.info("")
        
        # Test 4: Indexer class
        test_indexer_class()
        logger.info("")
        
        # Print summary
        print_indexing_summary(indexed_batch)
        
        logger.info("")
        logger.success("=" * 60)
        logger.success("ALL TESTS PASSED ✅")
        logger.success("=" * 60)
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
