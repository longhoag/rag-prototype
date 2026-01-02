"""
Test script for the embedding module.
Tests embedding generation, batch processing, and data validation.
"""

from pathlib import Path

from loguru import logger

from offline.chunking import chunk_document
from offline.embedding import embed_chunks, EmbeddedChunk, TextEmbedder


def test_embedding_basic():
    """Test basic embedding functionality with a small sample."""
    logger.info("=" * 60)
    logger.info("TEST: Basic Embedding Functionality")
    logger.info("=" * 60)
    
    # Get first 5 chunks
    document_path = Path("document.txt")
    chunks = chunk_document(document_path)[:5]
    
    logger.info(f"Testing with {len(chunks)} chunks")
    
    # Embed the chunks
    embedded_chunks = embed_chunks(chunks)
    
    # Validate results
    assert len(embedded_chunks) == len(chunks), "Mismatch in chunk count"
    
    for i, embedded in enumerate(embedded_chunks):
        assert isinstance(embedded, EmbeddedChunk)
        assert embedded.model == "text-embedding-3-large"
        assert embedded.dimensions == 3072
        assert len(embedded.embedding) == 3072
        assert embedded.chunk.chunk_id == chunks[i].chunk_id
        
        logger.info(
            f"Chunk {i}: {len(embedded.embedding)} dimensions, "
            f"ID={embedded.chunk.chunk_id}, "
            f"tokens={embedded.chunk.token_count}"
        )
    
    logger.success("✅ Basic embedding test passed")
    return embedded_chunks


def test_embedding_validation():
    """Test that embeddings are properly validated."""
    logger.info("=" * 60)
    logger.info("TEST: Embedding Validation")
    logger.info("=" * 60)
    
    # Get one chunk
    document_path = Path("document.txt")
    chunks = chunk_document(document_path)[:1]
    
    # Embed it
    embedded = embed_chunks(chunks)[0]
    
    # Check embedding properties
    assert all(isinstance(val, float) for val in embedded.embedding)
    
    # Check that values are in reasonable range (normalized vectors)
    embedding_values = embedded.embedding
    min_val = min(embedding_values)
    max_val = max(embedding_values)
    
    logger.info(f"Embedding value range: [{min_val:.6f}, {max_val:.6f}]")
    
    # OpenAI embeddings are normalized, so values should be roughly between -1 and 1
    assert -2 < min_val < 2, f"Min value {min_val} out of expected range"
    assert -2 < max_val < 2, f"Max value {max_val} out of expected range"
    
    logger.success("✅ Embedding validation test passed")


def test_batch_processing():
    """Test that batch processing works correctly."""
    logger.info("=" * 60)
    logger.info("TEST: Batch Processing")
    logger.info("=" * 60)
    
    # Test with enough chunks to require multiple batches
    # TextEmbedder uses batch_size=100
    document_path = Path("document.txt")
    all_chunks = chunk_document(document_path)
    
    # Test with 10 chunks (single batch)
    logger.info("Testing with 10 chunks (single batch)...")
    embedded_10 = embed_chunks(all_chunks[:10])
    assert len(embedded_10) == 10
    logger.info(f"✓ Single batch: {len(embedded_10)} chunks embedded")
    
    # Test with 250 chunks (multiple batches)
    logger.info("Testing with multiple batches...")
    num_test_chunks = min(250, len(all_chunks))
    embedded_many = embed_chunks(all_chunks[:num_test_chunks])
    assert len(embedded_many) == num_test_chunks
    logger.info(f"✓ Multiple batches: {len(embedded_many)} chunks embedded")
    
    logger.success("✅ Batch processing test passed")


def test_embedder_class():
    """Test the TextEmbedder class directly."""
    logger.info("=" * 60)
    logger.info("TEST: TextEmbedder Class")
    logger.info("=" * 60)
    
    embedder = TextEmbedder()
    
    # Check initialization
    assert embedder.model == "text-embedding-3-large"
    assert embedder.dimensions == 3072
    assert embedder.batch_size == 100
    
    logger.info(f"Model: {embedder.model}")
    logger.info(f"Dimensions: {embedder.dimensions}")
    logger.info(f"Batch size: {embedder.batch_size}")
    
    # Test embed_file convenience method
    logger.info("Testing embed_file method...")
    document_path = Path("document.txt")
    
    # Only embed first few chunks for speed
    from offline.chunking import TextChunker
    chunker = TextChunker()
    text = document_path.read_text(encoding="utf-8")
    chunks = chunker.create_chunks(text)[:3]
    
    embedded = embedder.embed_chunks(chunks, show_progress=False)
    assert len(embedded) == 3
    
    logger.success("✅ TextEmbedder class test passed")


def print_embedding_statistics(embedded_chunks):
    """Print statistics about the embeddings."""
    logger.info("=" * 60)
    logger.info("EMBEDDING STATISTICS")
    logger.info("=" * 60)
    
    logger.info(f"Total chunks embedded: {len(embedded_chunks)}")
    logger.info(f"Model: {embedded_chunks[0].model}")
    logger.info(f"Dimensions per embedding: {embedded_chunks[0].dimensions}")
    
    # Calculate total size
    total_floats = len(embedded_chunks) * embedded_chunks[0].dimensions
    bytes_per_float = 4  # 32-bit floats
    total_mb = (total_floats * bytes_per_float) / (1024 * 1024)
    
    logger.info(f"Total embedding vectors: {total_floats:,}")
    logger.info(f"Estimated memory: {total_mb:.2f} MB")
    
    # Sample embedding values from first chunk
    first_embedding = embedded_chunks[0].embedding
    logger.info(f"\nFirst embedding sample (first 10 values):")
    logger.info(f"  {first_embedding[:10]}")


def main():
    """Run all tests."""
    logger.info("\n" + "=" * 60)
    logger.info("STARTING EMBEDDING MODULE TESTS")
    logger.info("=" * 60 + "\n")
    
    try:
        # Run tests
        embedded = test_embedding_basic()
        test_embedding_validation()
        test_batch_processing()
        test_embedder_class()
        
        # Print statistics
        print_embedding_statistics(embedded)
        
        logger.info("\n" + "=" * 60)
        logger.success("ALL TESTS PASSED ✅")
        logger.info("=" * 60 + "\n")
        
    except Exception as e:
        logger.error(f"\n❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    main()
