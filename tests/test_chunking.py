#!/usr/bin/env python3
"""
Test script for chunking module.
Run with: poetry run python -m pytest tests/test_chunking.py -v
Or quick test: poetry run python tests/test_chunking.py
"""

import sys
from pathlib import Path
from loguru import logger

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from offline.chunking import chunk_document


def test_chunking():
    """Test the chunking implementation with document.txt"""
    
    # Check if document exists
    doc_path = Path(__file__).parent.parent / "document.txt"
    if not doc_path.exists():
        logger.error("document.txt not found in project root")
        return False
    
    logger.info("=" * 60)
    logger.info("Testing chunking implementation")
    logger.info("=" * 60)
    
    try:
        # Chunk the document
        chunks = chunk_document(str(doc_path))
        
        if not chunks:
            logger.error("No chunks created!")
            return False
        
        # Print summary
        logger.info(f"\n✅ Successfully created {len(chunks)} chunks\n")
        
        # Calculate statistics
        token_counts = [c.token_count for c in chunks]
        avg_tokens = sum(token_counts) / len(token_counts)
        min_tokens = min(token_counts)
        max_tokens = max(token_counts)
        total_tokens = sum(token_counts)
        
        logger.info("📊 Chunk Statistics:")
        logger.info(f"  • Total chunks: {len(chunks)}")
        logger.info(f"  • Total tokens: {total_tokens:,}")
        logger.info(f"  • Average tokens per chunk: {avg_tokens:.1f}")
        logger.info(f"  • Min tokens: {min_tokens}")
        logger.info(f"  • Max tokens: {max_tokens}")
        
        # Validate chunks meet requirements
        logger.info("\n🔍 Validation:")
        chunks_in_range = sum(1 for c in chunks if 400 <= c.token_count <= 800)
        logger.info(f"  • Chunks within 400-800 tokens: {chunks_in_range}/{len(chunks)} ({chunks_in_range/len(chunks)*100:.1f}%)")
        
        # Show sample chunks
        logger.info("\n📝 Sample Chunks:\n")
        
        # First chunk
        chunk = chunks[0]
        preview = chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text
        logger.info(f"Chunk 0 ({chunk.token_count} tokens):")
        logger.info(f"  {preview}\n")
        
        # Middle chunk
        if len(chunks) > 2:
            chunk = chunks[len(chunks) // 2]
            preview = chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text
            logger.info(f"Chunk {chunk.chunk_id} ({chunk.token_count} tokens):")
            logger.info(f"  {preview}\n")
        
        # Last chunk
        chunk = chunks[-1]
        preview = chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text
        logger.info(f"Chunk {len(chunks)-1} ({chunk.token_count} tokens):")
        logger.info(f"  {preview}\n")
        
        # Check for overlap between consecutive chunks
        if len(chunks) > 1:
            logger.info("🔗 Overlap Analysis:")
            for i in range(min(3, len(chunks) - 1)):
                chunk1 = chunks[i]
                chunk2 = chunks[i + 1]
                
                # Find common text
                # Simple approach: check if last part of chunk1 appears in start of chunk2
                chunk1_end = chunk1.text[-200:]  # Last 200 chars
                chunk2_start = chunk2.text[:200]  # First 200 chars
                
                has_overlap = any(
                    chunk1_end[j:j+50] in chunk2_start 
                    for j in range(0, len(chunk1_end) - 50, 10)
                )
                
                status = "✅" if has_overlap else "⚠️"
                logger.info(f"  {status} Chunks {i} → {i+1}: {'Overlap detected' if has_overlap else 'No overlap detected'}")
        
        logger.info("\n" + "=" * 60)
        logger.success("Chunking test completed successfully!")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_chunking()
    sys.exit(0 if success else 1)
