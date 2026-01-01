# Copilot Instructions for RAG Prototype

## Project Overview
This is a full Retrieval-Augmented Generation (RAG) pipeline project in early development. The architecture consists of offline indexing and online query stages using OpenAI embeddings, Pinecone vector database, and GPT-5 for generation.

## Architecture

### Offline Pipeline (Indexing)
1. **Source Data**: [document.txt](../document.txt) contains The Alchemist text - no preprocessing needed
2. **Chunking**: Sentence-split strategy with 400-800 tokens per chunk, 80-150 token overlap
3. **Embedding**: OpenAI `text-embedding-3-large` model
4. **Vector Store**: Pinecone with HNSW index (Pinecone manages HNSW parameters automatically)

### Online Pipeline (Query)
1. Embed query with same `text-embedding-3-large` model
2. Search Pinecone HNSW index for top-k=10 chunks
3. Construct prompt from query + retrieved chunks
4. Generate answer using OpenAI `gpt-5`

## Development Standards

### Dependencies & Environment
- **Use Poetry** for all dependency management (not pip/requirements.txt)
- **Environment Variables**: Store all credentials in `.env` file:
  - `OPENAI_API_KEY`
  - `PINECONE_API_KEY`
  - `PINECONE_ENVIRONMENT`
- **Config Module**: Create `config.py` to load environment variables consistently across the project

### Code Style
- **Logging**: Always use `loguru` logger instead of `print()` statements
- **Structure**: Separate offline and online pipeline code into distinct modules/scripts
- **Error Handling**: Implement retry logic for API calls:
  - Use `tenacity` library for exponential backoff with jitter
  - OpenAI: Retry on rate limits (429) and server errors (500-599), max 3 attempts
  - Pinecone: Retry on transient failures, max 3 attempts with 2s base delay
  - Log all retry attempts with context

## Implementation Guidelines

When implementing features:
1. Start with `config.py` to establish the configuration pattern
2. Build offline pipeline first (must run before online queries work)
3. Implement sentence-based chunking with `nltk.sent_tokenize()` for sentence splitting
4. Use `tiktoken` (encoding: `cl100k_base`) to count tokens and enforce 400-800 token chunks
5. Test embedding generation with small samples before processing full document
6. Initialize Pinecone index with appropriate dimensions (1536 for text-embedding-3-large)

### Chunking Workflow
1. **Read document.txt** → Load full text
2. **Split with nltk** → `nltk.sent_tokenize()` to get list of sentences
3. **Count with tiktoken** → Combine sentences until you hit 400-800 tokens
4. **Add overlap** → Include last 80-150 tokens of previous chunk
5. **Embed with OpenAI** → Use tenacity for retry logic if API fails
6. **Store in Pinecone** → Use tenacity for retry logic if connection fails

## Testing Strategy

### Unit Tests
- **Chunking**: Test sentence splitting, token counting, overlap logic with edge cases
  - Empty text, single sentence, very long sentences (>800 tokens)
  - Verify overlap boundaries don't duplicate or lose content
- **Config**: Test environment variable loading with missing/invalid values
- **Utilities**: Test token counting accuracy against tiktoken reference

### Integration Tests
- **Offline Pipeline**: End-to-end test with small sample (3-5 paragraphs)
  - Verify chunks are created correctly and uploaded to Pinecone
  - Check vector dimensions and metadata preservation
- **Online Pipeline**: Test query flow with known test queries
  - Mock OpenAI/Pinecone or use test indexes
  - Verify retrieval returns expected chunks for known content
- **API Resilience**: Test retry logic with simulated failures

## File Organization (Not Yet Created)
Expected structure:
```
config.py           # Environment variable loader
offline/            # Indexing pipeline
  chunking.py
  embedding.py
  indexing.py
online/             # Query pipeline
  query.py
  retrieval.py
  generation.py
docs/               # Documentation markdown files
pyproject.toml      # Poetry dependencies
.env                # API credentials (never commit)
```

## Key Considerations
- The document is ~226KB of text - monitor token usage when chunking
- Pinecone serverless indexes use HNSW automatically; no manual tuning of M/efConstruction needed
- Top-k=10 balances context richness with prompt token limits
- Chunk overlap (80-150 tokens) ensures context continuity across boundaries
- Sentence boundaries preserve semantic coherence better than fixed-length splits
