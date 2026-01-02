# RAG Prototype Setup Guide

This guide walks you through setting up the RAG pipeline project from scratch, including API configuration and implementation steps.

## Prerequisites

- Python 3.10 or higher
- Git
- A code editor (VS Code, PyCharm, etc.)
- OpenAI account
- Pinecone account

---

## Part 1: OpenAI API Setup

### 1.1 Create OpenAI Account and Get API Key

1. **Sign up for OpenAI**:
   - Go to [https://platform.openai.com](https://platform.openai.com)
   - Click "Sign up" and create an account
   - Verify your email address

2. **Access API Keys**:
   - Log in to [https://platform.openai.com](https://platform.openai.com)
   - Click on your profile icon (top right)
   - Select "View API keys" or go to [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)

3. **Create a New API Key**:
   - Click "Create new secret key"
   - Give it a descriptive name (e.g., "RAG Prototype")
   - **Important**: Copy the key immediately - you won't be able to see it again!
   - Store it securely (we'll add it to `.env` later)

4. **Add Billing Information** (Required):
   - Go to Settings → Billing ([https://platform.openai.com/account/billing](https://platform.openai.com/account/billing))
   - Add a payment method
   - Set usage limits to control costs (recommended: start with $10-20)

5. **Verify Access to Models**:
   - Check that you have access to:
     - `text-embedding-3-large` (for embeddings)
     - `gpt-4` or `gpt-4-turbo` (Note: GPT-5 may not be available yet; use latest GPT-4 variant)
   - Go to Settings → Limits to see available models

### 1.2 Estimate Costs

For this project with The Alchemist text (~226KB):
- **Embeddings** (`text-embedding-3-large`): ~$0.13 per 1M tokens
  - Estimated: ~$0.01-0.02 for the entire document
  - Note: Using 3072 dimensions (full quality) vs 1536 dimensions doesn't affect API costs
- **Generation** (`gpt-4-turbo`): ~$10 per 1M input tokens, ~$30 per 1M output tokens
  - Estimated: ~$0.10-0.50 per query (depending on retrieved chunks)

**Budget recommendation**: $5-10 for development and testing

---

## Part 2: Pinecone Setup

### 2.1 Create Pinecone Account

1. **Sign up for Pinecone**:
   - Go to [https://www.pinecone.io](https://www.pinecone.io)
   - Click "Start Free" or "Sign Up"
   - Create an account (can use Google/GitHub OAuth)

2. **Choose a Plan**:
   - Select "Starter" (Free tier) for development
   - Free tier includes:
     - 1 pod with 100K vectors
     - Sufficient for this prototype

### 2.2 Create an Index

1. **Access the Console**:
   - Log in to [https://app.pinecone.io](https://app.pinecone.io)
   - You'll see the Pinecone dashboard

2. **Create a New Index**:
   - Click "Create Index" button
   - **Index Name**: `rag-alchemist` (or your preferred name)
   
3. **Configure Index Settings**:
   
   **Recommendation**: Use **Model Preset** with **3072 dimensions** for this project. It provides the best embedding quality and is straightforward to configure. The document is small (~226KB), so storage costs are minimal.
   
   You have two options:
   
   **Option A: Use Model Preset (Recommended)**
   - Select "Model Preset"
   - Choose `text-embedding-3-large` from the dropdown
   - This provides the following configuration:
     - **Modality**: Text
     - **Vector Type**: Dense
     - **Max Input**: 8,191 tokens
     - **Dimensions**: Choose from `256`, `1024`, or `3072`
       - **Select `3072`** for highest quality (default for text-embedding-3-large)
       - Note: `1536` is the default dimension, but it's not listed as a preset option
       - Lower dimensions (256, 1024) trade quality for storage/cost
     - **Metric**: `cosine` or `dotproduct` (choose `cosine` for this project)
   
   **Important Note**: This model does not support integrated inference. You must manage your own embeddings:
   - Create OpenAI API key (covered in Part 1)
   - Generate embeddings using OpenAI API
   - Upsert embeddings into your Pinecone index (covered in Phase 4)
   
   **Option B: Custom Settings**
   - Select "Custom"
   - **Vector Type**: Choose `dense` (for text embeddings)
   - **Dimensions**: Enter `3072` (for text-embedding-3-large full quality)
     - Or `1536` if using reduced dimensions
   - **Metric**: Choose `cosine` (recommended for text similarity)

4. **Select Capacity Mode**:
   - Choose **"On Demand"** (serverless, pay-per-use)
   - Benefits:
     - No need to manage pod types or replicas
     - Automatically scales with usage
     - More cost-effective for development and variable workloads
   - Alternative: "Pod-based" (for consistent high-volume workloads)

5. **Cloud and Region**:
   - Select cloud provider and region closest to your location for lower latency
   - Common options: `us-east-1`, `us-west-2`, `eu-west-1`

6. **Create Index**:
   - Review your settings
   - Click "Create Index"

7. **Wait for Index Creation**:
   - Index initialization takes 1-2 minutes
   - Status will change from "Initializing" to "Ready"

### 2.3 Get API Key and Environment

1. **Get API Key**:
   - In the Pinecone console, click on "API Keys" in the left sidebar
   - You'll see your API key displayed
   - Click "Copy" to copy the key
   - Store it securely (we'll add it to `.env` later)

2. **Get Environment Name**:
   - The environment is shown in the console (e.g., `us-east-1-aws`, `gcp-starter`)
   - You can also find it in the index details page
   - Note this down - you'll need it for configuration

---

## Part 3: Project Setup

### 3.1 Clone and Initialize Repository

```bash
# Navigate to your workspace
cd /path/to/your/workspace

# Clone the repository (if not already done)
git clone <your-repo-url>
cd rag-prototype

# Verify project structure
ls -la
```

### 3.2 Install Poetry (if not already installed)

```bash
# macOS/Linux
curl -sSL https://install.python-poetry.org | python3 -

# Or via pip
pip install poetry

# Verify installation
poetry --version
```

### 3.3 Configure Environment Variables

1. **Create `.env` file**:
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` file** with your API credentials:
   ```bash
   # Open in your preferred editor
   nano .env
   # or
   code .env
   ```

3. **Add your API keys**:
   ```env
   # OpenAI API Configuration
   OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

   # Pinecone Configuration
   PINECONE_API_KEY=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
   PINECONE_ENVIRONMENT=us-east-1-aws
   PINECONE_INDEX_NAME=rag-alchemist
   ```

4. **Save and verify** `.env` is in `.gitignore`:
   ```bash
   # Check that .env won't be committed
   git status
   # .env should NOT appear in the list
   ```

### 3.4 Install Dependencies

```bash
# Install all dependencies with Poetry
poetry install

# Verify Python version
poetry run python --version  # Should be 3.10+
```

### 3.5 Download NLTK Data

```bash
# Run Python to download NLTK punkt tokenizer
poetry run python -c "import nltk; nltk.download('punkt')"
```

---

## Part 4: Implementation Steps

### Phase 1: Configuration Module (config.py)

**Goal**: Centralized environment variable loading

**Implementation checklist**:
- [ ] Import `os` and `python-dotenv`
- [ ] Load `.env` file using `load_dotenv()`
- [ ] Create configuration class or variables for:
  - `OPENAI_API_KEY`
  - `PINECONE_API_KEY`
  - `PINECONE_ENVIRONMENT`
  - `PINECONE_INDEX_NAME`
- [ ] Add validation to ensure all required variables are set
- [ ] Add error handling for missing variables

**Test**: Run `python config.py` to verify all variables load correctly

---

### Phase 2: Offline Pipeline - Chunking (offline/chunking.py)

**Goal**: Split document into 400-800 token chunks with 80-150 token overlap

**Implementation checklist**:
- [ ] Import `nltk.sent_tokenize()` and `tiktoken`
- [ ] Load `cl100k_base` encoding
- [ ] Create function to read `document.txt`
- [ ] Implement sentence splitting
- [ ] Implement token counting per sentence
- [ ] Implement chunk building logic:
  - Combine sentences until reaching 400-800 tokens
  - Track chunk boundaries
- [ ] Implement overlap logic:
  - Get last 80-150 tokens from previous chunk
  - Prepend to next chunk
- [ ] Add loguru logging for progress tracking
- [ ] Return list of chunks with metadata (position, token count)

**Test**:
- Test with small sample text (2-3 paragraphs)
- Verify chunk sizes are within 400-800 tokens
- Verify overlaps are 80-150 tokens
- Check no content is lost or duplicated

---

### Phase 3: Offline Pipeline - Embedding (offline/embedding.py)

**Goal**: Generate embeddings for all chunks using OpenAI API

**Implementation checklist**:
- [ ] Import OpenAI client and tenacity
- [ ] Initialize OpenAI client with API key from config
- [ ] Create retry decorator with tenacity:
  - Stop after 3 attempts
  - Exponential backoff with jitter
  - Retry on rate limit (429) and server errors (500-599)
- [ ] Implement function to generate embedding for single chunk:
  - Use `text-embedding-3-large` model
  - Default: Return 3072-dimensional vector (full quality)
  - Optional: Use `dimensions` parameter to reduce to 1536 or lower
  - Match dimension with your Pinecone index configuration
- [ ] Implement batch processing for multiple chunks
- [ ] Add loguru logging for each API call and retry
- [ ] Add error handling for API failures

**Test**:
- Test with small sample text (2-3 paragraphs)
- Verify chunk sizes are within 400-800 tokens
- Verify overlaps are 80-150 tokens
- Check no content is lost or duplicated

---

### Phase 3: Offline Pipeline - Embedding (offline/embedding.py)

**Goal**: Generate embeddings for all chunks using OpenAI API

**Implementation checklist**:
- [ ] Import OpenAI client and tenacity
- [ ] Initialize OpenAI client with API key from config
- [ ] Create retry decorator with tenacity:
  - Stop after 3 attempts
  - Exponential backoff with jitter
  - Retry on rate limit (429) and server errors (500-599)
- [ ] Implement function to generate embedding for single chunk:
  - Use `text-embedding-3-large` model
  - Default: Return 3072-dimensional vector (full quality)
  - Optional: Use `dimensions` parameter to reduce to 1536 or lower
  - Match dimension with your Pinecone index configuration
- [ ] Implement batch processing for multiple chunks
- [ ] Add loguru logging for each API call and retry
- [ ] Add error handling for API failures

**Test**:
- Test with 1-2 chunks first
- Verify embedding dimensions (should be 3072 by default, or 1536 if using dimension reduction)
- Simulate rate limit by making rapid requests
- Check retry logic works

---

### Phase 4: Offline Pipeline - Indexing (offline/indexing.py)

**Goal**: Store embeddings in Pinecone with metadata

**Implementation checklist**:
- [ ] Import Pinecone client and tenacity
- [ ] Initialize Pinecone with API key and environment
- [ ] Connect to your index
- [ ] Create retry decorator for Pinecone operations
- [ ] Implement upsert function:
  - Create vector IDs (e.g., `chunk_0`, `chunk_1`)
  - Prepare vectors with metadata (text, position, token_count)
  - Batch upsert to Pinecone (recommended batch size: 100)
- [ ] Add loguru logging for upload progress
- [ ] Implement index statistics check (verify vector count)

**Test**:
- Upload 5-10 test vectors
- Check Pinecone console to verify vectors appear
- Verify metadata is stored correctly
- Test with full document

---

### Phase 5: Create Offline Pipeline Orchestrator

**Goal**: Run the complete indexing pipeline

**Create file**: `offline/run_indexing.py`

**Implementation checklist**:
- [ ] Import all offline modules (chunking, embedding, indexing)
- [ ] Create main function that:
  1. Reads and chunks document
  2. Generates embeddings for all chunks
  3. Uploads to Pinecone
- [ ] Add progress tracking with loguru
- [ ] Add total time tracking
- [ ] Make it executable: `if __name__ == "__main__":`

**Run**:
```bash
poetry run python offline/run_indexing.py
```

---

### Phase 6: Online Pipeline - Retrieval (online/retrieval.py)

**Goal**: Search Pinecone for relevant chunks

**Implementation checklist**:
- [ ] Import Pinecone client and tenacity
- [ ] Initialize Pinecone connection
- [ ] Create retry decorator
- [ ] Implement query function:
  - Accept query embedding (1536-dimensional vector)
  - Search with `top_k=10`
  - Return matches with scores and metadata
- [ ] Add loguru logging
- [ ] Format results for easy consumption

**Test**:
- Create a test embedding
- Query Pinecone
- Verify top-10 results are returned
- Check metadata is included

---

### Phase 7: Online Pipeline - Query Processing (online/query.py)

**Goal**: Embed user queries

**Implementation checklist**:
- [ ] Import OpenAI client and tenacity
- [ ] Initialize OpenAI client
- [ ] Create retry decorator
- [ ] Implement query embedding function:
  - Use same `text-embedding-3-large` model
  - Use same dimensions as indexing (3072 or 1536)
  - Return vector matching Pinecone index dimensions
- [ ] Add validation for empty queries

**Test**:
- Test with sample questions about The Alchemist
- Verify embedding dimensions

---

### Phase 8: Online Pipeline - Generation (online/generation.py)

**Goal**: Generate answers using GPT-4 with retrieved context

**Implementation checklist**:
- [ ] Import OpenAI client and tenacity
- [ ] Initialize OpenAI client
- [ ] Create retry decorator
- [ ] Implement prompt construction:
  - System prompt (define assistant behavior)
  - Context: Retrieved chunks
  - User query
- [ ] Implement GPT-4 API call:
  - Use `gpt-4-turbo` model (or latest available)
  - Set appropriate temperature (0.7 for balanced responses)
  - Add max_tokens limit
- [ ] Add loguru logging
- [ ] Return generated answer with metadata (tokens used, cost estimate)

**Test**:
- Test with mock retrieved chunks
- Verify answers are coherent and relevant

---

### Phase 9: Create Online Pipeline Orchestrator

**Goal**: Run the complete query pipeline

**Create file**: `online/run_query.py`

**Implementation checklist**:
- [ ] Import all online modules (query, retrieval, generation)
- [ ] Create main function that:
  1. Accepts user query (command-line or input())
  2. Embeds query
  3. Retrieves relevant chunks from Pinecone
  4. Generates answer with GPT-4
  5. Displays answer
- [ ] Add loguru logging for each step
- [ ] Add token usage tracking
- [ ] Make it executable

**Run**:
```bash
poetry run python online/run_query.py
```

---

### Phase 10: Testing

**Create test files**:
- `tests/test_chunking.py`
- `tests/test_embedding.py`
- `tests/test_retrieval.py`
- `tests/test_generation.py`

**Run tests**:
```bash
poetry run pytest tests/ -v
```

---

## Part 5: Usage Examples

### Running the Offline Pipeline (Indexing)

```bash
# Run indexing with poetry
poetry run python offline/run_indexing.py

# Expected output:
# INFO: Reading document.txt...
# INFO: Found 294 sentences
# INFO: Created 45 chunks (avg 650 tokens)
# INFO: Generating embeddings... (1/45)
# ...
# INFO: Uploading to Pinecone...
# SUCCESS: Indexed 45 chunks in 3m 24s
```

### Running the Online Pipeline (Query)

```bash
# Run query with poetry
poetry run python online/run_query.py

# Enter your question:
> What is Santiago's Personal Legend?

# Expected output:
# INFO: Embedding query...
# INFO: Searching Pinecone (top-k=10)...
# INFO: Generating answer...
# 
# Answer:
# Santiago's Personal Legend is to travel to the Egyptian Pyramids...
# 
# (Retrieved 10 chunks, 6,234 tokens used, ~$0.12 cost)
```

---

## Part 6: Troubleshooting

### Common Issues

**1. "Module not found" errors**:
```bash
# Reinstall dependencies
poetry install

# Verify installation
poetry run python -c "import openai, pinecone"
```

**2. OpenAI API errors (401 Unauthorized)**:
- Check your API key in `.env`
- Verify billing is set up in OpenAI account
- Check if API key has necessary permissions

**3. Pinecone connection errors**:
- Verify API key and environment in `.env`
- Check index name matches what you created
- Ensure index is in "Ready" state in console

**4. NLTK punkt not found**:
```bash
python -c "import nltk; nltk.download('punkt')"
```

**5. Rate limit errors (429)**:
- Tenacity should automatically retry
- If persistent, add longer delays or reduce batch sizes
- Check your OpenAI account rate limits

**6. Out of memory errors**:
- Process document in smaller batches
- Reduce batch size when uploading to Pinecone

---

## Part 7: Cost Monitoring

### Track Usage

**OpenAI**:
- Monitor at [https://platform.openai.com/usage](https://platform.openai.com/usage)
- Set up usage alerts in Settings

**Pinecone**:
- Monitor at [https://app.pinecone.io](https://app.pinecone.io)
- Check index usage and query volume

### Estimated Costs for Full Project

- **One-time indexing**: $0.01-0.05
- **Per query**: $0.10-0.50 (depending on chunk count and answer length)
- **100 queries**: $10-50

---

## Next Steps

1. Complete implementation following the phases above
2. Test with sample queries about The Alchemist
3. Experiment with different parameters:
   - Chunk sizes
   - Overlap amounts
   - Top-k values
   - Temperature settings
4. Add advanced features:
   - Query history
   - Response caching
   - Multi-document support
   - Conversation memory

---

## Resources

- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Pinecone Documentation](https://docs.pinecone.io)
- [tiktoken Documentation](https://github.com/openai/tiktoken)
- [NLTK Documentation](https://www.nltk.org)
- [Tenacity Documentation](https://tenacity.readthedocs.io)
- [Loguru Documentation](https://loguru.readthedocs.io)
