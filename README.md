# rag-prototype

This project is to build a full RAG pipeline. My current project archirtecture planned are:

1. Offline:
   1.1. documents: there's no need for preprocessing the text. The text will be in the file documents.txt in this repo
   1.2. chunking: strategy: paragraph or sentence split; chunking size: 400-800 tokens; chunking overlap: 80-150 tokens 
   1.3. embedding: embedding model: openAI text-embedding-3-large
   1.4. vector database: we will use pinecone to store the vectors and build HNSW index from the vectors

2. Online:
   2.1. query 
   2.2. embedding: embedding model: openAI text-embedding-3-large
   2.3. similarity search: search the vector database using the the HNSW index we pre built
   2.4. return relevant chunks: top-k = 15
   2.5. Pass the (prompt = query + relevant chunks) to the LLM. Model choice: openAI gpt-5
   2.6. output the answers

These are 2 paths. We need to prepare the vector database and build index first before running the query.

Requirements:
- use .env file to store API, credentials and we would have a config.py file in the repo to retrieve the credentials consistently and abstractly in the project
- use poetry for dependencies management
- use loguru logger instead of printing statements 
