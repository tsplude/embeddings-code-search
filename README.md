# Metabase Code Semantic Search

This repository contains tools for creating a semantic search system for Metabase's codebase, enabling natural language queries against Clojure code.

## Problem Statement

We needed a way to understand and navigate the Metabase codebase using natural language queries. Traditional text-based searches like grep are limited to exact keyword matching, making it difficult to find code based on concepts or functionality.

The goal was to create a semantic search pipeline that would:
1. Extract function definitions from Clojure code
2. Generate embeddings for these functions using OpenAI
3. Store them in a vector database (ChromaDB)
4. Provide a query interface for natural language search

## What We've Accomplished

1. **Function Extraction**: We've created a process using clj-kondo to extract function definitions from Clojure files, capturing function names, namespaces, and source code locations.

2. **Embedding Generation**: We've built a Python script that:
   - Takes clj-kondo analysis data
   - Extracts function code and metadata
   - Generates embeddings using OpenAI's API
   - Stores these embeddings in a structured format

3. **Vector Database**: We've implemented ChromaDB integration to:
   - Store function embeddings in a persistent database
   - Enable semantic search with proper metadata
   - Support fast query operations

4. **Search Interface**: We've created a command-line tool that:
   - Takes natural language queries
   - Converts queries to embeddings
   - Searches the vector database
   - Returns relevant code snippets ranked by similarity

## Usage

```bash
# Step 1: Extract function definitions using clj-kondo
clj-kondo --config '{:analysis true :output {:format :edn}}' --lint src/metabase/channel/render > analysis-body.edn

# Step 2: Extract functions and generate embeddings
python main.py extract analysis-body.edn --output embeddings.json

# Step 3: Load embeddings into ChromaDB
python main.py load embeddings.json --db-path ./chroma_db

# Step 4: Search for code using natural language
python main.py search "how to render a PNG image" --n-results 5
```

## Technical Challenges

1. **Custom Macro Handling**: We discovered that clj-kondo doesn't inherently recognize function definitions created via custom macros like `mu/defmethod`. This requires additional configuration or post-processing to fully capture all function definitions.

2. **Embedding Context**: Determining the right amount of context to include in embeddings (e.g., just the function body, or function with surrounding context) to get the most relevant search results.

3. **Database Management**: Structuring the ChromaDB collections and metadata to support efficient semantic search while maintaining all necessary context for results.

## Future Work

1. Extend the extraction to capture all function types, including those defined through custom macros like `mu/defmethod`
2. Implement a web interface for easier searching
3. Add support for incremental updates to avoid re-embedding the entire codebase
4. Explore fine-tuning models on Clojure code to improve embedding quality

## Tools Used

- clj-kondo: Static analyzer for Clojure code
- OpenAI API: For generating text embeddings
- ChromaDB: Vector database for storing and querying embeddings
- Python: For script implementation and database management