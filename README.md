# Metabase Code Semantic Search

A semantic code search tool for the Metabase backend codebase using vector embeddings and vector search, enabling natural language queries against Clojure code.

## Usage

```bash
# Step 1: Extract function definitions using clj-kondo
clj-kondo --config '{:analysis true :output {:format :edn}}' --lint src/metabase/channel/render > analysis-body.edn

# Step 2: Extract functions from the analysis data
python main.py extract analysis-body.edn --output chunks.json

# Step 3: Generate embeddings from the extracted function chunks
python main.py embed chunks.json --output embeddings.json

# Step 4: Load embeddings into ChromaDB
python main.py load embeddings.json --db-path ./chroma_db

# Step 5: Search for code using natural language
python main.py search "how to render a PNG image" --n-results 5
```

### Command Details

#### Extract
Extracts function chunks from clj-kondo analysis data, including defmethod implementations.
```bash
python main.py extract analysis-body.edn --output chunks.json --rel-dir ../..
```

#### Embed
Generates embeddings for function chunks using OpenAI's API in batches.
```bash
python main.py embed chunks.json --output embeddings.json --model text-embedding-3-small --batch-size 100
```

#### Load
Loads embeddings into ChromaDB for vector search.
```bash
python main.py load embeddings.json --db-path ./chroma_db --collection clojure_code
```

#### Search
Searches for code using natural language queries.
```bash
python main.py search "your query here" --n-results 5 --model text-embedding-3-small
```

## Next Steps

- Implementation as a full MCP server for direct integration with Claude Code
- Test what impact this has on claude's ability to answer vague backend questions
