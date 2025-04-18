# Metabase Code Semantic Search

A semantic code search tool for the Metabase backend codebase using vector embeddings and vector search, enabling natural language queries against Clojure code.

## Usage

```bash
# Step 1: Extract function definitions using clj-kondo
clj-kondo --config '{:analysis true :output {:format :json}}' --lint src/metabase/ > metabase.json

# Step 2: Extract functions from the analysis data
python main.py extract metabase.json --output chunks.json

# Step 3: Generate embeddings from the extracted function chunks
python main.py embed chunks.json --output embeddings.json

# Step 4: Load embeddings into ChromaDB
python main.py load embeddings.json --db-path ./chroma_db

# Step 5: Search for code using natural language
python main.py search "how to render a PNG image" --n-results 5
```


Example search:
```bash
>> python main.py search "how are funnel charts rendered in notifications?"
Starting semantic search for: 'how are funnel charts rendered in notifications?'
Generating embedding for query using model text-embedding-3-small...
Query embedding generated in 0.68 seconds
Connected to collection 'clojure_code' with 25301 items
Searching collection for similar code...
Search completed in 0.00 seconds

Search completed in 1.01 seconds
Found 5 results

Search results for: 'how are funnel charts rendered in notifications?'
================================================================================
Result 1 (distance: 0.8036):
Function: metabase.channel.render.js.svg/funnel
File: ../../src/metabase/channel/render/js/svg.clj (lines 152-158)
----------------------------------------
metabase.channel.render.js.svg/funnel
(defn funnel
  "Clojure entrypoint to render a funnel chart. Data should be vec of [[Step Measure]] where Step is {:name name :format format-options} and Measure is {:format format-options} and you go and look to frontend/src/metabase/static-viz/components/FunnelChart/types.ts for the actual format options.
  Returns a byte array of a png file."
  [data settings]
  (let [svg-string (.asString (js.engine/execute-fn-name (context) "funnel" (json/encode data)
                                                         (json/encode settings)))]
    (svg-string->bytes svg-string)))

================================================================================
Result 2 (distance: 0.8036):
Function: metabase.channel.render.js.svg/funnel
File: ../../src/metabase/channel/render/js/svg.clj (lines 152-158)
----------------------------------------
metabase.channel.render.js.svg/funnel
(defn funnel
  "Clojure entrypoint to render a funnel chart. Data should be vec of [[Step Measure]] where Step is {:name name :format format-options} and Measure is {:format format-options} and you go and look to frontend/src/metabase/static-viz/components/FunnelChart/types.ts for the actual format options.
  Returns a byte array of a png file."
  [data settings]
  (let [svg-string (.asString (js.engine/execute-fn-name (context) "funnel" (json/encode data)
                                                         (json/encode settings)))]
    (svg-string->bytes svg-string)))

================================================================================
```

# Test Results
_Full notes in test-questions.md_


Steps:
  - Two new API keys
    - ON -> ...tyler_uius
    - OFF -> ...tyler_tmqs
  - Run claude with semantic search *ON*
  - Send questions one at a time
  - Record usage, save log
  - Accidentally delete log (oops)
  - Run claude with semantic search *OFF*
  - Send questions one at a time
  - Record usage, save log in `semantic_off.log`

<details>
  <summary>Question list</summary>
  
1. How does Metabase handle database connection pooling?
2. What's the process for rendering dashboard cards in PDF exports?
3. How does Metabase implement caching for query results?
4. How does the authentication system handle JWT tokens?
5. How are database permissions enforced in the query processor?
6. What's the implementation of data sandboxing in Metabase?
7. How does Metabase handle timezone conversions for queries?
8. What's the process for scheduling and sending dashboard subscriptions?
9. How does Metabase implement custom expressions in the query builder?
10. How are database driver plugins loaded at runtime?
11. What's the implementation of the audit system for tracking user actions?
12. How does Metabase handle parameter passing in embedded dashboards?
13. What's the process for syncing database metadata with Metabase?
14. How does Metabase implement native query execution with variables?
15. What's the implementation of data model persistence in Metabase?
16. How does the alert system detect and notify about threshold violations?
17. What's the architecture of the public sharing feature?
18. How does Metabase handle database-specific SQL dialects?
19. What's the process for migrating application data between versions?
20. How does Metabase implement custom geojson map visualizations on the backend?
</details>

<img width="751" alt="image" src="https://github.com/user-attachments/assets/e304abda-32d1-4e77-8b55-9bdde68ca3ce" />



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

## Improvements
- Testing different embedding models
- Testing different embedding dbs
- Using a smarter chunking strategy
- Use chroma's `where` query param: https://docs.trychroma.com/reference/python/collection#query

## Next Steps

- Further testing
- Analyze quality and accuracy of the responses

