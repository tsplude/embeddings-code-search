## 20250417 Test of semantic search on vs. off

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

CLAUDE.md addition:
```
# Backend Code Exploration
- For non-specific questions about the Clojure backend code, always use the semantic search tool as your default starting point
- Run: `python tsp-repos/embeddings-code-search/search.py -q "your natural language query"`
- This tool uses vector embeddings of all Clojure functions to perform semantic search
- Structure of results:
  - Each result shows function name, file location, and line numbers
  - Distance score (lower is better) showing semantic similarity to query
  - The actual function code for immediate review
- After identifying relevant code locations with this tool, you can use View/Edit tools to examine those areas in more detail
- Convert vague or exploratory questions into appropriate natural language queries
```

## [TEST] Semantic search ENABLED
api key: sk-ant-api03-73r...JAAA

Notes:
Did not listen to the CLAUDE.md instructions, had to specifically paste them in at the beginning
Question 2 triggered frontend searches
Hit autocompact on question 9
After autocompact, seemed to forget about the tool and effectively restarted its analysis of Q9 using default tools
Tried to correct this with "For the rest of this conversation please REMEMBER this from your memory file: <pasted the addition above>"
It then not only started each search with the semantic search, but every nested search thereafter also used the semantic search (whereas before it would fallback to its default tools)
For Q12 it switched back and forth between semantic search and file reads
Q14, tried reading nonexistent file src/metabase/driver/sql/parameters/parse.clj (?)
Q18, asked permission to do this...
⏺ Bash(ls -la /Users/tplude/code/metabase/src/metabase/driver/h2/*.clj | wc -l)…
  ⎿  1
Q20 hit auto compact
The instructions on how to use the tool could be better. e.g. when it uses its Task(...) tool it's using claude itself to generate the input instructions in some particular way:
Task(Search for frontend files related to map visualizations and geojson in the Metabase codebase. First, look for files in /frontend/src/metabase/visualizations/ that might be related to maps. Then check for any files related to handling geojson data for visualization. Focus on mapping, …
      choropleth, and related visualization files. Provide file paths and brief descriptions of their purpose.)
```
> /cost
  ⎿  Total cost:            $8.22
     Total duration (API):  23m 35.2s
     Total duration (wall): 31m 43.9s
     Total code changes:    0 lines added, 0 lines removed
```
^ is before attempting to log the conversation (which it did _not_ handle well)
tokens in: 18,205,781
tokens out: 53,526
claude_code_key_tyler_uius

## [TEST] Semantic search DISABLED
api key: sk-ant-api03-_XN...agAA

Notes:
Q7 hit autocompact
Q15 hit autocompact
Q20 hit autocompact
```
> /cost
  ⎿  Total cost:            $16.56
     Total duration (API):  37m 36.3s
     Total duration (wall): 1h 11m 18.1s
     Total code changes:    0 lines added, 0 lines removed
```
tokens in: 32,969,497
tokens out: 109,926
claude_code_key_tyler_

saved log is at: semantic_off.log
attempted to have it save a log just like before, multiple errors and hung there for 30+ minutes before I killed it
```
> Now please save an entire log of this conversation from beginning to end, verbatim and without modification. Save it to `claude-transcript.md`

⏺ I'll save the entire conversation to a markdown file without modifications.

⏺ Read(conversation-log.md)…
  ⎿  Read 9 lines (ctrl+r to expand)

⏺ Write
  ⎿  InputValidationError: Replace failed due to the following issue:
     The required parameter `content` is missing

⏺ Let me try again with the content included:

⏺ Write
  ⎿  InputValidationError: Replace failed due to the following issue:
     The required parameter `content` is missing

⏺ I'll use the Bash tool to save the file:

  Bash
  ⎿  Running…
```
