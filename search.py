#!/usr/bin/env python3

import os
import time
import dotenv
from openai import OpenAI

# Load environment variables
dotenv.load_dotenv('.env')

CHROMA_PATH="/Users/tplude/code/metabase/tsp-repos/embeddings-code-search/chroma_db"

def _log(log_str, verbose):
    """Helper function to handle logging based on verbose flag"""
    if verbose:
        print(log_str)

def search_chroma(query, db_path="tsp-repos/embeddings-code-search/chroma_db", collection_name="clojure_code", model="text-embedding-3-small", n_results=5, verbose=False):
    """
    Search for code in ChromaDB using semantic search.
    
    Args:
        query (str): Natural language query
        db_path (str): Path to ChromaDB data
        collection_name (str): Name of the collection
        model (str): OpenAI model to use for query embedding
        n_results (int): Number of results to return
        verbose (bool): Whether to print verbose logging information
        
    Returns:
        List of search results
    """
    start_time = time.time()
    _log(f"Starting semantic search for: '{query}'", verbose)
    
    try:
        import chromadb
    except ImportError:
        raise ImportError("ChromaDB not installed. Run: pip install chromadb")
    
    # Initialize OpenAI client for query embedding
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    
    client = OpenAI(api_key=api_key)
    
    # Get query embedding
    def get_query_embedding(text, model=model):
        text = text.replace("\n", " ")
        response = client.embeddings.create(input=[text], model=model)
        return response.data[0].embedding
    
    _log(f"Generating embedding for query using model {model}...", verbose)
    embedding_start_time = time.time()
    query_embedding = get_query_embedding(query)
    embedding_time = time.time() - embedding_start_time
    _log(f"Query embedding generated in {embedding_time:.2f} seconds", verbose)
    
    # Initialize ChromaDB client
    db_client = chromadb.PersistentClient(path=db_path)
    
    # Get collection
    try:
        collection = db_client.get_collection(collection_name)
        collection_size = collection.count()
        _log(f"Connected to collection '{collection_name}' with {collection_size} items", verbose)
    except ValueError:
        _log(f"Collection '{collection_name}' does not exist. Please create it first.", verbose)
        return []
    
    # Query the collection
    _log(f"Searching collection for similar code...", verbose)
    search_start_time = time.time()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    search_time = time.time() - search_start_time
    _log(f"Search completed in {search_time:.2f} seconds", verbose)
    
    # Format and return results
    formatted_results = []
    for i in range(len(results["ids"][0])):
        result = {
            "id": results["ids"][0][i],
            "document": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i]
        }
        formatted_results.append(result)
    
    # Calculate and report execution stats
    end_time = time.time()
    execution_time = end_time - start_time
    
    _log(f"\nSearch completed in {execution_time:.2f} seconds", verbose)
    _log(f"Found {len(formatted_results)} results", verbose)
    
    return formatted_results


# Make the file executable as a standalone script
if __name__ == "__main__":
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Semantic code search with ChromaDB")
    parser.add_argument("-q", "--query", required=True, help="Natural language query to search for")
    parser.add_argument("--db-path", default=CHROMA_PATH, help="Path to ChromaDB data")
    parser.add_argument("--collection", default="clojure_code", help="Name of the ChromaDB collection")
    parser.add_argument("--model", default="text-embedding-3-small", help="OpenAI model for query embedding")
    parser.add_argument("--n-results", type=int, default=5, help="Number of results to return")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose output")
    
    args = parser.parse_args()
    
    # Search with the provided arguments
    results = search_chroma(
        query=args.query,
        db_path=args.db_path,
        collection_name=args.collection,
        model=args.model,
        n_results=args.n_results,
        verbose=args.verbose
    )
    
    # Display results
    print(f"\nSearch results for: '{args.query}'")
    print("=" * 80)
    
    for i, result in enumerate(results):
        print(f"Result {i+1} (distance: {result['distance']:.4f}):")
        print(f"Function: {result['metadata']['namespace']}/{result['metadata']['name']}")
        print(f"File: {result['metadata']['filename'].strip('../..')} (lines {result['metadata']['start_row']}-{result['metadata']['end_row']})")
        print("-" * 40)
        # Display the first few lines of code
        code_lines = result['document'].split('\n')
        print('\n'.join(code_lines))
        print("=" * 80)
