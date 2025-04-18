#!/usr/bin/env python3

import os
import time
import dotenv
from openai import OpenAI

# Load environment variables
dotenv.load_dotenv('.env')

def search_code(query, db_path="./chroma_db", collection_name="clojure_code", model="text-embedding-3-small", n_results=5):
    """
    Search for code in ChromaDB using semantic search.
    
    Args:
        query (str): Natural language query
        db_path (str): Path to ChromaDB data
        collection_name (str): Name of the collection
        model (str): OpenAI model to use for query embedding
        n_results (int): Number of results to return
        
    Returns:
        List of search results
    """
    start_time = time.time()
    print(f"Starting semantic search for: '{query}'")
    
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
    
    print(f"Generating embedding for query using model {model}...")
    embedding_start_time = time.time()
    query_embedding = get_query_embedding(query)
    embedding_time = time.time() - embedding_start_time
    print(f"Query embedding generated in {embedding_time:.2f} seconds")
    
    # Initialize ChromaDB client
    db_client = chromadb.PersistentClient(path=db_path)
    
    # Get collection
    try:
        collection = db_client.get_collection(collection_name)
        collection_size = collection.count()
        print(f"Connected to collection '{collection_name}' with {collection_size} items")
    except ValueError:
        print(f"Collection '{collection_name}' does not exist. Please create it first.")
        return []
    
    # Query the collection
    print(f"Searching collection for similar code...")
    search_start_time = time.time()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    search_time = time.time() - search_start_time
    print(f"Search completed in {search_time:.2f} seconds")
    
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
    
    print(f"\nSearch completed in {execution_time:.2f} seconds")
    print(f"Found {len(formatted_results)} results")
    
    return formatted_results