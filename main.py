#!/usr/bin/env python3

import argparse
import json
import os
import pandas as pd
from openai import OpenAI
import time
import uuid
import hashlib
import dotenv
dotenv.load_dotenv('.env')

try:
    import chromadb
except ImportError:
    print("ChromaDB not installed. Run: pip install chromadb")

def generate_embeddings(code_chunks, output_file="embeddings.json", model="text-embedding-3-small"):
    """
    Put list of code chunk dicts into a pandas dataframe
    Initialize an open ai client instance using OPENAI_API_KEY from environment variable
    populate new dataframe column with embeddings, e.g.:
    def get_embedding(text, model="text-embedding-3-small"):
        text = text.replace("\n", " ")
        return client.embeddings.create(input = [text], model=model)
    save results to a json file
    
    Args:
        code_chunks (list): List of dictionaries containing code chunks and metadata
        output_file (str): Path to save the JSON file with embeddings
        model (str): OpenAI model to use for embeddings
        
    Returns:
        DataFrame: Pandas DataFrame with code chunks and embeddings
    """
    
    # Ensure OpenAI API key is available
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Create DataFrame from code chunks
    df = pd.DataFrame.from_dict(code_chunks)
    
    # Define embedding function
    def get_embedding(text, model=model):
        # Replace newlines with spaces for better embedding quality
        text = text.replace("\n", " ")
        
        # Rate limiting - simple protection against hitting API limits
        #time.sleep(0.5)
        
        try:
            # Get embeddings from OpenAI
            response = client.embeddings.create(input=[text], model=model)
            # Extract the embedding from the response
            embedding = response.data[0].embedding
            return embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None
    
    # Generate embeddings for each code chunk
    print(f"Generating embeddings for {len(df)} code chunks using model {model}...")
    
    # Create a progress tracking mechanism
    total = len(df)
    
    # Generate embeddings for each function
    df['embedding'] = None
    for idx, row in df.iterrows():
        # Combine function info and code for better context in embedding
        text_to_embed = f"{row['namespace']}/{row['name']}\n{row['code']}"
        
        # Get embedding
        embedding = get_embedding(text_to_embed)
        df.at[idx, 'embedding'] = embedding
        
        # Print progress
        if idx % 10 == 0 or idx == total - 1:
            print(f"Progress: {idx+1}/{total} ({((idx+1)/total)*100:.1f}%)")
    
    # Save to JSON file
    print(f"Saving embeddings to {output_file}...")
    
    # Convert DataFrame to records for JSON serialization
    records = df.to_dict(orient='records')
    
    with open(output_file, 'w') as f:
        json.dump(records, f, indent=2)
    
    print(f"Successfully saved {len(records)} embeddings to {output_file}")
    
    return df

def extract_function_chunks(analysis_file, rel_dir="../.."):
    """
    Extract function definitions from source files based on analysis data.
    
    Args:
        analysis_file (str): Path to a JSON file containing clj-kondo analysis data
        
    Returns:
        List of dictionaries containing function metadata and extracted source code
    """
    # Read the analysis data
    with open(analysis_file, 'r') as f:
        data = json.load(f)
    
    # Check if the expected data structure exists
    if 'analysis' not in data:
        print(f"Error: The file {analysis_file} does not contain an 'analysis' key")
        return []
    
    analysis = data['analysis']
    
    # Store the cached file contents to avoid reading the same file multiple times
    file_cache = {}
    
    # Process each function definition
    function_chunks = []
    for func_def in analysis['var-definitions']:
        # Extract metadata fields
        filename = os.path.join(rel_dir, func_def.get('filename'))
        if not filename:
            continue
            
        # Extract line numbers
        start_row = func_def.get('row', 0)
        end_row = func_def.get('end-row', 0)
        
        if start_row == 0 or end_row == 0 or start_row > end_row:
            print(f"Warning: Invalid row range for {func_def.get('name')} in {filename}")
            continue
        
        # Read the file if not already cached
        if filename not in file_cache:
            try:
                with open(filename, 'r') as f:
                    file_cache[filename] = f.readlines()
            except Exception as e:
                print(f"Error reading file {filename}: {e}")
                continue
        
        # Extract the function code (line numbers in analysis are 1-based)
        lines = file_cache[filename]
        if start_row <= len(lines):
            # Extract lines and join them
            function_code = ''.join(lines[start_row-1:end_row])
            
            # Create a result object with metadata and code
            result = {
                'name': func_def.get('name', ''),
                'namespace': func_def.get('ns', ''),
                'doc': func_def.get('doc', ''),
                'filename': filename,
                'start_row': start_row,
                'end_row': end_row,
                'private': func_def.get('private', False),
                'code': function_code
            }
            
            function_chunks.append(result)
            
            # Print the chunk to stdout
            print(f"Function: {result['namespace']}/{result['name']} ({filename}:{start_row}-{end_row})")
            print("-" * 40)
            print(function_code)
            print("=" * 40)
    
    return function_chunks


def populate_chromadb(embeddings_data, db_path="./chroma_db", collection_name="metabase-backend", embeddings_dim=1536):
    """
    Populate a persistent ChromaDB client with embeddings data.
    
    Args:
        embeddings_data: Either a pandas DataFrame with embeddings or a path to a JSON file
        db_path (str): Path where ChromaDB will store its data
        collection_name (str): Name of the ChromaDB collection to create/use
        embeddings_dim (int): Dimension of the embeddings
        
    Returns:
        ChromaDB collection object
    """
    
    print(f"Initializing persistent ChromaDB client at {db_path}...")
    client = chromadb.PersistentClient(path=db_path)
    
    # Get or create collection
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"description": "Clojure code function embeddings"}
    )
    
    # Check if input is a file path or DataFrame
    if isinstance(embeddings_data, str):
        print(f"Loading embeddings from file: {embeddings_data}")
        with open(embeddings_data, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame.from_dict(data)
    else:
        print("Using provided DataFrame")
        df = embeddings_data
    
    # Check if embeddings exist in the dataframe
    if 'embedding' not in df.columns:
        raise ValueError("DataFrame does not contain an 'embedding' column")
    
    # Prepare data for ChromaDB
    ids = []
    embeddings = []
    documents = []
    metadatas = []
    
    print(f"Processing {len(df)} code chunks for ChromaDB...")
    
    for i, row in df.iterrows():
        # Validate embedding
        if not row.get('embedding'):
            print(f"Warning: Missing embedding for {row.get('namespace')}/{row.get('name')}. Skipping.")
            continue

        # Generate a unique ID using a hash of the code
        code_hash = hashlib.md5(row['code'].encode()).hexdigest()
        doc_id = f"{row['namespace']}.{row['name']}_{code_hash[:8]}"
        
        # Create document text (for search context)
        document_text = f"{row['namespace']}/{row['name']}\n{row['code']}"
        
        # Create metadata
        metadata = {
            "namespace": row['namespace'],
            "name": row['name'],
            "filename": row['filename'],
            "start_row": int(row['start_row']),
            "end_row": int(row['end_row']),
            "private": bool(row.get('private', False))
        }
        
        # Add doc to the collection
        if i % 100 == 0:
            print(f"Processing item {i+1}/{len(df)}...")
            
        ids.append(doc_id)
        embeddings.append(row['embedding'])
        documents.append(document_text)
        metadatas.append(metadata)
    
    # Add to ChromaDB in batches (to avoid potential memory issues)
    batch_size = 500
    total_batches = (len(ids) + batch_size - 1) // batch_size
    
    print(f"Adding data to ChromaDB in {total_batches} batches...")
    
    for i in range(0, len(ids), batch_size):
        end_idx = min(i + batch_size, len(ids))
        batch_ids = ids[i:end_idx]
        batch_embeddings = embeddings[i:end_idx]
        batch_documents = documents[i:end_idx]
        batch_metadatas = metadatas[i:end_idx]
        
        print(f"Adding batch {(i//batch_size)+1}/{total_batches} ({len(batch_ids)} items)...")
        
        # Add batch to collection
        collection.add(
            ids=batch_ids,
            embeddings=batch_embeddings,
            documents=batch_documents,
            metadatas=batch_metadatas
        )
    
    print(f"Successfully populated ChromaDB collection '{collection_name}' with {collection.count()} items.")
    return collection


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
    
    print(f"Generating embedding for query: '{query}'")
    query_embedding = get_query_embedding(query)
    
    # Initialize ChromaDB client
    db_client = chromadb.PersistentClient(path=db_path)
    
    # Get collection
    try:
        collection = db_client.get_collection(collection_name)
    except ValueError:
        print(f"Collection '{collection_name}' does not exist. Please create it first.")
        return []
    
    # Query the collection
    print(f"Searching for: '{query}'")
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    
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
    
    return formatted_results


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process clj-kondo analysis data to extract function chunks, generate embeddings, and search code')
    
    # Create subparsers for different operations
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Extract command
    extract_parser = subparsers.add_parser('extract', help='Extract function chunks from clj-kondo analysis')
    extract_parser.add_argument('analysis_file', help='Path to the JSON file containing clj-kondo analysis data')
    extract_parser.add_argument('--output', default='embeddings.json', help='Output file path for embeddings JSON')
    extract_parser.add_argument('--model', default='text-embedding-3-small', help='OpenAI model to use for embeddings')
    extract_parser.add_argument('--extract-only', action='store_true', help='Only extract function chunks without generating embeddings')
    extract_parser.add_argument('--rel-dir', default='../..', help='Relative directory path to prepend to filenames')
    
    # Load to ChromaDB command
    load_parser = subparsers.add_parser('load', help='Load embeddings to ChromaDB')
    load_parser.add_argument('embeddings_file', help='Path to the JSON file containing embeddings')
    load_parser.add_argument('--db-path', default='./chroma_db', help='Path to store ChromaDB data')
    load_parser.add_argument('--collection', default='clojure_code', help='Name of the ChromaDB collection')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search code in ChromaDB')
    search_parser.add_argument('query', help='Natural language query to search for')
    search_parser.add_argument('--db-path', default='./chroma_db', help='Path to ChromaDB data')
    search_parser.add_argument('--collection', default='clojure_code', help='Name of the ChromaDB collection')
    search_parser.add_argument('--model', default='text-embedding-3-small', help='OpenAI model for query embedding')
    search_parser.add_argument('--n-results', type=int, default=5, help='Number of results to return')
    
    args = parser.parse_args()
    
    # Handle commands
    if args.command == 'extract':
        # Extract function chunks
        function_chunks = extract_function_chunks(args.analysis_file, rel_dir=args.rel_dir)
        
        if not args.extract_only and function_chunks:
            # Generate embeddings for the extracted chunks
            df = generate_embeddings(function_chunks, output_file=args.output, model=args.model)
            
            print(f"\nTo load these embeddings into ChromaDB, run:")
            print(f"python {os.path.basename(__file__)} load {args.output}")
            
    elif args.command == 'load':
        # Load embeddings to ChromaDB
        populate_chromadb(
            args.embeddings_file,
            db_path=args.db_path,
            collection_name=args.collection
        )
        
        print(f"\nTo search the code using ChromaDB, run:")
        print(f"python {os.path.basename(__file__)} search \"your query here\"")
        
    elif args.command == 'search':
        # Search code
        results = search_code(
            args.query,
            db_path=args.db_path,
            collection_name=args.collection,
            model=args.model,
            n_results=args.n_results
        )
        
        # Display results
        if results:
            print(f"\nSearch results for: '{args.query}'")
            print("=" * 80)
            
            for i, result in enumerate(results):
                print(f"Result {i+1} (distance: {result['distance']:.4f}):")
                print(f"Function: {result['metadata']['namespace']}/{result['metadata']['name']}")
                print(f"File: {result['metadata']['filename']} (lines {result['metadata']['start_row']}-{result['metadata']['end_row']})")
                print("-" * 40)
                # Display the first few lines of code
                code_lines = result['document'].split('\n')
                display_lines = min(10, len(code_lines))  # Show at most 10 lines
                print('\n'.join(code_lines[:display_lines]))
                if display_lines < len(code_lines):
                    print(f"... ({len(code_lines) - display_lines} more lines)")
                print("=" * 80)
        else:
            print("No results found.")
    
    else:
        # No command specified, show help
        parser.print_help()


if __name__ == "__main__":
    main()
