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
from search import search_code

try:
    import chromadb
except ImportError:
    print("ChromaDB not installed. Run: pip install chromadb")

def generate_embeddings(code_chunks, output_file="embeddings.json", model="text-embedding-3-small", batch_size=100):
    """
    Generate embeddings for a list of code chunks using OpenAI's API in batches.
    
    Args:
        code_chunks (list): List of dictionaries containing code chunks and metadata
        output_file (str): Path to save the JSON file with embeddings
        model (str): OpenAI model to use for embeddings
        batch_size (int): Number of texts to batch in a single API call
        
    Returns:
        DataFrame: Pandas DataFrame with code chunks and embeddings
    """
    start_time = time.time()
    print(f"Starting embedding generation using model {model} with batch size {batch_size}...")
    
    # Ensure OpenAI API key is available
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Create DataFrame from code chunks
    df = pd.DataFrame.from_dict(code_chunks)
    
    # Generate embeddings for each code chunk
    print(f"Generating embeddings for {len(df)} code chunks...")
    
    # Create a progress tracking mechanism
    total = len(df)
    
    # Initialize embedding column
    df['embedding'] = None
    api_call_count = 0
    api_errors = 0
    
    # Process in batches
    batches = [range(i, min(i + batch_size, len(df))) for i in range(0, len(df), batch_size)]
    
    for batch_idx, batch_range in enumerate(batches):
        batch_start_time = time.time()
        
        # Prepare batch texts
        batch_texts = []
        batch_indices = []
        
        for idx in batch_range:
            row = df.iloc[idx]
            # Combine function info and code for better context in embedding
            text_to_embed = f"{row['namespace']}/{row['name']}\n{row['code']}"
            # Replace newlines with spaces for better embedding quality
            text_to_embed = text_to_embed.replace("\n", " ")
            batch_texts.append(text_to_embed)
            batch_indices.append(idx)
        
        try:
            # Get embeddings for batch from OpenAI
            response = client.embeddings.create(input=batch_texts, model=model)
            
            # Process results
            for i, embedding_data in enumerate(response.data):
                df_idx = batch_indices[i]
                df.at[df_idx, 'embedding'] = embedding_data.embedding
            
            # Count successful API call
            api_call_count += 1
            
        except Exception as e:
            print(f"Error generating embeddings for batch {batch_idx+1}: {e}")
            api_errors += 1
            # Mark all items in the failed batch as None
            for idx in batch_indices:
                df.at[idx, 'embedding'] = None
        
        # Print progress
        batch_items_processed = min((batch_idx + 1) * batch_size, total)
        print(f"Progress: Batch {batch_idx+1}/{len(batches)} - {batch_items_processed}/{total} items " +
              f"({(batch_items_processed/total)*100:.1f}%) - " + 
              f"Batch time: {time.time() - batch_start_time:.2f}s")
    
    # Calculate success rate
    success_rate = ((api_call_count - api_errors) / api_call_count) * 100 if api_call_count > 0 else 0
    
    # Save to JSON file
    print(f"Saving embeddings to {output_file}...")
    
    # Convert DataFrame to records for JSON serialization
    records = df.to_dict(orient='records')
    
    with open(output_file, 'w') as f:
        json.dump(records, f, indent=2)
    
    # Calculate and report execution stats
    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"\nEmbedding generation completed in {execution_time:.2f} seconds")
    print(f"Successfully generated {len(records)} embeddings ({api_errors} errors)")
    print(f"API success rate: {success_rate:.1f}%")
    print(f"Average time per embedding: {execution_time/total:.2f} seconds")
    
    return df

def extract_function_chunks(analysis_file, rel_dir="../.."):
    """
    Extract function definitions from source files based on analysis data.
    Avoids duplicates by tracking already processed functions.
    
    Args:
        analysis_file (str): Path to a JSON file containing clj-kondo analysis data
        rel_dir (str): Relative directory path to prepend to filenames
        
    Returns:
        List of dictionaries containing function metadata and extracted source code
    """
    start_time = time.time()
    print(f"Starting function extraction from {analysis_file}...")
    
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
    # Track unique functions to avoid duplicates (using sets for fast lookups)
    function_keys = set()      # Track by namespace:name:file:start:end
    function_code_hashes = set() # Track by content hash as fallback duplicates
    
    # Track statistics
    chunk_count = 0
    duplicate_count = 0
    duplicate_by_hash = 0
    files_processed = set()
    skipped_count = 0
    
    # Get total counts for progress reporting
    var_def_count = len(analysis.get('var-definitions', []))
    defmethod_count = len([u for u in analysis.get('var-usages', []) if u.get('name') == 'defmethod'])
    total_count = var_def_count + defmethod_count
    
    print(f"Found {var_def_count} standard functions and {defmethod_count} defmethod functions to process")
    
    # Process standard function definitions first
    for i, func_def in enumerate(analysis['var-definitions']):
        # Report progress
        if i % 20 == 0 or i == var_def_count - 1:
            print(f"Processing standard functions: {i+1}/{var_def_count} ({((i+1)/var_def_count)*100:.1f}%)")
        
        # Extract metadata fields
        filename = os.path.join(rel_dir, func_def.get('filename'))
        if not filename:
            skipped_count += 1
            continue
            
        # Extract line numbers
        start_row = func_def.get('row', 0)
        end_row = func_def.get('end-row', 0)
        
        if start_row == 0 or end_row == 0 or start_row > end_row:
            print(f"Warning: Invalid row range for {func_def.get('name')} in {filename}")
            skipped_count += 1
            continue
        
        # Generate a unique key for this function
        func_name = func_def.get('name', '')
        namespace = func_def.get('ns', '')
        func_key = f"{namespace}::{func_name}::{filename}::{start_row}::{end_row}"
        
        # Check if we've already processed this function
        if func_key in function_keys:
            duplicate_count += 1
            continue
        
        # Add to the set of processed functions
        function_keys.add(func_key)
        
        # Read the file if not already cached
        if filename not in file_cache:
            try:
                with open(filename, 'r') as f:
                    file_cache[filename] = f.readlines()
                files_processed.add(filename)
            except Exception as e:
                print(f"Error reading file {filename}: {e}")
                skipped_count += 1
                continue
        
        # Extract the function code (line numbers in analysis are 1-based)
        lines = file_cache[filename]
        if start_row <= len(lines):
            # Extract lines and join them
            function_code = ''.join(lines[start_row-1:end_row])
            
            # Create a hash of the code for duplicate detection
            code_hash = hashlib.md5(function_code.encode()).hexdigest()
            
            # Check for duplicate code content
            if code_hash in function_code_hashes:
                duplicate_count += 1
                duplicate_by_hash += 1
                continue
                
            # Add hash to our tracking set
            function_code_hashes.add(code_hash)
            
            # Create a result object with metadata and code
            result = {
                'name': func_name,
                'namespace': namespace,
                'doc': func_def.get('doc', ''),
                'filename': filename,
                'start_row': start_row,
                'end_row': end_row,
                'private': func_def.get('private', False),
                'code': function_code,
                'code_hash': code_hash
            }
            
            function_chunks.append(result)
            chunk_count += 1
    
    # Now process defmethod functions from var-usages
    if 'var-usages' in analysis:
        # Filter var-usages to just defmethod entries
        defmethod_usages = [usage for usage in analysis.get('var-usages', []) 
                           if usage.get('name') == 'defmethod']
        
        for i, defmethod in enumerate(defmethod_usages):
            # Report progress
            if i % 20 == 0 or i == len(defmethod_usages) - 1:
                print(f"Processing defmethod functions: {i+1}/{len(defmethod_usages)} ({((i+1)/len(defmethod_usages))*100:.1f}%)")
            
            # Extract metadata fields
            filename = os.path.join(rel_dir, defmethod.get('filename'))
            if not filename:
                skipped_count += 1
                continue
                
            # Extract line numbers
            start_row = defmethod.get('row', 0)
            end_row = defmethod.get('end-row', 0)
            
            if start_row == 0 or end_row == 0 or start_row > end_row:
                print(f"Warning: Invalid row range for defmethod in {filename}")
                skipped_count += 1
                continue
            
            # Read the file if not already cached
            if filename not in file_cache:
                try:
                    with open(filename, 'r') as f:
                        file_cache[filename] = f.readlines()
                    files_processed.add(filename)
                except Exception as e:
                    print(f"Error reading file {filename}: {e}")
                    skipped_count += 1
                    continue
            
            # Extract the function code (line numbers in analysis are 1-based)
            lines = file_cache[filename]
            if start_row <= len(lines):
                # Extract lines and join them
                function_code = ''.join(lines[start_row-1:end_row])
                
                # Parse the defmethod line to get function name and dispatch value
                defmethod_line = lines[start_row-1].strip()
                
                # Extract function name and dispatch value from the defmethod line
                # Expected format: (mu/defmethod render :attached :- ::RenderedPartCard
                # or similar patterns with the function name after defmethod and dispatch value after that
                parts = defmethod_line.split()
                if len(parts) >= 3:  # Need at least "(mu/defmethod function-name dispatch-value"
                    # Find the index of defmethod in the parts
                    defmethod_index = -1
                    for i, part in enumerate(parts):
                        if part.endswith('defmethod'):
                            defmethod_index = i
                            break
                    
                    if defmethod_index >= 0 and defmethod_index + 2 < len(parts):
                        function_name = parts[defmethod_index + 1]
                        dispatch_value = parts[defmethod_index + 2]
                        
                        # Remove any extraneous characters like parentheses
                        function_name = function_name.strip('()')
                        dispatch_value = dispatch_value.strip('()')
                        
                        # Full name of the defmethod function
                        full_name = f"{function_name} {dispatch_value}"
                        
                        # Get the namespace
                        namespace = defmethod.get('from', '')
                        
                        # Generate a unique key for this function
                        func_key = f"{namespace}::{full_name}::{filename}::{start_row}::{end_row}"
                        
                        # Check if we've already processed this function
                        if func_key in function_keys:
                            duplicate_count += 1
                            continue
                        
                        # Add to the set of processed functions
                        function_keys.add(func_key)
                        
                        # Create a hash of the code for duplicate detection
                        code_hash = hashlib.md5(function_code.encode()).hexdigest()
                        
                        # Check for duplicate code content
                        if code_hash in function_code_hashes:
                            duplicate_count += 1
                            duplicate_by_hash += 1
                            continue
                            
                        # Add hash to our tracking set
                        function_code_hashes.add(code_hash)
                        
                        # Create a result object with metadata and code
                        result = {
                            'name': full_name,
                            'namespace': namespace,
                            'doc': '',  # defmethods don't typically have doc in the analysis
                            'filename': filename,
                            'start_row': start_row,
                            'end_row': end_row,
                            'private': False,  # defmethods are typically not private
                            'code': function_code,
                            'dispatch_value': dispatch_value,  # Include dispatch value as additional metadata
                            'code_hash': code_hash
                        }
                        
                        function_chunks.append(result)
                        chunk_count += 1
                    else:
                        skipped_count += 1
                else:
                    skipped_count += 1
    
    # Calculate and report execution stats
    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"\nExtraction completed in {execution_time:.2f} seconds")
    print(f"Extracted {chunk_count} unique function chunks from {len(files_processed)} files")
    print(f"- {var_def_count} standard functions processed")
    print(f"- {defmethod_count} defmethod functions processed")
    print(f"- {duplicate_count} duplicates detected and skipped")
    if duplicate_by_hash > 0:
        print(f"  - {duplicate_by_hash} duplicates detected by content hash")
    if skipped_count > 0:
        print(f"- {skipped_count} functions skipped due to errors or invalid data")
    
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
    start_time = time.time()
    print(f"Starting ChromaDB population process...")
    
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
    skipped_count = 0
    
    print(f"Processing {len(df)} code chunks for ChromaDB...")
    
    for i, row in df.iterrows():
        # Validate embedding
        if not row.get('embedding'):
            print(f"Warning: Missing embedding for {row.get('namespace')}/{row.get('name')}. Skipping.")
            skipped_count += 1
            continue

        # Generate a unique ID using a hash of the code
        code_hash = hashlib.md5(row['code'].encode()).hexdigest()
        doc_id = f"{row['namespace']}.{row['name']}_{i}_{code_hash[:8]}"
        
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
        if i % 100 == 0 or i == len(df) - 1:
            print(f"Processing: {i+1}/{len(df)} ({((i+1)/len(df))*100:.1f}%)")
            
        ids.append(doc_id)
        embeddings.append(row['embedding'])
        documents.append(document_text)
        metadatas.append(metadata)

    # Add to ChromaDB in batches (to avoid potential memory issues)
    batch_size = 500
    total_batches = (len(ids) + batch_size - 1) // batch_size
    
    print(f"Adding data to ChromaDB in {total_batches} batches...")
    batch_start_time = time.time()
    
    for i in range(0, len(ids), batch_size):
        end_idx = min(i + batch_size, len(ids))
        batch_ids = ids[i:end_idx]
        batch_embeddings = embeddings[i:end_idx]
        batch_documents = documents[i:end_idx]
        batch_metadatas = metadatas[i:end_idx]
        
        batch_num = (i//batch_size) + 1
        print(f"Adding batch {batch_num}/{total_batches} ({len(batch_ids)} items)...")
        
        # Add batch to collection
        collection.add(
            ids=batch_ids,
            embeddings=batch_embeddings,
            documents=batch_documents,
            metadatas=batch_metadatas
        )
        
        # Report batch time if multiple batches
        if total_batches > 1:
            batch_end_time = time.time()
            batch_time = batch_end_time - batch_start_time
            print(f"Batch {batch_num} completed in {batch_time:.2f} seconds")
            batch_start_time = time.time()
    
    # Calculate and report execution stats
    end_time = time.time()
    execution_time = end_time - start_time
    
    final_count = collection.count()
    print(f"\nChromaDB population completed in {execution_time:.2f} seconds")
    print(f"Successfully populated ChromaDB collection '{collection_name}' with {final_count} items")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} items due to missing embeddings")
    
    return collection


# The search_code function has been moved to search.py


def save_function_chunks(function_chunks, output_file="chunks.json"):
    """
    Save function chunks to a JSON file.
    
    Args:
        function_chunks (list): List of dictionaries containing function chunks
        output_file (str): Path to output file
    """
    print(f"Saving {len(function_chunks)} function chunks to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(function_chunks, f, indent=2)
    print(f"Successfully saved {len(function_chunks)} function chunks to {output_file}")

def load_function_chunks(chunks_file):
    """
    Load function chunks from a JSON file.
    
    Args:
        chunks_file (str): Path to JSON file containing function chunks
        
    Returns:
        List of dictionaries containing function chunks
    """
    print(f"Loading function chunks from {chunks_file}...")
    with open(chunks_file, 'r') as f:
        chunks = json.load(f)
    print(f"Successfully loaded {len(chunks)} function chunks from {chunks_file}")
    return chunks

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process clj-kondo analysis data to extract function chunks, generate embeddings, and search code')
    
    # Create subparsers for different operations
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Extract command - only extracts function chunks without generating embeddings
    extract_parser = subparsers.add_parser('extract', help='Extract function chunks from clj-kondo analysis')
    extract_parser.add_argument('analysis_file', help='Path to the JSON file containing clj-kondo analysis data')
    extract_parser.add_argument('--output', default='chunks.json', help='Output file path for function chunks JSON')
    extract_parser.add_argument('--rel-dir', default='../..', help='Relative directory path to prepend to filenames')
    
    # Embed command - takes chunks and generates embeddings
    embed_parser = subparsers.add_parser('embed', help='Generate embeddings from function chunks')
    embed_parser.add_argument('chunks_file', help='Path to the JSON file containing function chunks')
    embed_parser.add_argument('--output', default='embeddings.json', help='Output file path for embeddings JSON')
    embed_parser.add_argument('--model', default='text-embedding-3-small', help='OpenAI model to use for embeddings')
    embed_parser.add_argument('--batch-size', type=int, default=100, help='Number of texts to batch in a single API call')
    
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
        
        if function_chunks:
            # Save function chunks to file
            save_function_chunks(function_chunks, output_file=args.output)
            
            print(f"\nTo generate embeddings from these chunks, run:")
            print(f"python {os.path.basename(__file__)} embed {args.output}")

    elif args.command == 'embed':
        # Load function chunks from file
        function_chunks = load_function_chunks(args.chunks_file)
        
        if function_chunks:
            # Generate embeddings for the loaded chunks
            df = generate_embeddings(
                function_chunks, 
                output_file=args.output, 
                model=args.model,
                batch_size=args.batch_size
            )
            
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
