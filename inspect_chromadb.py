#!/usr/bin/env python3
"""
Simple script to inspect ChromaDB database contents.
Shows ingested documents, collection stats, and sample metadata.
"""

import chromadb
from pathlib import Path
import json
from datetime import datetime

def inspect_chromadb():
    """Inspect the ChromaDB database and show ingested documents."""
    
    # Connect to ChromaDB
    db_path = "/Users/kweng/AI/RAG/data/indices/chroma_db"
    client = chromadb.PersistentClient(path=db_path)
    
    # Get collection
    collection_name = "rag_documents"
    try:
        collection = client.get_collection(name=collection_name)
        print(f"✅ Connected to collection: {collection_name}")
    except Exception as e:
        print(f"❌ Error connecting to collection: {e}")
        return
    
    # Get collection stats
    count = collection.count()
    print(f"📊 Total chunks in collection: {count:,}")
    
    if count == 0:
        print("❌ No documents found in collection")
        return
    
    # Get sample documents to inspect structure
    print("\n🔍 Inspecting sample documents...")
    results = collection.get(
        limit=10,
        include=['metadatas', 'documents']
    )
    
    # Extract unique document sources
    documents = {}
    total_chunks = 0
    
    for i, metadata in enumerate(results['metadatas']):
        if metadata:
            filename = metadata.get('filename', 'Unknown')
            source_path = metadata.get('source_path', 'Unknown')
            
            if filename not in documents:
                documents[filename] = {
                    'source_path': source_path,
                    'chunks': 0,
                    'file_size': metadata.get('file_size', 0),
                    'created_at': metadata.get('created_at', 'Unknown'),
                    'modified_at': metadata.get('modified_at', 'Unknown'),
                    'file_type': metadata.get('file_type', 'Unknown')
                }
            documents[filename]['chunks'] += 1
            total_chunks += 1
    
    print(f"\n📁 Sample Documents from first 10 chunks:")
    print("=" * 80)
    for filename, info in sorted(documents.items()):
        print(f"📄 {filename}")
        print(f"   📍 Path: {info['source_path']}")
        print(f"   🧩 Chunks (in sample): {info['chunks']}")
        print(f"   📊 Size: {info['file_size']:,} bytes")
        print(f"   📅 Modified: {info['modified_at']}")
        print(f"   🗂️  Type: {info['file_type']}")
        print()
    
    # Get all unique filenames in the entire collection
    print("\n📋 Getting all documents in collection...")
    all_results = collection.get(
        include=['metadatas']
    )
    
    all_documents = {}
    for metadata in all_results['metadatas']:
        if metadata and 'filename' in metadata:
            filename = metadata['filename']
            if filename not in all_documents:
                all_documents[filename] = {
                    'chunks': 0,
                    'file_size': metadata.get('file_size', 0),
                    'file_type': metadata.get('file_type', 'Unknown'),
                    'source_path': metadata.get('source_path', 'Unknown')
                }
            all_documents[filename]['chunks'] += 1
    
    print(f"\n📈 Complete Document Summary:")
    print("=" * 80)
    print(f"Total unique documents: {len(all_documents):,}")
    print(f"Total chunks: {count:,}")
    print(f"Average chunks per document: {count/len(all_documents):.1f}")
    
    # Show document types
    file_types = {}
    total_size = 0
    for doc_info in all_documents.values():
        file_type = doc_info['file_type']
        file_size = doc_info['file_size']
        if file_type not in file_types:
            file_types[file_type] = {'count': 0, 'size': 0}
        file_types[file_type]['count'] += 1
        file_types[file_type]['size'] += file_size
        total_size += file_size
    
    print(f"\n📊 Document Types:")
    print("-" * 50)
    for file_type, info in sorted(file_types.items()):
        print(f"{file_type:>10}: {info['count']:>3} files ({info['size']/1024/1024:.1f} MB)")
    
    print(f"\n💾 Storage Summary:")
    print("-" * 50)
    print(f"Original documents size: {total_size/1024/1024:.1f} MB")
    print(f"ChromaDB size: 182 MB")
    print(f"Compression ratio: {(total_size/1024/1024)/182:.2f}x")
    
    # Show recent documents (by creation time)
    print(f"\n🕒 Recent Documents (top 10):")
    print("-" * 80)
    doc_list = []
    for filename, info in all_documents.items():
        doc_list.append((filename, info['chunks'], info['file_size']))
    
    # Sort by chunks (as proxy for document importance)
    doc_list.sort(key=lambda x: x[1], reverse=True)
    
    for i, (filename, chunks, size) in enumerate(doc_list[:10], 1):
        print(f"{i:2}. {filename[:60]:<60} ({chunks:>3} chunks, {size/1024:.0f} KB)")
    
    print(f"\n✅ Database inspection complete!")

if __name__ == "__main__":
    inspect_chromadb()