#!/usr/bin/env python3
"""
Simple web-based ChromaDB browser using Flask.
Provides a basic interface to browse your ChromaDB data.
"""

from flask import Flask, render_template_string, request, jsonify
import chromadb
import json
from datetime import datetime
import math

app = Flask(__name__)

# Connect to ChromaDB
DB_PATH = "/Users/kweng/AI/RAG/data/indices/chroma_db"
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection(name="rag_documents")

# HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>ChromaDB Browser - RAG System</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header { background: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 20px; }
        .stat-card { background: #ecf0f1; padding: 15px; border-radius: 6px; text-align: center; }
        .stat-number { font-size: 2em; font-weight: bold; color: #3498db; }
        .search-box { margin: 20px 0; }
        .search-box input { width: 300px; padding: 10px; border: 1px solid #ddd; border-radius: 4px; }
        .search-box button { padding: 10px 20px; background: #3498db; color: white; border: none; border-radius: 4px; cursor: pointer; }
        .documents { margin-top: 20px; }
        .doc-item { background: #fff; border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 6px; }
        .doc-title { font-weight: bold; color: #2c3e50; margin-bottom: 5px; }
        .doc-meta { color: #666; font-size: 0.9em; }
        .doc-preview { margin-top: 10px; padding: 10px; background: #f8f9fa; border-left: 4px solid #3498db; font-family: monospace; font-size: 0.85em; }
        .pagination { text-align: center; margin: 20px 0; }
        .pagination a { display: inline-block; padding: 8px 12px; margin: 0 2px; background: #3498db; color: white; text-decoration: none; border-radius: 4px; }
        .pagination .current { background: #e74c3c; }
        .file-type { display: inline-block; padding: 2px 6px; background: #95a5a6; color: white; border-radius: 3px; font-size: 0.8em; margin-right: 5px; }
        .chunk-count { color: #27ae60; font-weight: bold; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 ChromaDB Browser - RAG System</h1>
            <p>Database: {{ db_path }}</p>
            <p>Collection: rag_documents</p>
        </div>
        
        <div class="stats">
            <div class="stat-card">
                <div class="stat-number">{{ total_chunks | format_number }}</div>
                <div>Total Chunks</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{{ total_docs | format_number }}</div>
                <div>Documents</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{{ avg_chunks }}</div>
                <div>Avg Chunks/Doc</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">182 MB</div>
                <div>Database Size</div>
            </div>
        </div>
        
        <div class="search-box">
            <form method="GET">
                <input type="text" name="search" placeholder="Search documents..." value="{{ search_query }}">
                <button type="submit">🔍 Search</button>
                <a href="/" style="margin-left: 10px;">📋 Show All</a>
            </form>
        </div>
        
        <div class="documents">
            <h3>📁 Documents (Page {{ page }} of {{ total_pages }})</h3>
            {% for doc in documents %}
            <div class="doc-item">
                <div class="doc-title">
                    <span class="file-type">{{ doc.file_type.upper() }}</span>
                    {{ doc.filename }}
                </div>
                <div class="doc-meta">
                    <span class="chunk-count">{{ doc.chunks }} chunks</span> | 
                    {{ (doc.file_size / 1024) | round | int }} KB | 
                    Modified: {{ doc.modified_at[:10] if doc.modified_at != 'Unknown' else 'Unknown' }}
                </div>
                {% if doc.preview %}
                <div class="doc-preview">{{ doc.preview }}...</div>
                {% endif %}
            </div>
            {% endfor %}
        </div>
        
        {% if total_pages > 1 %}
        <div class="pagination">
            {% for p in range(1, total_pages + 1) %}
                {% if p == page %}
                    <a href="?page={{ p }}&search={{ search_query }}" class="current">{{ p }}</a>
                {% else %}
                    <a href="?page={{ p }}&search={{ search_query }}">{{ p }}</a>
                {% endif %}
            {% endfor %}
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.template_filter('format_number')
def format_number(value):
    """Format numbers with commas."""
    return f"{value:,}"

@app.route('/')
def index():
    # Get parameters
    page = int(request.args.get('page', 1))
    search_query = request.args.get('search', '').strip()
    per_page = 20
    
    # Get all data
    all_results = collection.get(include=['metadatas', 'documents'])
    
    # Process documents
    documents_dict = {}
    for i, metadata in enumerate(all_results['metadatas']):
        if not metadata:
            continue
            
        filename = metadata.get('filename', f'unknown_{i}')
        
        if filename not in documents_dict:
            documents_dict[filename] = {
                'filename': filename,
                'chunks': 0,
                'file_size': metadata.get('file_size', 0),
                'file_type': filename.split('.')[-1].lower() if '.' in filename else 'unknown',
                'modified_at': metadata.get('modified_at', 'Unknown'),
                'preview': None
            }
        
        documents_dict[filename]['chunks'] += 1
        
        # Add preview from first chunk
        if documents_dict[filename]['preview'] is None and i < len(all_results['documents']):
            preview_text = all_results['documents'][i] or ''
            documents_dict[filename]['preview'] = preview_text[:200] if preview_text else ''
    
    # Convert to list and filter by search
    documents_list = list(documents_dict.values())
    
    if search_query:
        documents_list = [
            doc for doc in documents_list 
            if search_query.lower() in doc['filename'].lower() or 
               (doc['preview'] and search_query.lower() in doc['preview'].lower())
        ]
    
    # Sort by chunk count (most content first)
    documents_list.sort(key=lambda x: x['chunks'], reverse=True)
    
    # Pagination
    total_docs = len(documents_list)
    total_pages = math.ceil(total_docs / per_page)
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    page_documents = documents_list[start_idx:end_idx]
    
    return render_template_string(
        HTML_TEMPLATE,
        documents=page_documents,
        total_chunks=len(all_results['ids']),
        total_docs=len(documents_dict),
        avg_chunks=round(len(all_results['ids']) / len(documents_dict), 1),
        db_path=DB_PATH,
        page=page,
        total_pages=total_pages,
        search_query=search_query
    )

@app.route('/api/search')
def api_search():
    """API endpoint for searching documents."""
    query = request.args.get('q', '')
    limit = int(request.args.get('limit', 10))
    
    if not query:
        return jsonify({'error': 'No query provided'})
    
    try:
        # Perform similarity search
        results = collection.query(
            query_texts=[query],
            n_results=limit,
            include=['metadatas', 'documents', 'distances']
        )
        
        return jsonify({
            'query': query,
            'results': results,
            'count': len(results['ids'][0]) if results['ids'] else 0
        })
    except Exception as e:
        return jsonify({'error': str(e)})

def find_available_port(start_port=5000, max_attempts=10):
    """Find an available port starting from start_port."""
    import socket
    
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except OSError:
            continue
    
    raise RuntimeError(f"Could not find available port in range {start_port}-{start_port + max_attempts}")

if __name__ == '__main__':
    import sys
    
    # Check for port argument
    port = 5000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print("❌ Invalid port number. Usage: python simple_chromadb_browser.py [port]")
            sys.exit(1)
    
    # Find available port if default is busy
    try:
        available_port = find_available_port(port)
        if available_port != port:
            print(f"⚠️  Port {port} is busy, using port {available_port} instead")
            port = available_port
    except RuntimeError as e:
        print(f"❌ {e}")
        sys.exit(1)
    
    print("🚀 Starting Simple ChromaDB Browser...")
    print(f"📁 Database: {DB_PATH}")
    print(f"🌐 Web Interface: http://localhost:{port}")
    print(f"🔍 API Search: http://localhost:{port}/api/search?q=your_query")
    print()
    print("💡 Usage: python simple_chromadb_browser.py [port]")
    print("🛑 Press Ctrl+C to stop")
    print("=" * 50)
    
    app.run(host='localhost', port=port, debug=False)