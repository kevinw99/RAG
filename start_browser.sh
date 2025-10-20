#!/bin/bash

echo "🌐 ChromaDB Web Browser Launcher"
echo "================================"
echo ""
echo "🚀 Starting ChromaDB Browser..."
echo "📊 Database: /Users/kweng/AI/RAG/data/indices/chroma_db"
echo "📁 Documents: 251 files"
echo "🧩 Chunks: 13,565"
echo "💾 Size: 182 MB"
echo ""

cd /Users/kweng/AI/RAG

# Start the browser
python simple_chromadb_browser.py

echo ""
echo "👋 ChromaDB Browser stopped."