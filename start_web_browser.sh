#!/bin/bash

echo "🌐 ChromaDB Web Browser Options for RAG System"
echo "=============================================="
echo ""
echo "📊 Your Database Summary:"
echo "   Location: /Users/kweng/AI/RAG/data/indices/chroma_db"
echo "   Size: 182 MB"
echo "   Documents: 251"
echo "   Chunks: 13,565"
echo ""
echo "🚀 Available Web Interfaces:"
echo ""
echo "1. 🎯 Simple Python Browser (Recommended - No Docker needed)"
echo "   python /Users/kweng/AI/RAG/simple_chromadb_browser.py"
echo "   Access: http://localhost:5000"
echo ""
echo "2. 🔧 ChromaDB Admin (Requires ChromaDB HTTP server)"
echo "   Terminal 1: python /Users/kweng/AI/RAG/start_chromadb_server.py"
echo "   Terminal 2: cd chromadb-admin && npm run dev"
echo "   Access: http://localhost:3001"
echo ""
echo "3. 📋 Command Line Tools"
echo "   python /Users/kweng/AI/RAG/inspect_chromadb.py"
echo "   python /Users/kweng/AI/RAG/generate_ingestion_report.py"
echo ""
echo "Choose an option:"
echo "1) Start Simple Python Browser"
echo "2) Start ChromaDB Server (for admin interface)"
echo "3) View CLI inspection"
echo "4) Generate full report"
echo ""
read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        echo "🚀 Starting Simple Python Browser..."
        python /Users/kweng/AI/RAG/simple_chromadb_browser.py
        ;;
    2)
        echo "🚀 Starting ChromaDB Server..."
        echo "💡 After this starts, run in another terminal:"
        echo "   cd chromadb-admin && npm run dev"
        python /Users/kweng/AI/RAG/start_chromadb_server.py
        ;;
    3)
        echo "🔍 Running database inspection..."
        python /Users/kweng/AI/RAG/inspect_chromadb.py
        ;;
    4)
        echo "📊 Generating full report..."
        python /Users/kweng/AI/RAG/generate_ingestion_report.py
        ;;
    *)
        echo "❌ Invalid choice. Please run the script again."
        ;;
esac