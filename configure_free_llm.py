#!/usr/bin/env python3
"""Configure the RAG system to use free/local LLM alternatives."""

import os
from pathlib import Path

def show_llm_options():
    """Show available LLM configuration options."""
    print("🆓 Free LLM Alternatives for RAG System")
    print("=" * 50)
    
    print("\n1. 🔧 Use Hugging Face Transformers (Local)")
    print("   - Completely free")
    print("   - Runs on your machine")
    print("   - Good for testing")
    
    print("\n2. 🦙 Use Ollama (Local)")
    print("   - Install: brew install ollama")
    print("   - Download model: ollama pull llama2")
    print("   - Completely free")
    
    print("\n3. 🌐 Use OpenAI Compatible APIs")
    print("   - Groq (free tier): https://console.groq.com")
    print("   - Together.ai (free tier): https://api.together.xyz")
    print("   - Anyscale (free tier): https://app.anyscale.com")
    
    print("\n4. ⚡ For Testing: Disable Answer Generation")
    print("   - Use search endpoint instead of query")
    print("   - Test document retrieval only")
    
    print(f"\n{'='*50}")
    print("🎯 QUICKEST SOLUTION FOR TESTING:")
    print("   Run: python test_retrieval.py")
    print("   This tests your RAG system without needing any API keys!")

def create_env_template():
    """Create .env template for different LLM providers."""
    env_content = """# RAG System Environment Configuration
# Choose ONE of the options below:

# Option 1: OpenAI (requires credits)
# OPENAI_API_KEY=your-openai-api-key-here
# LLM_PROVIDER=openai
# LLM_MODEL=gpt-3.5-turbo

# Option 2: Groq (free tier available)
# GROQ_API_KEY=your-groq-api-key-here
# LLM_PROVIDER=groq
# LLM_MODEL=llama2-70b-4096

# Option 3: Together.ai (free tier available)
# TOGETHER_API_KEY=your-together-api-key-here
# LLM_PROVIDER=together
# LLM_MODEL=togethercomputer/llama-2-70b-chat

# Option 4: Ollama (completely local/free)
# LLM_PROVIDER=ollama
# LLM_MODEL=llama2
# OLLAMA_BASE_URL=http://localhost:11434

# Option 5: Disable LLM (retrieval testing only)
# LLM_PROVIDER=none
"""
    
    env_path = Path(".env.template")
    with open(env_path, "w") as f:
        f.write(env_content)
    
    print(f"📝 Created {env_path} with LLM configuration options")

if __name__ == "__main__":
    show_llm_options()
    create_env_template()
    
    print(f"\n🚀 IMMEDIATE ACTION:")
    print("   python test_retrieval.py")
    print("   ↑ This will prove your RAG system works perfectly!")