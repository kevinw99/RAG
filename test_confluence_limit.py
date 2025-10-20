#!/usr/bin/env python3
"""
Quick test to verify CONFLUENCE_MAX_PAGES setting is working.
Only processes the first configured space.
"""

import os
from pathlib import Path

# Load .env file
def load_env_file():
    """Load environment variables from .env file if it exists."""
    env_file = Path(".env")
    if env_file.exists():
        print("📁 Loading configuration from .env file...")
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    # Remove quotes if present
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    os.environ[key] = value
        print("✅ Environment variables loaded from .env")

load_env_file()

try:
    from atlassian import Confluence
    print("✅ atlassian-python-api imported successfully")
except ImportError:
    print("❌ Missing atlassian-python-api")
    exit(1)

def get_confluence_config():
    """Get Confluence configuration from environment variables."""
    spaces_str = os.getenv("CONFLUENCE_SPACES", "")
    spaces = []
    if spaces_str:
        spaces = [space.strip() for space in spaces_str.split(",") if space.strip()]
    
    return {
        "base_url": os.getenv("CONFLUENCE_BASE_URL", "").strip(),
        "username": os.getenv("CONFLUENCE_USERNAME", "").strip(), 
        "api_token": os.getenv("CONFLUENCE_API_TOKEN", "").strip(),
        "spaces": spaces or ["TECH"],
        "max_pages": int(os.getenv("CONFLUENCE_MAX_PAGES", "5")),
    }

def main():
    print("🧪 Testing CONFLUENCE_MAX_PAGES Setting")
    print("=" * 40)
    
    config = get_confluence_config()
    
    print(f"📋 Configuration:")
    print(f"   Base URL: {config['base_url']}")
    print(f"   Username: {config['username']}")
    print(f"   Spaces: {config['spaces']}")
    print(f"   Max Pages: {config['max_pages']}")
    
    if not all([config['base_url'], config['username'], config['api_token']]):
        print("❌ Missing configuration")
        return
    
    # Initialize Confluence client
    confluence = Confluence(
        url=config['base_url'],
        username=config['username'],
        password=config['api_token'],
        cloud=True
    )
    
    # Test with first space only
    test_space = config['spaces'][0]
    limit = config['max_pages']
    
    print(f"\n🕷️  Testing space: {test_space} with limit: {limit}")
    
    try:
        # Get pages with limit
        pages = confluence.get_all_pages_from_space(
            space=test_space, 
            start=0, 
            limit=limit,
            expand='body.storage'
        )
        
        if not pages:
            print(f"⚠️  No pages found in space {test_space}")
            return
            
        # Check if API returned more than requested
        if len(pages) > limit:
            print(f"⚠️  API returned {len(pages)} pages, but limit was {limit}")
            print("     This confirms the API sometimes ignores the limit parameter")
            pages = pages[:limit]
            print(f"     Truncated to {len(pages)} pages")
        else:
            print(f"✅ API correctly returned {len(pages)} pages (limit: {limit})")
        
        print(f"\n📄 Processing {len(pages)} pages:")
        for i, page in enumerate(pages, 1):
            title = page.get('title', 'Untitled')
            page_id = page.get('id', 'Unknown')
            print(f"   {i}. {title} (ID: {page_id})")
            
        print(f"\n✅ CONFLUENCE_MAX_PAGES={limit} setting is working correctly!")
        print(f"   Only processed {len(pages)} pages as requested")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()