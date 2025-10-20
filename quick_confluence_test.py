#!/usr/bin/env python3
"""Quick test of Confluence POC with proper limiting - just first space."""

import sys
import os
from pathlib import Path

# Load .env.confluence file (preferred) or .env file
env_files = [".env.confluence", ".env"]
for env_file_name in env_files:
    env_file = Path(env_file_name)
    if env_file.exists():
        with open(env_file, 'r') as f:
            confluence_vars = 0
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
                    if key.startswith('CONFLUENCE_'):
                        confluence_vars += 1
        if confluence_vars > 0:
            break

from confluence_poc import ConfluencePOC, get_confluence_config

def main():
    print("🧪 Quick Confluence Test - First Space Only")
    print("=" * 45)
    
    config = get_confluence_config()
    print(f"Max pages setting: {config['max_pages']}")
    print(f"Testing space: {config['spaces'][0]}")
    
    # Create POC instance
    poc = ConfluencePOC(config)
    
    # Test connection
    if not poc.connect():
        print("❌ Connection failed")
        return
    
    # Process just the first space
    first_space = config['spaces'][0]
    print(f"\n🕷️  Testing {first_space} with max_pages={config['max_pages']}")
    
    # Get pages (this will show the limiting behavior)
    pages_data = poc.get_pages_from_space(first_space, config['max_pages'])
    
    # Process pages
    processed_pages = []
    for page_data in pages_data:
        processed_page = poc.process_page(page_data)
        if processed_page:
            processed_pages.append(processed_page)
    
    # Summary
    print(f"\n📊 Summary:")
    print(f"   Configured limit: {config['max_pages']}")
    print(f"   Pages retrieved: {len(pages_data)}")
    print(f"   Pages processed: {len(processed_pages)}")
    
    if len(processed_pages) == config['max_pages']:
        print("✅ CONFLUENCE_MAX_PAGES setting is working correctly!")
    else:
        print("⚠️  Page limiting may not be working as expected")
    
    # Show sample
    if processed_pages:
        print(f"\n📝 Sample pages:")
        for i, page in enumerate(processed_pages[:3], 1):
            print(f"   {i}. {page.title} ({page.word_count} words)")

if __name__ == "__main__":
    main()