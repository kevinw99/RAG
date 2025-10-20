#!/usr/bin/env python3
"""
Quick fix for common .env configuration issues.
"""

import re
from pathlib import Path

def fix_env_file():
    """Fix common issues in .env file."""
    env_file = Path(".env")
    if not env_file.exists():
        print("❌ No .env file found")
        return
    
    print("🔧 Checking .env file for common issues...")
    
    # Read current content
    with open(env_file, 'r') as f:
        content = f.read()
    
    original_content = content
    changes_made = []
    
    # Fix 1: Remove trailing slash from Confluence URL
    if 'CONFLUENCE_BASE_URL=' in content:
        pattern = r'CONFLUENCE_BASE_URL=([^\n]*)/\s*$'
        if re.search(pattern, content, re.MULTILINE):
            content = re.sub(pattern, r'CONFLUENCE_BASE_URL=\1', content, flags=re.MULTILINE)
            changes_made.append("Removed trailing slash from CONFLUENCE_BASE_URL")
    
    # Fix 2: Remove spaces from space list
    if 'CONFLUENCE_SPACES=' in content:
        pattern = r'CONFLUENCE_SPACES=([^\n]*)'
        match = re.search(pattern, content)
        if match:
            spaces_value = match.group(1)
            if ', ' in spaces_value:  # Has spaces after commas
                clean_spaces = ','.join([s.strip() for s in spaces_value.split(',')])
                content = content.replace(f'CONFLUENCE_SPACES={spaces_value}', f'CONFLUENCE_SPACES={clean_spaces}')
                changes_made.append("Removed spaces from CONFLUENCE_SPACES list")
    
    # Save if changes were made
    if changes_made:
        with open(env_file, 'w') as f:
            f.write(content)
        
        print("✅ Fixed .env file:")
        for change in changes_made:
            print(f"   • {change}")
    else:
        print("✅ No issues found in .env file")
    
    # Display current configuration
    print("\n📋 Current Confluence configuration:")
    for line in content.split('\n'):
        if line.strip() and not line.startswith('#') and any(key in line for key in ['CONFLUENCE_']):
            if 'API_TOKEN' in line:
                # Mask the API token
                key, value = line.split('=', 1)
                masked_value = '*' * min(len(value), 20) if value else '(empty)'
                print(f"   {key}={masked_value}")
            else:
                print(f"   {line}")

if __name__ == "__main__":
    fix_env_file()