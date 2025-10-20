#!/usr/bin/env python3
"""
Setup script for Confluence POC

Installs required dependencies and helps with configuration.
"""

import subprocess
import sys
import os
from pathlib import Path

def install_dependencies():
    """Install required packages for Confluence POC."""
    print("📦 Installing Confluence POC dependencies...")
    
    packages = [
        "atlassian-python-api",
        "beautifulsoup4", 
        "lxml"  # Better HTML parser for BeautifulSoup
    ]
    
    for package in packages:
        try:
            print(f"   Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"   ✅ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Failed to install {package}: {e}")
            return False
    
    return True

def create_env_template():
    """Create .env template for Confluence configuration."""
    env_template = """# Confluence POC Configuration
# Copy this to .env and update with your actual values

# Your Confluence URL (include https://)
CONFLUENCE_BASE_URL=https://your-company.atlassian.net

# Your Confluence username (usually your email)
CONFLUENCE_USERNAME=your.email@company.com

# Your Confluence API token (generate at https://id.atlassian.com/manage/api-tokens)
CONFLUENCE_API_TOKEN=your-api-token-here

# Comma-separated list of space keys to crawl
CONFLUENCE_SPACES=TECH,DOCS,HELP

# Maximum pages to crawl per space (for testing)
CONFLUENCE_MAX_PAGES=5

# Optional: Corporate proxy settings (if needed)
# HTTP_PROXY=http://proxy.company.com:8080
# HTTPS_PROXY=https://proxy.company.com:8080
"""
    
    env_file = Path(".env.confluence.template")
    with open(env_file, "w") as f:
        f.write(env_template)
    
    print(f"📝 Created configuration template: {env_file}")
    print("   Copy this file to .env and update with your Confluence details")

def check_existing_config():
    """Check if configuration already exists."""
    env_file = Path(".env")
    if env_file.exists():
        print("✅ Found existing .env file")
        
        # Check if it has Confluence settings
        with open(env_file) as f:
            content = f.read()
            
        has_confluence = any(key in content for key in [
            "CONFLUENCE_BASE_URL", 
            "CONFLUENCE_USERNAME",
            "CONFLUENCE_API_TOKEN"
        ])
        
        if has_confluence:
            print("✅ Confluence configuration found in .env")
            return True
        else:
            print("⚠️  .env exists but no Confluence configuration found")
            print("   Add Confluence settings to your .env file")
            return False
    else:
        print("📝 No .env file found")
        return False

def test_imports():
    """Test if required packages can be imported."""
    print("\n🧪 Testing package imports...")
    
    try:
        from atlassian import Confluence
        print("✅ atlassian-python-api imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import atlassian-python-api: {e}")
        return False
        
    try:
        from bs4 import BeautifulSoup
        print("✅ beautifulsoup4 imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import beautifulsoup4: {e}")
        return False
        
    return True

def main():
    """Main setup process."""
    print("🚀 Confluence POC Setup")
    print("=" * 40)
    
    # Step 1: Install dependencies
    if not install_dependencies():
        print("❌ Dependency installation failed")
        return 1
    
    # Step 2: Test imports
    if not test_imports():
        print("❌ Package import test failed")
        return 1
    
    # Step 3: Check/create configuration
    if not check_existing_config():
        create_env_template()
        print("\n📋 Next steps:")
        print("1. Copy .env.confluence.template to .env")
        print("2. Update .env with your Confluence details:")
        print("   - Get API token: https://id.atlassian.com/manage/api-tokens")
        print("   - Find your Confluence URL (usually company.atlassian.net)")
        print("   - Use your email as username")
        print("3. Run: python confluence_poc.py")
    else:
        print("\n✅ Configuration looks good!")
        print("Ready to run: python confluence_poc.py")
    
    print("\n🔗 Helpful links:")
    print("   • Confluence API tokens: https://id.atlassian.com/manage/api-tokens")
    print("   • API documentation: https://developer.atlassian.com/cloud/confluence/rest/v2/intro/")
    print("   • Python library docs: https://atlassian-python-api.readthedocs.io/")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())