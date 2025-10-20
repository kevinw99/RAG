#!/usr/bin/env python3
"""
Test Confluence setup without requiring real credentials.

This script validates that all dependencies are installed correctly
and helps diagnose common setup issues.
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test all required imports."""
    print("🧪 Testing package imports...")
    
    try:
        from atlassian import Confluence
        print("✅ atlassian-python-api: OK")
    except ImportError as e:
        print(f"❌ atlassian-python-api: FAILED - {e}")
        return False
        
    try:
        from bs4 import BeautifulSoup
        print("✅ beautifulsoup4: OK")
    except ImportError as e:
        print(f"❌ beautifulsoup4: FAILED - {e}")
        return False
        
    try:
        import json
        print("✅ json: OK")
    except ImportError as e:
        print(f"❌ json: FAILED - {e}")
        return False
        
    try:
        import re
        print("✅ re: OK")
    except ImportError as e:
        print(f"❌ re: FAILED - {e}")
        return False
        
    return True

def test_confluence_client_creation():
    """Test creating Confluence client with dummy credentials."""
    print("\n🔧 Testing Confluence client creation...")
    
    try:
        from atlassian import Confluence
        
        # Create client with dummy credentials (won't connect)
        client = Confluence(
            url="https://example.atlassian.net",
            username="test@example.com",
            password="dummy-token",
            cloud=True
        )
        
        print("✅ Confluence client creation: OK")
        return True
        
    except Exception as e:
        print(f"❌ Confluence client creation: FAILED - {e}")
        return False

def test_html_processing():
    """Test HTML processing functionality."""
    print("\n🧹 Testing HTML processing...")
    
    try:
        from bs4 import BeautifulSoup
        import re
        
        # Sample Confluence HTML
        sample_html = """
        <div>
            <h1>Test Page</h1>
            <p>This is a test paragraph with <strong>bold text</strong>.</p>
            <ac:structured-macro ac:name="info">
                <ac:rich-text-body>
                    <p>This is a Confluence macro that should be removed.</p>
                </ac:rich-text-body>
            </ac:structured-macro>
            <p>Another paragraph after the macro.</p>
            <script>alert('This script should be removed');</script>
        </div>
        """
        
        # Parse and clean
        soup = BeautifulSoup(sample_html, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(["script", "style"]):
            element.decompose()
        for element in soup.find_all(['ac:structured-macro']):
            element.decompose()
            
        # Get clean text
        clean_text = soup.get_text().strip()
        
        expected_content = ["Test Page", "This is a test paragraph", "bold text", "Another paragraph"]
        
        # Verify expected content is present
        for content in expected_content:
            if content not in clean_text:
                print(f"❌ HTML processing: Missing expected content '{content}'")
                return False
                
        # Verify unwanted content is removed
        unwanted_content = ["ac:structured-macro", "alert('This script", "Confluence macro"]
        for content in unwanted_content:
            if content in clean_text:
                print(f"❌ HTML processing: Found unwanted content '{content}'")
                return False
                
        print("✅ HTML processing: OK")
        print(f"   Clean text: {clean_text[:100]}...")
        return True
        
    except Exception as e:
        print(f"❌ HTML processing: FAILED - {e}")
        return False

def test_json_processing():
    """Test JSON serialization/deserialization."""
    print("\n📄 Testing JSON processing...")
    
    try:
        import json
        from datetime import datetime
        
        # Sample data structure
        test_data = {
            'page_id': '12345',
            'title': 'Test Page',
            'content': 'This is test content with special chars: áéíóú',
            'created_date': datetime.now().isoformat(),
            'labels': ['test', 'poc', 'confluence'],
            'metadata': {
                'space_key': 'TEST',
                'author': 'Test User'
            }
        }
        
        # Test serialization
        json_str = json.dumps(test_data, indent=2, ensure_ascii=False)
        
        # Test deserialization
        parsed_data = json.loads(json_str)
        
        # Verify data integrity
        if parsed_data['title'] != test_data['title']:
            print("❌ JSON processing: Data integrity check failed")
            return False
            
        print("✅ JSON processing: OK")
        return True
        
    except Exception as e:
        print(f"❌ JSON processing: FAILED - {e}")
        return False

def test_configuration_loading():
    """Test configuration loading logic."""
    print("\n⚙️ Testing configuration loading...")
    
    try:
        # Test environment variable reading
        test_config = {
            "base_url": os.getenv("CONFLUENCE_BASE_URL", "https://default.atlassian.net"),
            "username": os.getenv("CONFLUENCE_USERNAME", "default@example.com"),
            "api_token": os.getenv("CONFLUENCE_API_TOKEN", "default-token"),
            "spaces": os.getenv("CONFLUENCE_SPACES", "").split(",") if os.getenv("CONFLUENCE_SPACES") else ["TEST"],
            "max_pages": int(os.getenv("CONFLUENCE_MAX_PAGES", "5")),
        }
        
        # Validate config structure
        required_keys = ['base_url', 'username', 'api_token', 'spaces', 'max_pages']
        for key in required_keys:
            if key not in test_config:
                print(f"❌ Configuration: Missing key '{key}'")
                return False
                
        print("✅ Configuration loading: OK")
        print(f"   Base URL: {test_config['base_url']}")
        print(f"   Username: {test_config['username']}")
        print(f"   Spaces: {test_config['spaces']}")
        print(f"   Max pages: {test_config['max_pages']}")
        return True
        
    except Exception as e:
        print(f"❌ Configuration loading: FAILED - {e}")
        return False

def check_file_structure():
    """Check if required files exist."""
    print("\n📁 Checking file structure...")
    
    required_files = [
        'confluence_poc.py',
        'setup_confluence_poc.py', 
        'CONFLUENCE_POC_README.md',
        '.env.confluence.template'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
        else:
            print(f"✅ {file_path}: Found")
            
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
        
    print("✅ All required files present")
    return True

def provide_next_steps():
    """Provide guidance on next steps."""
    print("\n📋 Next Steps:")
    
    # Check if .env exists
    env_file = Path(".env")
    if env_file.exists():
        with open(env_file) as f:
            content = f.read()
            
        has_confluence = any(key in content for key in [
            "CONFLUENCE_BASE_URL", 
            "CONFLUENCE_USERNAME", 
            "CONFLUENCE_API_TOKEN"
        ])
        
        if has_confluence:
            print("✅ Configuration found in .env file")
            print("   Ready to run: python confluence_poc.py")
        else:
            print("⚠️  Add Confluence settings to your .env file")
            print("   Copy from: .env.confluence.template")
    else:
        print("📝 Create .env file with your Confluence credentials:")
        print("   1. Copy .env.confluence.template to .env")
        print("   2. Update with your actual Confluence details")
        print("   3. Get API token: https://id.atlassian.com/manage/api-tokens")
        
    print("\n🔗 Helpful Resources:")
    print("   • API Token: https://id.atlassian.com/manage/api-tokens")
    print("   • Find spaces: Go to your Confluence → Spaces → View all spaces")
    print("   • Test connection: python confluence_poc.py")

def main():
    """Run all tests."""
    print("🧪 Confluence POC Setup Validation")
    print("=" * 40)
    
    tests = [
        ("Package Imports", test_imports),
        ("File Structure", check_file_structure),
        ("Confluence Client", test_confluence_client_creation),
        ("HTML Processing", test_html_processing),
        ("JSON Processing", test_json_processing),
        ("Configuration", test_configuration_loading),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} test failed")
    
    print(f"\n{'='*60}")
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Setup is ready.")
        provide_next_steps()
        return 0
    else:
        print("❌ Some tests failed. Check the errors above.")
        print("   Try running: python setup_confluence_poc.py")
        return 1

if __name__ == "__main__":
    sys.exit(main())