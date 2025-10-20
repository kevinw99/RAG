# Confluence POC - Simple Crawler Test

A proof-of-concept script to test Confluence crawling with user credentials before building the full enterprise integration.

## 🎯 Purpose

This POC validates:
- ✅ Confluence API authentication with user credentials
- ✅ Basic page content extraction and HTML cleanup
- ✅ Integration potential with existing RAG system
- ✅ Corporate network compatibility (proxy/firewall testing)

## 🚀 Quick Start

### 1. Setup Dependencies
```bash
python setup_confluence_poc.py
```

### 2. Configure Credentials
Create `.env.confluence` file (keeps POC config separate from main RAG system):
```bash
# Your Confluence details
CONFLUENCE_BASE_URL=https://your-company.atlassian.net
CONFLUENCE_USERNAME=your.email@company.com
CONFLUENCE_API_TOKEN=your-api-token

# Spaces to test (comma-separated)
CONFLUENCE_SPACES=TECH,DOCS,HELP

# Limit pages for testing
CONFLUENCE_MAX_PAGES=5
```

**Note**: We use `.env.confluence` to avoid conflicts with the main RAG system's configuration.

### 3. Get API Token
1. Go to https://id.atlassian.com/manage/api-tokens
2. Click "Create API token"
3. Give it a name (e.g., "RAG POC Test")
4. Copy the token to your `.env` file

### 4. Run POC
```bash
python confluence_poc.py
```

## 📋 What the POC Tests

### Authentication Test
- Connects to Confluence using username + API token
- Validates credentials work through corporate network
- Tests proxy/firewall compatibility

### Content Extraction Test
- Lists available spaces
- Extracts pages from configured spaces
- Cleans Confluence HTML to readable text
- Preserves metadata (author, dates, labels, etc.)

### Integration Test
- Converts Confluence pages to RAG Document format
- Tests chunking with existing text processing
- Validates metadata compatibility

## 📊 Expected Output

```
🚀 Confluence POC - Testing Basic Crawling
==================================================
🔐 Connecting to Confluence: https://company.atlassian.net
👤 Username: user@company.com
✅ Successfully connected as: John Doe

📚 Listing available spaces...
Found 3 spaces:
  • TECH: Technical Documentation
  • DOCS: General Documentation  
  • HELP: Help Center

🕷️  Crawling space: TECH
📄 Getting pages from space: TECH
Found 5 pages in TECH
  ✅ Processed: API Authentication Guide (1,234 words)
  ✅ Processed: Database Setup (892 words)
  ...

📊 Crawling Summary:
   Total pages: 10
   Total words: 15,432
   Spaces crawled: 2
     • TECH: 5 pages, 8,123 words
     • DOCS: 5 pages, 7,309 words

💾 Results saved to: confluence_poc_results_20250801_143022.json

🔗 Testing integration with existing RAG system...
✅ Successfully imported RAG system components
📂 Loading results from: confluence_poc_results_20250801_143022.json
✅ Created 12 chunks from Confluence page
✅ Basic RAG integration successful!

🎉 POC completed successfully!
```

## 🔧 Troubleshooting

### Authentication Issues
- **"Connection failed"**: Check base URL format (include https://)
- **"Unauthorized"**: Verify API token is correct and not expired
- **"403 Forbidden"**: Check if you have read access to the spaces

### Corporate Network Issues
- **"Connection timeout"**: May need proxy configuration
- **"SSL errors"**: Corporate firewalls may intercept HTTPS

Add proxy settings to `.env`:
```bash
HTTP_PROXY=http://proxy.company.com:8080
HTTPS_PROXY=https://proxy.company.com:8080
```

### No Spaces/Pages Found
- **Empty spaces list**: Check permissions, you may not have access
- **No pages in space**: Space might be empty or pages are restricted

## 📁 Output Files

The POC creates JSON files with crawled data:
- `confluence_poc_results_YYYYMMDD_HHMMSS.json` - Full crawling results
- Includes metadata, content, and configuration used

## 🔗 Next Steps

After successful POC:

1. **Validate Results**: Review crawled content quality
2. **Test Corporate Network**: Run from corporate network/VPN
3. **Scale Testing**: Increase `CONFLUENCE_MAX_PAGES` for larger test
4. **Security Review**: Ensure compliance with corporate policies
5. **Full Integration**: Proceed with enterprise integration PRP

## 📚 Key Files

- `confluence_poc.py` - Main POC script
- `setup_confluence_poc.py` - Dependency installer and setup helper
- `.env.confluence.template` - Configuration template
- `confluence_poc_results_*.json` - Output data

## ⚠️ Important Notes

- **API Token Security**: Never commit API tokens to version control
- **Rate Limiting**: POC respects Confluence API rate limits
- **Permissions**: Only crawls content you have read access to
- **Corporate Compliance**: Check with IT before running on corporate network

This POC provides the foundation for building the full enterprise Confluence integration!