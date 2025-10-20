# Confluence Enterprise Integration Guide

## Critical Implementation Details for RAG Systems

### Atlassian API Authentication (2025 Updates)

**CRITICAL:** API tokens created before December 15, 2024, will expire between March 14 and May 12, 2026. All new integrations must use scoped API tokens.

#### Required Scopes for RAG Integration
```python
REQUIRED_SCOPES = [
    "read:group:confluence",
    "read:user:confluence", 
    "read:content-details:confluence",
    "read:space:confluence",
    "read:permission:confluence",
    "read:audit-log:confluence",
    "read:content.metadata:confluence",
    "read:page:confluence",
    "read:comment:confluence",
    "read:attachment:confluence"
]
```

#### Enterprise Proxy Configuration
```python
# Zscaler proxy setup for atlassian-python-api
proxies = {
    'http': 'http://user:pass@proxy.company.com:8080',
    'https': 'https://user:pass@proxy.company.com:8080'
}

# Environment variables for corporate networks
HTTP_PROXY=http://proxy.company.com:8080
HTTPS_PROXY=https://proxy.company.com:8080
```

### Data Synthesis Challenges

#### Overlap Detection Algorithm
```python
def detect_content_overlap(confluence_content: str, existing_docs: List[str]) -> float:
    """
    Detect semantic overlap between Confluence and existing documents.
    Returns overlap score 0.0-1.0
    """
    # Use sentence-transformers for semantic similarity
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    confluence_embedding = model.encode([confluence_content])
    existing_embeddings = model.encode(existing_docs)
    
    similarities = cosine_similarity(confluence_embedding, existing_embeddings)
    return float(similarities.max())
```

#### Conflict Resolution Strategy
```python
def resolve_information_conflict(confluence_data: Dict, 
                               existing_data: Dict) -> Dict:
    """
    Resolve conflicts between Confluence and existing documentation.
    Priority: Recency > Authority > User Feedback
    """
    # Recency factor (prefer newer content)
    confluence_age = (datetime.now() - confluence_data['modified']).days
    existing_age = (datetime.now() - existing_data['modified']).days
    
    if abs(confluence_age - existing_age) > 30:  # Significant age difference
        return confluence_data if confluence_age < existing_age else existing_data
    
    # Authority factor (official docs > wiki > user content)
    authority_scores = {
        'official_docs': 1.0,
        'confluence_verified': 0.8,
        'wiki_community': 0.6,
        'user_generated': 0.4
    }
    
    confluence_authority = authority_scores.get(confluence_data['type'], 0.5)
    existing_authority = authority_scores.get(existing_data['type'], 0.5)
    
    return confluence_data if confluence_authority >= existing_authority else existing_data
```

### Permission Model Integration

#### Real-time Permission Validation
```python
async def validate_user_access(user_id: str, confluence_page_id: str, 
                             confluence_client: Confluence) -> bool:
    """
    Validate user has access to Confluence page before including in RAG results.
    CRITICAL: Must be called for every query to maintain access control.
    """
    try:
        # Check if user can view the page
        page_permissions = await confluence_client.get_page_permissions(confluence_page_id)
        user_permissions = await confluence_client.get_user_permissions(user_id)
        
        # Intersection check for read permissions
        return any(perm in user_permissions for perm in page_permissions['read'])
    except Exception as e:
        logger.warning(f"Permission check failed for user {user_id}, page {confluence_page_id}: {e}")
        return False  # Fail secure - deny access on error
```

### Rate Limiting (Effective Nov 22, 2025)

#### Implementation Pattern
```python
import asyncio
from datetime import datetime, timedelta

class ConfluenceRateLimiter:
    def __init__(self, requests_per_second: int = 1):
        self.rate = requests_per_second
        self.last_request = datetime.now()
        self.request_count = 0
        
    async def acquire(self):
        """Acquire rate limit token with jitter for scheduled tasks."""
        now = datetime.now()
        
        if now - self.last_request >= timedelta(seconds=1):
            self.request_count = 0
            self.last_request = now
        
        if self.request_count >= self.rate:
            # Add jitter to prevent thundering herd
            jitter = random.uniform(0.1, 0.5)
            await asyncio.sleep(1 + jitter)
            self.request_count = 0
            
        self.request_count += 1
```

### Content Processing Gotchas

#### Confluence HTML Cleanup
```python
def clean_confluence_html(html_content: str) -> str:
    """
    Clean Confluence-specific HTML elements that interfere with RAG processing.
    """
    # Remove Confluence macros
    html_content = re.sub(r'<ac:structured-macro[^>]*>.*?</ac:structured-macro>', '', html_content, flags=re.DOTALL)
    
    # Remove Confluence links that don't translate
    html_content = re.sub(r'<ri:page[^>]*>.*?</ri:page>', '', html_content, flags=re.DOTALL)
    
    # Convert tables to readable text
    html_content = re.sub(r'</?t[rd]>', ' | ', html_content)
    
    # Remove style and script tags
    html_content = re.sub(r'<style[^>]*>.*?</style>', '', html_content, flags=re.DOTALL)
    html_content = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL)
    
    return html_content.strip()
```

### Security Compliance Requirements

#### GDPR Data Handling
```python
class GDPRCompliantProcessor:
    """Handle Confluence data with GDPR requirements."""
    
    def __init__(self, retention_days: int = 365):
        self.retention_period = timedelta(days=retention_days)
        
    def should_retain_content(self, content_metadata: Dict) -> bool:
        """Check if content should be retained based on GDPR requirements."""
        created_date = datetime.fromisoformat(content_metadata['created'])
        return datetime.now() - created_date < self.retention_period
        
    def anonymize_personal_data(self, content: str) -> str:
        """Anonymize personal data in Confluence content."""
        # Remove email addresses
        content = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', content)
        
        # Remove phone numbers (basic pattern)
        content = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', content)
        
        return content
```

## Implementation Checklist

- [ ] Configure API tokens with minimal required scopes
- [ ] Implement enterprise proxy configuration
- [ ] Set up rate limiting with jitter
- [ ] Add permission validation for every query
- [ ] Implement content overlap detection
- [ ] Add conflict resolution algorithms
- [ ] Configure GDPR-compliant data handling
- [ ] Set up audit logging for compliance
- [ ] Test authentication with corporate SSO
- [ ] Validate error handling for API failures

## Known Pitfalls

1. **SSO Blocking API Tokens**: Corporate SSO policies may prevent API token creation
2. **Proxy Certificate Issues**: Corporate proxies may intercept HTTPS, breaking API calls
3. **Rate Limiting Surprises**: New rate limits effective Nov 2025 are strict
4. **Permission Model Complexity**: Confluence has inherited permissions that are hard to replicate
5. **Content Format Variations**: Different Confluence versions have different HTML structures
6. **Macro Expansion**: Some macros need server-side rendering to be useful