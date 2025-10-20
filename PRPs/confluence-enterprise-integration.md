name: "Confluence Enterprise Integration PRP - Multi-Source RAG with Zscaler/Okta Support"
description: |

## Purpose

Comprehensive PRP for integrating Confluence as a secure enterprise data source in RAG system with advanced data synthesis, conflict resolution, and corporate security integration (Zscaler/Okta).

## Core Principles

1. **Enterprise Security First**: Maintain Confluence permissions, GDPR compliance, audit trails
2. **Data Synthesis Intelligence**: Handle overlapping/conflicting information across sources  
3. **Corporate Network Integration**: Seamless operation behind Zscaler/Okta security layers
4. **Validation-Driven Development**: Executable tests ensure enterprise-grade reliability

---

## Goal

Build enterprise-grade Confluence integration for RAG system that:
- Securely crawls Confluence pages behind corporate Zscaler/Okta authentication
- Intelligently synthesizes Confluence data with existing document library 
- Resolves conflicts between overlapping information sources
- Maintains Confluence permission model in RAG responses
- Provides GDPR-compliant data handling with audit trails
- Delivers production-ready reliability with comprehensive monitoring

## Why

- **Business Value**: Unlock 50%+ more enterprise knowledge trapped in Confluence wikis
- **User Impact**: Single RAG interface for both formal docs and collaborative wiki content  
- **Integration Need**: Existing 44K+ chunk document library needs Confluence augmentation
- **Security Requirement**: Enterprise compliance demands proper authentication and access control
- **Data Quality**: Advanced synthesis prevents conflicting information in responses

## What

Enterprise Confluence integration with intelligent data synthesis capabilities:

### User-Visible Behavior
- RAG queries return synthesized results from both documents and Confluence
- Confluence content appears with proper source attribution and trust scores
- Users only see Confluence content they have permission to access
- Conflicting information is intelligently resolved with provenance tracking
- Real-time synchronization keeps Confluence data current

### Technical Requirements  
- Zscaler proxy-aware authentication with Okta SSO support
- Permission-preserving RAG responses respecting Confluence access control
- Advanced deduplication and conflict resolution algorithms
- GDPR-compliant data processing with audit trails
- Rate-limited API integration following Atlassian 2025 guidelines
- Multi-source trust scoring and ranking system

### Success Criteria

- [ ] Successfully authenticate through corporate Zscaler/Okta infrastructure
- [ ] Index minimum 1000 Confluence pages with proper metadata extraction
- [ ] Achieve <85% semantic overlap detection accuracy between sources
- [ ] Maintain <2 second query response time with Confluence integration
- [ ] Pass enterprise security audit with comprehensive access logging
- [ ] Demonstrate 90%+ user satisfaction with conflict resolution quality
- [ ] Achieve 99.5% uptime with proper error handling and monitoring

## All Needed Context

### Documentation & References

```yaml
# MUST READ - Include these in your context window
- url: https://developer.atlassian.com/cloud/confluence/rest/v2/intro/
  why: Official Confluence API v2 documentation for authentication and content access

- url: https://github.com/atlassian-api/atlassian-python-api
  why: Primary Python library for Confluence integration - check auth patterns

- url: https://developer.atlassian.com/cloud/confluence/rate-limiting/
  why: CRITICAL - New rate limiting effective Nov 22, 2025 - 429 error handling required

- file: /Users/kweng/AI/RAG/rag_system/core/document_processor.py
  why: Document processing patterns to extend for Confluence content

- file: /Users/kweng/AI/RAG/rag_system/storage/vector_store.py  
  why: ChromaDB integration patterns for metadata handling and chunking

- file: /Users/kweng/AI/RAG/rag_system/config/settings.py
  why: Configuration patterns using Pydantic BaseSettings with env var integration

- file: /Users/kweng/AI/RAG/rag_system/core/data_models.py
  why: Data model patterns using dataclasses for Confluence-specific metadata

- docfile: /Users/kweng/AI/RAG/PRPs/ai_docs/confluence_enterprise_integration_guide.md
  why: Enterprise-specific implementation patterns, auth gotchas, security requirements

- docfile: /Users/kweng/AI/RAG/PRPs/ai_docs/enterprise_data_synthesis_patterns.md  
  why: Advanced algorithms for deduplication, conflict resolution, trust scoring

- url: https://support.atlassian.com/organization-administration/docs/ip-addresses-and-domains-for-atlassian-cloud-products/
  why: Network configuration for corporate firewalls and proxy setup

- url: https://www.atlassian.com/trust/compliance/compliance-faq
  why: Compliance requirements for enterprise deployment (GDPR, SOC2, ISO27001)
```

### Current Codebase Structure

```bash
/Users/kweng/AI/RAG/
├── rag_system/
│   ├── core/
│   │   ├── document_processor.py      # Extend for Confluence
│   │   ├── data_models.py            # Add Confluence models
│   │   ├── pipeline.py               # Integration point
│   │   └── retriever.py              # Multi-source retrieval
│   ├── storage/
│   │   └── vector_store.py           # ChromaDB patterns
│   ├── config/
│   │   └── settings.py               # Add Confluence config
│   ├── api/
│   │   └── server.py                 # Add Confluence endpoints
│   └── monitoring/
│       └── health.py                 # Add Confluence health checks
├── streamlit_chat.py                 # UI integration
├── start_server.py                   # Server startup
└── tests/                            # Test patterns
```

### Desired Codebase Structure with New Files

```bash
/Users/kweng/AI/RAG/
├── rag_system/
│   ├── sources/                      # NEW: External data sources
│   │   ├── __init__.py
│   │   ├── confluence/
│   │   │   ├── __init__.py
│   │   │   ├── client.py             # Confluence API client with auth
│   │   │   ├── processor.py          # Content processing & cleanup
│   │   │   ├── permissions.py        # Permission validation
│   │   │   └── models.py             # Confluence-specific data models
│   │   └── synthesis/
│   │       ├── __init__.py
│   │       ├── deduplicator.py       # Semantic deduplication
│   │       ├── conflict_resolver.py  # Multi-source conflict resolution
│   │       ├── trust_scorer.py       # Source authority & freshness
│   │       └── feedback_system.py    # User feedback integration
│   └── core/
│       └── multi_source_retriever.py # NEW: Multi-source query handling
└── tests/
    ├── test_confluence_integration.py # NEW: Confluence tests
    ├── test_data_synthesis.py        # NEW: Synthesis algorithm tests
    └── test_enterprise_security.py   # NEW: Security & compliance tests
```

### Known Gotchas & Library Quirks

```python
# CRITICAL: Atlassian API Authentication Changes (2025)
# API tokens created before Dec 15, 2024 expire Mar-May 2026
# Must use scoped tokens with minimal permissions

# CRITICAL: Rate Limiting (Effective Nov 22, 2025) 
# HTTP 429 responses with Retry-After headers
# Maximum 10 requests/second, implement exponential backoff

# GOTCHA: Corporate Proxy Configuration
# Zscaler proxies intercept HTTPS, may break certificate validation
# Use proxies parameter in atlassian-python-api, not environment variables

# GOTCHA: Confluence HTML Cleanup
# Confluence macros (ac:structured-macro) don't render in API responses
# Must strip or expand macros for meaningful content extraction

# GOTCHA: Permission Model Complexity  
# Confluence has space, page, and inherited permissions
# Cannot replicate full model - must query API for each user request

# CRITICAL: ChromaDB Metadata Limitations
# Only accepts str, int, float, bool, None values
# Must sanitize datetime and complex objects in metadata

# GOTCHA: Memory Management with Large Confluence Instances
# 44K+ existing chunks + Confluence pages can exceed memory limits  
# Use batch processing with garbage collection (existing pattern)
```

## Implementation Blueprint

### Data Models and Structure

Create Confluence-specific data models extending existing patterns:

```python
# rag_system/sources/confluence/models.py
@dataclass
class ConfluencePage:
    page_id: str
    title: str
    content: str
    space_key: str
    created_date: datetime
    modified_date: datetime
    author: str
    labels: List[str]
    permissions: List[str]
    page_url: str
    parent_page_id: Optional[str] = None
    
@dataclass  
class ConfluenceSpace:
    space_key: str
    name: str
    description: str
    homepage_id: str
    permissions: List[str]

# rag_system/sources/synthesis/models.py
@dataclass
class ConflictResolution:
    method: str
    winning_source: str
    alternatives_count: int
    trust_score: float
    conflicting_sources: List[str]
    user_feedback_influence: float = 0.0
```

### List of Tasks (Implementation Order)

```yaml
Task 1: 
MODIFY rag_system/config/settings.py:
  - FIND pattern: "# Environment Configuration" 
  - INJECT Confluence configuration section after line
  - ADD confluence_base_url, api_token, spaces, proxy settings
  - PRESERVE existing Pydantic BaseSettings pattern

CREATE rag_system/sources/__init__.py:
  - MIRROR pattern from: rag_system/core/__init__.py
  - DEFINE sources module structure

Task 2:
CREATE rag_system/sources/confluence/client.py:
  - IMPLEMENT ConfluenceClient class with enterprise auth
  - PATTERN: Follow existing async patterns from retriever.py
  - CRITICAL: Handle Zscaler proxy, rate limiting, error handling
  - ADD permission validation methods

CREATE rag_system/sources/confluence/models.py:
  - MIRROR pattern from: rag_system/core/data_models.py
  - DEFINE ConfluencePage, ConfluenceSpace dataclasses
  - PRESERVE validation patterns with field validators

Task 3:
CREATE rag_system/sources/confluence/processor.py:
  - EXTEND DocumentProcessor pattern from core/document_processor.py
  - IMPLEMENT Confluence HTML cleanup and metadata extraction
  - PRESERVE async processing with batch management
  - ADD Confluence-specific content sanitization

Task 4:
CREATE rag_system/sources/synthesis/deduplicator.py:
  - IMPLEMENT SemanticDeduplicator using sentence-transformers
  - PATTERN: Follow embedding patterns from storage/vector_store.py
  - USE existing embedding model configuration
  - ADD semantic clustering with DBSCAN

Task 5:
CREATE rag_system/sources/synthesis/trust_scorer.py:
  - IMPLEMENT multi-factor trust scoring algorithm
  - PATTERN: Follow confidence scoring patterns from core/generator.py
  - PRESERVE scoring methodology and normalization
  - ADD source authority, recency, feedback integration

Task 6:
CREATE rag_system/sources/synthesis/conflict_resolver.py:
  - IMPLEMENT conflict detection and resolution algorithms
  - USE trust scoring and user feedback systems
  - ADD provenance tracking for transparency
  - PRESERVE existing error handling patterns

Task 7:
MODIFY rag_system/core/document_processor.py:
  - FIND pattern: "class DocumentType(Enum)"
  - INJECT CONFLUENCE = "confluence" after existing types
  - FIND pattern: "self.processors = {"
  - ADD DocumentType.CONFLUENCE: self._process_confluence
  - IMPLEMENT _process_confluence method

Task 8:
CREATE rag_system/core/multi_source_retriever.py:
  - EXTEND existing retriever patterns from core/retriever.py
  - IMPLEMENT multi-source query coordination
  - INTEGRATE deduplication and conflict resolution
  - PRESERVE hybrid search and reranking functionality

Task 9:
MODIFY rag_system/api/server.py:
  - FIND pattern: "@app.post("/ingest")"
  - INJECT Confluence ingestion endpoint after existing
  - ADD ConfluenceIngestionRequest model
  - PRESERVE background task patterns

Task 10:
MODIFY rag_system/monitoring/health.py:
  - FIND pattern: "async def _check_vector_store"
  - INJECT _check_confluence_api method after
  - ADD Confluence connectivity and auth validation
  - PRESERVE health check structure and error handling

Task 11:
CREATE comprehensive test suite:
  - tests/test_confluence_integration.py
  - tests/test_data_synthesis.py  
  - tests/test_enterprise_security.py
  - MIRROR existing test patterns from test_*.py files
```

### Per Task Pseudocode

```python
# Task 1: Configuration Extension
# CRITICAL: Corporate proxy configuration pattern
class Settings(BaseSettings):
    # Confluence Configuration  
    confluence_base_url: Optional[str] = Field(default=None)
    confluence_username: Optional[str] = Field(default=None) 
    confluence_api_token: Optional[str] = Field(default=None)
    confluence_spaces: List[str] = Field(default=[])
    confluence_proxy_host: Optional[str] = Field(default=None)
    confluence_proxy_port: Optional[int] = Field(default=None)
    confluence_batch_size: int = Field(default=50)
    
# Task 2: Enterprise Authentication Client
class ConfluenceClient:
    def __init__(self, base_url: str, username: str, api_token: str, 
                 proxy_config: Optional[Dict] = None):
        # PATTERN: Use atlassian-python-api with enterprise proxy
        self.confluence = Confluence(
            url=base_url,
            username=username, 
            password=api_token,
            proxies=proxy_config  # CRITICAL: Not environment variables
        )
        self.rate_limiter = RateLimiter(requests_per_second=1)
        
    async def get_pages_batch(self, space_key: str, start: int = 0, 
                             limit: int = 50) -> List[ConfluencePage]:
        # CRITICAL: Rate limiting with jitter
        await self.rate_limiter.acquire()
        
        # PATTERN: Error handling with retries  
        @retry(attempts=3, backoff=exponential)
        async def _fetch():
            return self.confluence.get_all_pages_from_space(
                space_key, start=start, limit=limit, expand='body.storage'
            )

# Task 4: Semantic Deduplication
class SemanticDeduplicator:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        # PATTERN: Use existing embedding model from settings
        self.model = SentenceTransformer(model_name)
        
    def deduplicate_content(self, documents: List[Dict], 
                          threshold: float = 0.85) -> List[Dict]:
        # PATTERN: Batch processing with memory management
        embeddings = self.model.encode([doc['content'] for doc in documents])
        
        # CRITICAL: Use cosine distance = 1 - similarity for DBSCAN
        clustering = DBSCAN(eps=1-threshold, metric='cosine', min_samples=1)
        clusters = clustering.fit_predict(embeddings)
        
        # Keep highest trust score from each cluster
        return self._select_best_from_clusters(documents, clusters)

# Task 8: Multi-Source Query Processing  
class MultiSourceRetriever:
    async def retrieve(self, query: str, user_id: str, k: int = 5) -> RetrievalResult:
        # PATTERN: Parallel retrieval from multiple sources
        doc_results, conf_results = await asyncio.gather(
            self.document_retriever.retrieve(query, k),
            self.confluence_retriever.retrieve(query, user_id, k)  # User-aware
        )
        
        # CRITICAL: Permission filtering for Confluence results
        permitted_conf = await self._filter_by_permissions(conf_results, user_id)
        
        # Synthesis pipeline
        combined = doc_results.chunks + permitted_conf.chunks
        deduplicated = self.deduplicator.deduplicate_content(combined)
        resolved = self.conflict_resolver.resolve_conflicts(deduplicated)
        
        # PATTERN: Use existing MMR for diversity
        final_chunks = self._mmr_selection(query, resolved, k)
        
        return RetrievalResult(chunks=final_chunks, scores=..., metadata=...)
```

### Integration Points

```yaml
DATABASE:
  - extend: ChromaDB metadata schema
  - add: source_type, confluence_space, page_id, permissions_hash
  - index: "CREATE INDEX ON confluence_pages(space_key, page_id)"

CONFIG:
  - add to: rag_system/config/settings.py
  - pattern: "confluence_* = Field(default=..., description='...')"
  - validation: "@field_validator for URL and proxy settings"

ROUTES:
  - add to: rag_system/api/server.py  
  - pattern: "router.post('/ingest/confluence', response_model=IngestionResponse)"
  - background: "BackgroundTasks for long-running confluence ingestion"

MONITORING:
  - add to: rag_system/monitoring/health.py
  - pattern: "async def _check_confluence_api() -> HealthCheck"
  - metrics: "track_confluence_processing(pages, api_calls, rate_limit)"
```

## Validation Loop

### Level 1: Syntax & Style

```bash
# Run these FIRST - fix any errors before proceeding
ruff check rag_system/sources/ --fix     # Auto-fix formatting
mypy rag_system/sources/                 # Type checking
ruff check rag_system/core/ --fix        # Check modified files

# Expected: No errors. If errors, READ the error message and fix code.
```

### Level 2: Unit Tests

```python
# CREATE tests/test_confluence_integration.py
@pytest.mark.asyncio
async def test_confluence_authentication():
    """Test corporate authentication with Zscaler proxy"""
    client = ConfluenceClient(
        base_url=settings.confluence_base_url,
        username=settings.confluence_username,
        api_token=settings.confluence_api_token,
        proxy_config=get_proxy_config()
    )
    
    # Test authentication without making external calls
    assert client.confluence is not None
    assert client.rate_limiter is not None

@pytest.mark.asyncio  
async def test_permission_validation():
    """Test user permission checking"""
    permissions = ConfluencePermissions()
    
    # Mock Confluence API response
    with mock.patch.object(permissions, 'confluence_client') as mock_client:
        mock_client.get_page_permissions.return_value = {'read': ['user123']}
        
        result = await permissions.user_can_access('user123', 'page456')
        assert result is True
        
        result = await permissions.user_can_access('user999', 'page456') 
        assert result is False

def test_semantic_deduplication():
    """Test content deduplication across sources"""
    deduplicator = SemanticDeduplicator()
    
    documents = [
        {'content': 'OAuth 2.0 is an authorization framework', 'source': 'doc'},
        {'content': 'OAuth2 is a framework for authorization', 'source': 'confluence'},
        {'content': 'JWT tokens are used for authentication', 'source': 'confluence'}
    ]
    
    result = deduplicator.deduplicate_content(documents, threshold=0.8)
    assert len(result) == 2  # First two should be deduplicated
    
def test_conflict_resolution():
    """Test authority-based conflict resolution"""
    resolver = ConflictResolver()
    
    conflicting_chunks = [
        {
            'content': 'API rate limit is 100 requests/hour',
            'metadata': {
                'source_type': 'confluence_community',
                'last_modified': '2023-01-01'
            }
        },
        {
            'content': 'API rate limit is 1000 requests/hour', 
            'metadata': {
                'source_type': 'official_documentation',
                'last_modified': '2024-01-01'
            }
        }
    ]
    
    result = resolver.resolve_conflicts(conflicting_chunks, strategy='authority_based')
    assert 'API rate limit is 1000' in result['content']
    assert result['conflict_resolution']['method'] == 'authority_based'
```

```bash
# Run and iterate until passing:
uv run pytest tests/test_confluence_integration.py -v
uv run pytest tests/test_data_synthesis.py -v
# If failing: Read error, understand root cause, fix code, re-run
```

### Level 3: Integration Tests

```bash
# Test Confluence API connectivity (requires corporate network)
python -c "
from rag_system.sources.confluence.client import ConfluenceClient
from rag_system.config.settings import settings

client = ConfluenceClient(
    base_url=settings.confluence_base_url,
    username=settings.confluence_username,
    api_token=settings.confluence_api_token
)

# Test authentication 
spaces = client.get_spaces()
print(f'Successfully connected: {len(spaces)} spaces found')
"

# Test full ingestion pipeline
curl -X POST http://localhost:8000/ingest/confluence \
  -H "Content-Type: application/json" \
  -d '{
    "confluence_base_url": "https://company.atlassian.net",
    "spaces": ["TECH", "DOCS"],
    "max_pages": 100
  }'

# Expected: {"task_id": "confluence_ingest_123", "message": "Started"}

# Test multi-source query with permission checking
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -H "X-User-ID: user123" \
  -d '{
    "query": "How to configure OAuth authentication?", 
    "k": 5,
    "include_confluence": true
  }'

# Expected: Mixed results from documents and Confluence with source attribution
```

### Level 4: Enterprise Security & Compliance Validation

```bash
# Security scanning
safety check                              # Dependency vulnerability scan
bandit -r rag_system/sources/             # Security linting

# GDPR compliance validation
python tests/test_gdpr_compliance.py      # Test data retention, deletion

# Performance testing with enterprise load
python tests/test_performance.py --pages=1000 --concurrent=10

# Audit trail validation  
python tests/test_audit_logging.py        # Verify all access is logged

# Network security testing
python tests/test_proxy_auth.py           # Test Zscaler/Okta integration

# Rate limiting compliance
python tests/test_rate_limiting.py        # Verify API rate limit adherence
```

## Final Validation Checklist

- [ ] All tests pass: `uv run pytest tests/ -v --cov=rag_system`
- [ ] No linting errors: `uv run ruff check rag_system/`  
- [ ] No type errors: `uv run mypy rag_system/`
- [ ] Confluence authentication successful through corporate proxy
- [ ] Multi-source queries return synthesized results in <2 seconds
- [ ] Permission validation prevents unauthorized access
- [ ] Conflict resolution produces coherent responses
- [ ] GDPR compliance verified with audit trail
- [ ] Rate limiting prevents API violations
- [ ] Security scan passes (no high/critical vulnerabilities)
- [ ] Memory usage stays within limits during large ingestion
- [ ] Error handling graceful for network failures
- [ ] Documentation updated with Confluence integration guide

---

## Anti-Patterns to Avoid

- ❌ Don't store Confluence content without permission validation
- ❌ Don't ignore rate limiting - will cause API token suspension
- ❌ Don't use environment proxy variables - breaks corporate configs  
- ❌ Don't skip HTML cleanup - macros break content quality
- ❌ Don't cache permissions long-term - permissions change frequently
- ❌ Don't ignore GDPR requirements - compliance violations costly
- ❌ Don't hardcode corporate URLs - use configuration management
- ❌ Don't skip error handling for 429 responses - implement backoff

---

## PRP Quality Score: 9/10

**Confidence Level for One-Pass Implementation Success: 9/10**

**Strengths:**
- Comprehensive research with real-world enterprise patterns
- Detailed authentication and security considerations  
- Advanced data synthesis algorithms with conflict resolution
- Extensive validation gates including security and compliance
- Rich context from existing codebase analysis
- Step-by-step implementation plan with pseudocode

**Minor Gaps:**
- Could benefit from more specific corporate network troubleshooting
- Performance benchmarks could be more detailed for enterprise scale

This PRP provides sufficient context and validation loops for successful one-pass implementation of enterprise-grade Confluence integration with intelligent data synthesis capabilities.