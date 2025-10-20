# Enterprise Data Synthesis Patterns for Multi-Source RAG

## Advanced Algorithms for Handling Overlapping Information Sources

### 1. Reciprocal Rank Fusion (RRF) Implementation

```python
def reciprocal_rank_fusion(ranking_lists: List[List[Tuple]], k: int = 60) -> List[Tuple]:
    """
    Combine multiple ranking lists using RRF algorithm.
    Used to merge results from different document sources (Confluence + existing docs).
    """
    fused_scores = {}
    
    for ranking in ranking_lists:
        for rank, (doc_id, original_score) in enumerate(ranking):
            if doc_id not in fused_scores:
                fused_scores[doc_id] = {
                    'rrf_score': 0,
                    'original_score': original_score,
                    'sources': []
                }
            
            # RRF formula: 1 / (k + rank)
            fused_scores[doc_id]['rrf_score'] += 1 / (k + rank + 1)
            fused_scores[doc_id]['sources'].append(f"source_{len(ranking_lists)}")
    
    # Sort by combined RRF score
    return sorted(fused_scores.items(), key=lambda x: x[1]['rrf_score'], reverse=True)
```

### 2. Maximal Marginal Relevance (MMR) for Diversity

```python
def mmr_selection(query_embedding: np.ndarray, 
                  doc_embeddings: List[np.ndarray],
                  doc_metadata: List[Dict],
                  lambda_param: float = 0.7, 
                  k: int = 5) -> List[int]:
    """
    Select diverse documents using MMR to reduce redundancy from overlapping sources.
    
    Args:
        lambda_param: Balance relevance (1.0) vs diversity (0.0)
        k: Number of documents to select
    """
    from sklearn.metrics.pairwise import cosine_similarity
    
    selected_indices = []
    remaining_indices = list(range(len(doc_embeddings)))
    
    while len(selected_indices) < k and remaining_indices:
        mmr_scores = []
        
        for i in remaining_indices:
            # Relevance to query
            relevance = cosine_similarity([query_embedding], [doc_embeddings[i]])[0][0]
            
            # Maximum similarity to already selected documents
            if selected_indices:
                selected_embeddings = [doc_embeddings[j] for j in selected_indices]
                similarities = cosine_similarity([doc_embeddings[i]], selected_embeddings)[0]
                max_similarity = max(similarities)
            else:
                max_similarity = 0
            
            # MMR score: λ * relevance - (1-λ) * max_similarity
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity
            mmr_scores.append((mmr_score, i))
        
        # Select document with highest MMR score
        best_score, best_idx = max(mmr_scores)
        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)
    
    return selected_indices
```

### 3. Multi-Factor Trust Scoring

```python
from datetime import datetime, timedelta
import math

class TrustScorer:
    """Calculate trust scores for multi-source content."""
    
    def __init__(self):
        self.source_weights = {
            'official_documentation': 1.0,
            'confluence_verified': 0.85,
            'confluence_community': 0.7,
            'wiki_user_generated': 0.6,
            'draft_content': 0.4
        }
        
    def calculate_trust_score(self, content_metadata: Dict) -> float:
        """Calculate comprehensive trust score for content."""
        
        # Base source weight
        source_type = content_metadata.get('source_type', 'wiki_user_generated')
        base_weight = self.source_weights.get(source_type, 0.5)
        
        # Author reputation factor (0-1 scale)
        author_reputation = content_metadata.get('author_reputation', 50)
        author_factor = min(author_reputation / 100, 1.0)
        
        # Recency factor (newer content gets higher scores)
        last_modified = datetime.fromisoformat(content_metadata.get('last_modified', '2020-01-01'))
        days_old = (datetime.now() - last_modified).days
        recency_factor = max(0.1, math.exp(-days_old / 180))  # Exponential decay over 6 months
        
        # Edit stability (balance between maintenance and instability)
        edit_count = content_metadata.get('edit_count', 1)
        stability_factor = min(1.0, math.log(edit_count + 1) / 3)
        
        # View count factor (popular content gets slight boost)
        view_count = content_metadata.get('view_count', 0)
        popularity_factor = min(1.0, 0.5 + math.log(view_count + 1) / 20)
        
        # Combine all factors
        trust_score = (
            base_weight * 0.4 +           # Source type is most important
            author_factor * 0.2 +         # Author reputation
            recency_factor * 0.2 +        # Content freshness
            stability_factor * 0.1 +      # Edit stability
            popularity_factor * 0.1       # Community validation
        )
        
        return min(max(trust_score, 0.0), 1.0)  # Clamp to [0, 1]
```

### 4. Semantic Deduplication with Clustering

```python
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
import hashlib

class SemanticDeduplicator:
    """Remove semantically duplicate content across sources."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        
    def deduplicate_content(self, documents: List[Dict], 
                          similarity_threshold: float = 0.85) -> List[Dict]:
        """
        Deduplicate documents using semantic similarity clustering.
        
        Args:
            documents: List of document dicts with 'content' and 'metadata'
            similarity_threshold: Cosine similarity threshold for duplicates
        """
        if not documents:
            return []
            
        # Extract content for embedding
        contents = [doc['content'] for doc in documents]
        
        # Generate embeddings
        embeddings = self.model.encode(contents)
        
        # Use DBSCAN for clustering (eps = 1 - similarity_threshold for cosine distance)
        clustering = DBSCAN(
            eps=1 - similarity_threshold, 
            metric='cosine', 
            min_samples=1
        )
        cluster_labels = clustering.fit_predict(embeddings)
        
        # Keep the best document from each cluster
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append((i, documents[i]))
        
        deduplicated = []
        for cluster_docs in clusters.values():
            # Choose document with highest trust score from cluster
            best_doc = max(cluster_docs, key=lambda x: self._get_trust_score(x[1]))
            deduplicated.append(best_doc[1])
            
        return deduplicated
    
    def _get_trust_score(self, document: Dict) -> float:
        """Get trust score for document selection."""
        scorer = TrustScorer()
        return scorer.calculate_trust_score(document.get('metadata', {}))
```

### 5. Conflict Resolution with Provenance Tracking

```python
class ConflictResolver:
    """Resolve conflicts between information from different sources."""
    
    def __init__(self):
        self.resolution_strategies = {
            'latest_wins': self._latest_content_wins,
            'authority_based': self._authority_based_resolution,
            'consensus_based': self._consensus_based_resolution,
            'user_feedback': self._feedback_based_resolution
        }
        
    def resolve_conflicts(self, conflicting_chunks: List[Dict], 
                         strategy: str = 'authority_based') -> Dict:
        """
        Resolve conflicts between chunks with similar content but different information.
        """
        resolver = self.resolution_strategies.get(strategy, self._authority_based_resolution)
        return resolver(conflicting_chunks)
    
    def _authority_based_resolution(self, chunks: List[Dict]) -> Dict:
        """Resolve based on source authority and trust scores."""
        scorer = TrustScorer()
        
        # Calculate trust scores for each chunk
        scored_chunks = []
        for chunk in chunks:
            trust_score = scorer.calculate_trust_score(chunk.get('metadata', {}))
            scored_chunks.append((trust_score, chunk))
        
        # Return chunk with highest trust score
        best_chunk = max(scored_chunks, key=lambda x: x[0])
        
        # Add provenance information
        result = best_chunk[1].copy()
        result['conflict_resolution'] = {
            'method': 'authority_based',
            'alternatives_count': len(chunks) - 1,
            'trust_score': best_chunk[0],
            'conflicting_sources': [
                chunk['metadata'].get('source', 'unknown') 
                for chunk in chunks if chunk != best_chunk[1]
            ]
        }
        
        return result
    
    def _latest_content_wins(self, chunks: List[Dict]) -> Dict:
        """Resolve by preferring most recently modified content."""
        def get_modified_date(chunk):
            metadata = chunk.get('metadata', {})
            modified_str = metadata.get('last_modified', '2020-01-01')
            return datetime.fromisoformat(modified_str)
        
        latest_chunk = max(chunks, key=get_modified_date)
        
        latest_chunk['conflict_resolution'] = {
            'method': 'latest_wins',
            'selected_date': latest_chunk['metadata'].get('last_modified'),
            'alternatives_count': len(chunks) - 1
        }
        
        return latest_chunk
    
    def _consensus_based_resolution(self, chunks: List[Dict]) -> Dict:
        """Resolve based on information that appears in majority of sources."""
        # This is a simplified version - real implementation would need NLP
        # to identify common facts across different phrasings
        
        # For now, use the chunk with most similar content to others
        similarities = []
        for i, chunk_a in enumerate(chunks):
            similarity_sum = 0
            for j, chunk_b in enumerate(chunks):
                if i != j:
                    # Calculate semantic similarity (simplified)
                    model = SentenceTransformer('all-MiniLM-L6-v2')
                    emb_a = model.encode([chunk_a['content']])
                    emb_b = model.encode([chunk_b['content']])
                    similarity = cosine_similarity(emb_a, emb_b)[0][0]
                    similarity_sum += similarity
            similarities.append((similarity_sum, chunk_a))
        
        consensus_chunk = max(similarities, key=lambda x: x[0])
        result = consensus_chunk[1].copy()
        result['conflict_resolution'] = {
            'method': 'consensus_based',
            'consensus_score': consensus_chunk[0],
            'alternatives_count': len(chunks) - 1
        }
        
        return result
```

### 6. User Feedback Integration System

```python
class FeedbackSystem:
    """Integrate user feedback to improve source prioritization."""
    
    def __init__(self, feedback_storage_path: str = "feedback.json"):
        self.feedback_storage = feedback_storage_path
        self.feedback_data = self._load_feedback()
        
    def collect_feedback(self, query: str, response: str, sources: List[str], 
                        user_rating: int, user_comments: str = None):
        """Collect structured user feedback on RAG responses."""
        feedback_entry = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'response_hash': hashlib.md5(response.encode()).hexdigest(),
            'sources': sources,
            'rating': user_rating,  # 1-5 scale
            'comments': user_comments
        }
        
        self.feedback_data.append(feedback_entry)
        self._save_feedback()
        
    def get_source_reputation(self, source_id: str) -> float:
        """Calculate source reputation based on user feedback."""
        source_feedback = [
            f for f in self.feedback_data 
            if source_id in f.get('sources', [])
        ]
        
        if not source_feedback:
            return 0.5  # Neutral reputation for new sources
            
        total_rating = sum(f['rating'] for f in source_feedback)
        avg_rating = total_rating / len(source_feedback)
        
        # Normalize to 0-1 scale (5-point scale becomes 0-1)
        return (avg_rating - 1) / 4
        
    def _load_feedback(self) -> List[Dict]:
        """Load feedback data from storage."""
        try:
            with open(self.feedback_storage, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return []
            
    def _save_feedback(self):
        """Save feedback data to storage."""
        with open(self.feedback_storage, 'w') as f:
            json.dump(self.feedback_data, f, indent=2)
```

## Integration Pattern for RAG System

```python
class MultiSourceRAGProcessor:
    """Integrate all synthesis patterns into RAG processing."""
    
    def __init__(self):
        self.deduplicator = SemanticDeduplicator()
        self.trust_scorer = TrustScorer()
        self.conflict_resolver = ConflictResolver()
        self.feedback_system = FeedbackSystem()
        
    async def process_query(self, query: str, user_id: str) -> Dict:
        """Process query with multi-source synthesis."""
        
        # 1. Retrieve from all sources
        confluence_results = await self._retrieve_confluence(query, user_id)
        document_results = await self._retrieve_documents(query)
        
        # 2. Combine and deduplicate
        all_results = confluence_results + document_results
        deduplicated = self.deduplicator.deduplicate_content(all_results)
        
        # 3. Detect conflicts and resolve
        conflicts = self._detect_conflicts(deduplicated)
        resolved = []
        for conflict_group in conflicts:
            if len(conflict_group) > 1:
                resolved_chunk = self.conflict_resolver.resolve_conflicts(conflict_group)
                resolved.append(resolved_chunk)
            else:
                resolved.extend(conflict_group)
        
        # 4. Apply trust scoring and ranking
        for chunk in resolved:
            chunk['trust_score'] = self.trust_scorer.calculate_trust_score(
                chunk.get('metadata', {})
            )
        
        # 5. Use MMR for final selection
        final_chunks = self._mmr_selection(query, resolved)
        
        return {
            'chunks': final_chunks,
            'synthesis_metadata': {
                'total_sources': len(all_results),
                'deduplicated_count': len(deduplicated),
                'conflicts_resolved': len([g for g in conflicts if len(g) > 1]),
                'final_count': len(final_chunks)
            }
        }
```

This comprehensive pattern library provides robust handling of multi-source information with advanced deduplication, conflict resolution, and quality scoring mechanisms.