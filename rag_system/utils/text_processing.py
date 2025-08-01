"""Text processing utilities for RAG system.

Optimized text cleaning, preprocessing, and PII detection for 612MB dataset.
"""

import re
import unicodedata
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class TextProcessor:
    """Text processing utilities for document preparation."""
    
    def __init__(self):
        """Initialize text processor with patterns."""
        # Common patterns for cleaning
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_pattern = re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b')
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.extra_whitespace_pattern = re.compile(r'\s+')
        self.special_chars_pattern = re.compile(r'[^\w\s\-.,!?;:()\[\]{}"\']')
        
        # PII patterns for detection
        self.ssn_pattern = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
        self.credit_card_pattern = re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b')
    
    def clean_text(self, text: str, remove_pii: bool = False) -> str:
        """Clean and normalize text content.
        
        Args:
            text: Raw text content
            remove_pii: Whether to remove PII data
            
        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text)
        
        # Remove or replace URLs (keep domain for context)
        text = self.url_pattern.sub('[URL]', text)
        
        # Handle PII if requested
        if remove_pii:
            text = self._remove_pii(text)
        
        # Clean up whitespace
        text = self.extra_whitespace_pattern.sub(' ', text)
        
        # Remove excessive special characters but keep some punctuation
        # text = self.special_chars_pattern.sub(' ', text)
        
        # Strip and ensure single spaces
        text = text.strip()
        
        return text
    
    def _remove_pii(self, text: str) -> str:
        """Remove personally identifiable information."""
        # Replace email addresses
        text = self.email_pattern.sub('[EMAIL]', text)
        
        # Replace phone numbers  
        text = self.phone_pattern.sub('[PHONE]', text)
        
        # Replace SSNs
        text = self.ssn_pattern.sub('[SSN]', text)
        
        # Replace credit card numbers
        text = self.credit_card_pattern.sub('[CREDIT_CARD]', text)
        
        return text
    
    def detect_pii(self, text: str) -> Dict[str, List[str]]:
        """Detect PII in text without removing it.
        
        Returns:
            Dictionary with PII types and found instances
        """
        pii_found = {
            'emails': self.email_pattern.findall(text),
            'phones': self.phone_pattern.findall(text),
            'ssns': self.ssn_pattern.findall(text),
            'credit_cards': self.credit_card_pattern.findall(text)
        }
        
        # Remove empty lists
        return {k: v for k, v in pii_found.items() if v}
    
    def extract_metadata_from_text(self, text: str) -> Dict[str, any]:
        """Extract metadata from text content.
        
        Returns:
            Metadata dictionary with text statistics
        """
        if not text:
            return {}
        
        words = text.split()
        sentences = text.split('.')
        
        return {
            'char_count': len(text),
            'word_count': len(words),
            'sentence_count': len([s for s in sentences if s.strip()]),
            'avg_word_length': sum(len(word) for word in words) / len(words) if words else 0,
            'has_pii': bool(self.detect_pii(text)),
            'language_hint': self._detect_language_hint(text)
        }
    
    def _detect_language_hint(self, text: str) -> str:
        """Simple language detection hint based on common patterns."""
        # Very basic detection - in production, use langdetect library
        if not text:
            return 'unknown'
        
        # Count English common words
        english_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = set(text.lower().split())
        english_score = len(words.intersection(english_words))
        
        return 'en' if english_score > 2 else 'unknown'
    
    def prepare_for_embedding(self, text: str) -> str:
        """Prepare text specifically for embedding generation.
        
        Args:
            text: Raw text
            
        Returns:
            Text optimized for embeddings
        """
        # Clean text but preserve semantic meaning
        text = self.clean_text(text, remove_pii=False)
        
        # Ensure reasonable length for sentence-transformers (optimal: 256-512 tokens)
        if len(text) > 2000:  # Rough token estimation
            text = text[:2000] + "..."
        
        return text
    
    def create_contextual_chunk(self, chunk_text: str, doc_context: str, 
                              max_context_chars: int = 200) -> str:
        """Create contextual chunk following Anthropic's contextual retrieval pattern.
        
        Implements the technique from https://www.anthropic.com/news/contextual-retrieval
        that achieves 67% reduction in incorrect retrievals.
        
        Args:
            chunk_text: The chunk content
            doc_context: Document context (title, summary, etc.)
            max_context_chars: Maximum context to prepend
            
        Returns:
            Contextualized chunk text
        """
        if not doc_context or len(doc_context) > max_context_chars:
            doc_context = doc_context[:max_context_chars] + "..." if doc_context else ""
        
        # Format: "Document: {context}\n\nContent: {chunk}"
        contextualized = f"Document: {doc_context.strip()}\n\nContent: {chunk_text}"
        
        return contextualized
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for better chunking."""
        if not text:
            return []
        
        # Simple sentence splitting - in production, use spaCy or NLTK
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity for overlap detection.
        
        Returns:
            Similarity score between 0 and 1
        """
        if not text1 or not text2:
            return 0.0
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0


# Global text processor instance
text_processor = TextProcessor()