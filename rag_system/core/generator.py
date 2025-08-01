"""Response generation system with LangChain integration and confidence scoring.

Features:
- Multi-provider LLM support (OpenAI, Anthropic, local models)
- Confidence scoring and response validation
- Source attribution and citation generation
- Prompt templates for different query types
- Error handling with graceful degradation
"""

import logging
import time
import math
import asyncio
from typing import List, Dict, Optional, Any, Tuple, Union
from datetime import datetime

# LangChain imports
from langchain.llms.base import BaseLLM
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import HumanMessage, SystemMessage

from ..core.data_models import Chunk, RAGResponse, RetrievalResult
from ..config.settings import settings

logger = logging.getLogger(__name__)


class PromptTemplates:
    """Collection of prompt templates for different RAG scenarios."""
    
    DEFAULT_SYSTEM_PROMPT = """You are a knowledgeable assistant that answers questions based on provided context. 
Follow these guidelines:

1. Use ONLY the provided context to answer the question
2. If the context doesn't contain enough information, say so clearly
3. Include specific references to sources when possible
4. Be concise but comprehensive
5. If uncertain, indicate your level of confidence

Context will be provided as numbered chunks with source information."""

    DEFAULT_USER_PROMPT = """Context:
{context}

Question: {question}

Please provide a comprehensive answer based on the context above. Include source references where appropriate."""

    CITATION_PROMPT = """You are a helpful assistant that answers questions using provided context and includes proper citations.

Rules:
1. Answer based ONLY on the provided context
2. Include citations in format [Source: filename, page/section if available]
3. If context is insufficient, state this clearly
4. Be accurate and concise

Context:
{context}

Question: {question}

Answer with citations:"""

    SUMMARY_PROMPT = """Based on the provided context, create a comprehensive summary that addresses the question.

Context:
{context}

Question: {question}

Summary:"""

    COMPARISON_PROMPT = """Compare and contrast information from the provided sources to answer the question.

Context:
{context}

Question: {question}

Comparison:"""


class ConfidenceScorer:
    """Compute confidence scores for generated responses."""
    
    def __init__(self):
        """Initialize confidence scorer."""
        self.factors = {
            'context_relevance': 0.3,
            'answer_completeness': 0.25,
            'source_coverage': 0.2,
            'response_coherence': 0.15,
            'uncertainty_indicators': 0.1
        }
    
    async def compute_confidence(self, 
                                query: str,
                                context_chunks: List[Chunk],
                                response: str,
                                retrieval_scores: List[float] = None) -> float:
        """Compute confidence score for a RAG response.
        
        Args:
            query: Original query
            context_chunks: Retrieved context chunks
            response: Generated response
            retrieval_scores: Retrieval confidence scores
            
        Returns:
            Confidence score between 0 and 1
        """
        if not response or not context_chunks:
            return 0.0
        
        try:
            scores = {}
            
            # Context relevance (based on retrieval scores)
            scores['context_relevance'] = self._score_context_relevance(
                retrieval_scores or [0.5] * len(context_chunks)
            )
            
            # Answer completeness (response length and structure)
            scores['answer_completeness'] = self._score_answer_completeness(
                query, response
            )
            
            # Source coverage (how many sources were used)
            scores['source_coverage'] = self._score_source_coverage(
                context_chunks, response
            )
            
            # Response coherence (linguistic quality)
            scores['response_coherence'] = self._score_response_coherence(response)
            
            # Uncertainty indicators (hedging language)
            scores['uncertainty_indicators'] = self._score_uncertainty_indicators(response)
            
            # Weighted average
            confidence = sum(
                scores[factor] * weight 
                for factor, weight in self.factors.items()
            )
            
            final_confidence = min(max(confidence, 0.0), 1.0)  # Clamp to [0, 1]
            
            return final_confidence
            
        except Exception as e:
            logger.error(f"Error computing confidence score: {e}")
            import traceback
            logger.error(f"Confidence scoring traceback: {traceback.format_exc()}")
            return 0.5  # Default moderate confidence
    
    def _score_context_relevance(self, retrieval_scores: List[float]) -> float:
        """Score based on retrieval confidence."""
        if not retrieval_scores:
            return 0.5
        
        # Filter valid scores and normalize if needed
        valid_scores = [score for score in retrieval_scores if isinstance(score, (int, float)) and not math.isnan(score)]
        
        if not valid_scores:
            return 0.5
            
        # Normalize scores to [0, 1] range if they're not already
        # Assume scores are similarity scores (higher = better)
        normalized_scores = []
        for score in valid_scores:
            # If scores are already in [0, 1], keep them
            if 0 <= score <= 1:
                normalized_scores.append(score)
            # If scores are similarity scores (could be negative), normalize
            else:
                # Convert to [0, 1] range: (score + 1) / 2 for cosine similarity
                normalized_score = max(0, min(1, (score + 1) / 2))
                normalized_scores.append(normalized_score)
        
        if not normalized_scores:
            return 0.5
        
        # Average of top retrieval scores
        top_scores = sorted(normalized_scores, reverse=True)[:3]
        return sum(top_scores) / len(top_scores)
    
    def _score_answer_completeness(self, query: str, response: str) -> float:
        """Score based on response completeness."""
        if not response or not response.strip():
            return 0.0
        
        try:
            # Simple heuristics for completeness
            words = response.split()
            word_count = len(words)
            sentences = [s for s in response.split('.') if s.strip()]
            sentence_count = len(sentences)
            
            if word_count == 0:
                return 0.0
            
            # Optimal range: 50-200 words, 3-8 sentences
            if word_count < 100:
                word_score = min(word_count / 100.0, 1.0)
            else:
                word_score = max(1.0 - (word_count - 200) / 300.0, 0.5)
            
            sentence_score = min(sentence_count / 5.0, 1.0) if sentence_count > 0 else 0.5
            
            final_score = (word_score + sentence_score) / 2.0
            return max(0.0, min(1.0, final_score))  # Ensure [0, 1] range
            
        except Exception as e:
            logger.warning(f"Error in answer completeness scoring: {e}")
            return 0.5
    
    def _score_source_coverage(self, chunks: List[Chunk], response: str) -> float:
        """Score based on how well sources are represented."""
        if not chunks:
            return 0.0
        
        try:
            # Count unique sources referenced
            unique_sources = set()
            for chunk in chunks:
                if hasattr(chunk, 'metadata') and chunk.metadata:
                    filename = chunk.metadata.get('filename', '')
                    if filename:
                        unique_sources.add(filename)
            
            # Heuristic: good responses reference multiple sources
            source_diversity = min(len(unique_sources) / 3.0, 1.0) if unique_sources else 0.3
            
            # Check if response mentions sources or shows synthesis
            if response:
                response_lower = response.lower()
                synthesis_indicators = ['according to', 'based on', 'as stated in', 'multiple sources', 'source', 'document']
                synthesis_count = sum(1 for indicator in synthesis_indicators if indicator in response_lower)
                synthesis_score = min(synthesis_count / 2.0, 1.0)
            else:
                synthesis_score = 0.0
            
            final_score = (source_diversity + synthesis_score) / 2.0
            return max(0.0, min(1.0, final_score))
            
        except Exception as e:
            logger.warning(f"Error in source coverage scoring: {e}")
            return 0.5
    
    def _score_response_coherence(self, response: str) -> float:
        """Score response coherence and structure."""
        if not response or not response.strip():
            return 0.0
        
        try:
            # Simple coherence indicators
            sentences = [s.strip() for s in response.split('.') if s.strip()]
            
            if len(sentences) < 1:
                return 0.3
            elif len(sentences) < 2:
                return 0.6
            
            # Check for transition words and logical flow
            transition_words = [
                'however', 'furthermore', 'moreover', 'additionally', 
                'consequently', 'therefore', 'in contrast', 'similarly',
                'first', 'second', 'finally', 'also', 'in addition'
            ]
            
            response_lower = response.lower()
            transition_count = sum(1 for word in transition_words if word in response_lower)
            transition_score = min(transition_count / max(len(sentences), 1), 1.0)
            
            # Check for structured response (paragraphs, lists)
            structure_score = 0.7  # Default decent structure
            if '\n\n' in response or response.count('\n') > 2:
                structure_score = 1.0
            elif '1.' in response or '2.' in response or '-' in response:
                structure_score = 0.9  # Lists indicate good structure
            
            final_score = (transition_score + structure_score) / 2.0
            return max(0.0, min(1.0, final_score))
            
        except Exception as e:
            logger.warning(f"Error in response coherence scoring: {e}")
            return 0.7
    
    def _score_uncertainty_indicators(self, response: str) -> float:
        """Score based on uncertainty language (inverted - less uncertainty = higher score)."""
        if not response or not response.strip():
            return 0.5
        
        try:
            uncertainty_phrases = [
                'i don\'t know', 'uncertain', 'unclear', 'might be', 'possibly',
                'perhaps', 'may be', 'not sure', 'insufficient information',
                'cannot determine', 'unable to answer', 'i\'m not sure',
                'it\'s unclear', 'hard to say', 'difficult to determine'
            ]
            
            response_lower = response.lower()
            uncertainty_count = sum(1 for phrase in uncertainty_phrases if phrase in response_lower)
            
            # More uncertainty = lower confidence
            if uncertainty_count == 0:
                return 1.0
            elif uncertainty_count == 1:
                return 0.7
            elif uncertainty_count == 2:
                return 0.5
            else:
                return max(0.2, 1.0 - uncertainty_count * 0.15)
                
        except Exception as e:
            logger.warning(f"Error in uncertainty scoring: {e}")
            return 0.8  # Default to low uncertainty


class ResponseGenerator:
    """Generate responses using LLM with context and confidence scoring."""
    
    def __init__(self, 
                 llm_provider: str = None,
                 llm_model: str = None,
                 temperature: float = None,
                 max_tokens: int = None):
        """Initialize response generator.
        
        Args:
            llm_provider: LLM provider (openai, anthropic, local)
            llm_model: Model name
            temperature: Generation temperature
            max_tokens: Maximum response tokens
        """
        self.llm_provider = llm_provider or settings.llm_provider
        self.llm_model = llm_model or settings.llm_model
        self.temperature = temperature or settings.llm_temperature
        self.max_tokens = max_tokens or settings.max_tokens
        
        # Initialize LLM
        self.llm = self._init_llm()
        
        # Initialize confidence scorer
        self.confidence_scorer = ConfidenceScorer()
        
        # Prompt templates
        self.templates = PromptTemplates()
        
        # Performance tracking
        self.stats = {
            'total_generations': 0,
            'avg_generation_time': 0.0,
            'avg_confidence_score': 0.0,
            'successful_generations': 0,
            'failed_generations': 0
        }
        
        logger.info(f"ResponseGenerator initialized: {self.llm_provider}/{self.llm_model}")
    
    def _init_llm(self) -> BaseLLM:
        """Initialize LLM based on provider."""
        try:
            if self.llm_provider.lower() == 'openai':
                if not settings.openai_api_key:
                    raise ValueError("OpenAI API key not configured")
                
                return ChatOpenAI(
                    model=self.llm_model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    api_key=settings.openai_api_key
                )
            
            elif self.llm_provider.lower() == 'anthropic':
                if not settings.anthropic_api_key:
                    raise ValueError("Anthropic API key not configured")
                
                return ChatAnthropic(
                    model=self.llm_model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    api_key=settings.anthropic_api_key
                )
            
            else:
                raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
                
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise
    
    async def generate(self, 
                      query: str,
                      retrieval_result: RetrievalResult,
                      template_type: str = "default") -> RAGResponse:
        """Generate response from query and retrieved context.
        
        Args:
            query: User query
            retrieval_result: Retrieved chunks and scores
            template_type: Type of prompt template to use
            
        Returns:
            RAGResponse with answer, sources, and confidence
        """
        start_time = time.time()
        
        if not query.strip():
            return self._create_error_response(query, "Empty query provided", start_time)
        
        if not retrieval_result.chunks:
            return self._create_error_response(query, "No relevant context found", start_time)
        
        try:
            # Prepare context
            context = self._prepare_context(retrieval_result.chunks)
            
            # Generate response
            response_text = await self._generate_response(query, context, template_type)
            
            if not response_text:
                return self._create_error_response(query, "Failed to generate response", start_time)
            
            # Compute confidence score
            confidence = await self.confidence_scorer.compute_confidence(
                query, retrieval_result.chunks, response_text, retrieval_result.scores
            )
            
            # Check confidence threshold
            if confidence < settings.confidence_threshold:
                logger.warning(f"Low confidence response: {confidence:.2f}")
                response_text = self._handle_low_confidence_response(
                    response_text, confidence, retrieval_result.chunks
                )
            
            # Extract sources
            sources = self._extract_sources(retrieval_result.chunks)
            
            # Create response
            response = RAGResponse(
                answer=response_text,
                sources=sources,
                confidence_score=confidence,
                query=query,
                response_time=time.time() - start_time,
                retrieval_result=retrieval_result
            )
            
            # Update statistics
            self._update_stats(True, time.time() - start_time, confidence)
            
            logger.debug(f"Generated response: confidence={confidence:.2f}, "
                        f"time={response.response_time:.3f}s")
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            self._update_stats(False, time.time() - start_time, 0.0)
            return self._create_error_response(query, str(e), start_time)
    
    def _prepare_context(self, chunks: List[Chunk]) -> str:
        """Prepare context string from chunks."""
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            filename = chunk.metadata.get('filename', 'Unknown')
            
            # Format chunk with source info
            chunk_text = f"[{i}] Source: {filename}\n{chunk.content.strip()}"
            context_parts.append(chunk_text)
        
        return "\n\n".join(context_parts)
    
    async def _generate_response(self, query: str, context: str, template_type: str) -> str:
        """Generate response using LLM."""
        try:
            # Select prompt template
            if template_type == "citation":
                prompt_template = self.templates.CITATION_PROMPT
            elif template_type == "summary":
                prompt_template = self.templates.SUMMARY_PROMPT
            elif template_type == "comparison":
                prompt_template = self.templates.COMPARISON_PROMPT
            else:
                prompt_template = self.templates.DEFAULT_USER_PROMPT
            
            # Create messages
            system_message = SystemMessage(content=self.templates.DEFAULT_SYSTEM_PROMPT)
            user_message = HumanMessage(content=prompt_template.format(
                context=context,
                question=query
            ))
            
            # Generate response
            response = await self.llm.ainvoke([system_message, user_message])
            
            # Extract text content
            if hasattr(response, 'content'):
                return response.content.strip()
            else:
                return str(response).strip()
                
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise
    
    def _handle_low_confidence_response(self, response: str, confidence: float, 
                                       chunks: List[Chunk]) -> str:
        """Handle low confidence responses."""
        confidence_note = (
            f"\n\n**Note**: This response has moderate confidence ({confidence:.1%}). "
            f"Please verify the information from the original sources."
        )
        
        return response + confidence_note
    
    def _extract_sources(self, chunks: List[Chunk]) -> List[Dict[str, str]]:
        """Extract source information from chunks."""
        sources = []
        seen_sources = set()
        
        for chunk in chunks:
            filename = chunk.metadata.get('filename', 'Unknown')
            file_path = chunk.metadata.get('file_path', '')
            
            # Create unique source identifier
            source_id = f"{filename}:{chunk.start_char}"
            
            if source_id not in seen_sources:
                source_info = {
                    'filename': filename,
                    'chunk_id': chunk.chunk_id,
                    'doc_id': chunk.doc_id,
                    'start_char': chunk.start_char,
                    'end_char': chunk.end_char,
                }
                
                if file_path:
                    source_info['file_path'] = file_path
                
                sources.append(source_info)
                seen_sources.add(source_id)
        
        return sources
    
    def _create_error_response(self, query: str, error_msg: str, start_time: float) -> RAGResponse:
        """Create error response."""
        return RAGResponse(
            answer=f"I apologize, but I encountered an error: {error_msg}",
            sources=[],
            confidence_score=0.0,
            query=query,
            response_time=time.time() - start_time,
            metadata={'error': error_msg}
        )
    
    def _update_stats(self, success: bool, generation_time: float, confidence: float):
        """Update generation statistics."""
        self.stats['total_generations'] += 1
        
        if success:
            self.stats['successful_generations'] += 1
            
            # Update averages
            prev_avg_time = self.stats['avg_generation_time']
            prev_avg_conf = self.stats['avg_confidence_score']
            n = self.stats['successful_generations']
            
            self.stats['avg_generation_time'] = (
                (prev_avg_time * (n - 1) + generation_time) / n
            )
            self.stats['avg_confidence_score'] = (
                (prev_avg_conf * (n - 1) + confidence) / n
            )
        else:
            self.stats['failed_generations'] += 1
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get generation statistics."""
        stats = dict(self.stats)
        
        if self.stats['total_generations'] > 0:
            stats['success_rate'] = (
                self.stats['successful_generations'] / self.stats['total_generations']
            )
        else:
            stats['success_rate'] = 0.0
        
        return stats
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on response generator."""
        health = {
            'status': 'healthy',
            'llm_provider': self.llm_provider,
            'llm_model': self.llm_model,
            'stats': self.get_generation_stats()
        }
        
        try:
            # Test LLM with simple query
            test_response = await self.llm.ainvoke([
                HumanMessage(content="Hello, please respond with 'OK' if you're working.")
            ])
            
            if hasattr(test_response, 'content'):
                response_text = test_response.content
            else:
                response_text = str(test_response)
            
            if response_text and len(response_text.strip()) > 0:
                health['llm_test'] = 'passed'
            else:
                health['llm_test'] = 'failed'
                health['status'] = 'degraded'
                
        except Exception as e:
            health['llm_test'] = f'failed: {e}'
            health['status'] = 'unhealthy'
        
        return health


# Convenience functions
async def generate_response(query: str, 
                          retrieval_result: RetrievalResult,
                          **kwargs) -> RAGResponse:
    """Convenience function to generate response."""
    generator = ResponseGenerator(**kwargs)
    response = await generator.generate(query, retrieval_result)
    return response