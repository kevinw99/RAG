"""
Comprehensive evaluation framework using RAGAs metrics.

Optimized for 612MB document library with:
- RAGAs metrics (faithfulness, answer_relevancy, context_precision, context_recall)
- Async evaluation for performance
- Batch evaluation for large test sets
- Custom metrics for domain-specific requirements
- Evaluation result storage and reporting
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import asdict

# RAGAs imports
try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        context_utilization,
        answer_similarity,
    )
    from ragas.testset.generator import TestsetGenerator
    from ragas.testset.evolutions import simple, reasoning, multi_context
    from datasets import Dataset
except ImportError as e:
    logger.error(f"RAGAs not available: {e}")
    evaluate = None
    faithfulness = None
    answer_relevancy = None
    context_precision = None
    context_recall = None

# Core imports
from rag_system.config.settings import settings
from rag_system.core.models import RAGResponse, EvaluationResult, Chunk

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """RAG system evaluator using RAGAs framework.
    
    Provides comprehensive evaluation of RAG system performance.
    """
    
    def __init__(
        self,
        metrics: Optional[List[str]] = None,
        batch_size: int = None,
        enable_custom_metrics: bool = True,
    ):
        """Initialize RAG evaluator.
        
        Args:
            metrics: List of metric names to use
            batch_size: Batch size for evaluation
            enable_custom_metrics: Enable custom domain-specific metrics
        """
        self.batch_size = batch_size or settings.evaluation_batch_size
        self.enable_custom_metrics = enable_custom_metrics
        
        # Initialize metrics
        self.metrics = self._initialize_metrics(metrics)
        
        # Track evaluation statistics
        self.stats = {
            "total_evaluations": 0,
            "successful_evaluations": 0,
            "failed_evaluations": 0,
            "total_evaluation_time": 0.0,
            "avg_faithfulness": 0.0,
            "avg_answer_relevancy": 0.0,
            "avg_context_precision": 0.0,
            "avg_context_recall": 0.0,
        }
        
        logger.info(
            f"RAGEvaluator initialized with metrics: {list(self.metrics.keys())}"
        )
    
    def _initialize_metrics(self, metric_names: Optional[List[str]]) -> Dict[str, Any]:
        """Initialize RAGAs metrics.
        
        Args:
            metric_names: List of metric names to use
            
        Returns:
            Dictionary of metric name to metric object
        """
        if not evaluate:
            logger.error("RAGAs not available, evaluation disabled")
            return {}
        
        # Default metrics if none specified
        if not metric_names:
            metric_names = settings.evaluation_metrics
        
        metrics = {}
        
        # Available RAGAs metrics
        available_metrics = {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
            "context_recall": context_recall,
            "context_utilization": context_utilization,
            "answer_similarity": answer_similarity,
        }
        
        for metric_name in metric_names:
            if metric_name in available_metrics and available_metrics[metric_name]:
                metrics[metric_name] = available_metrics[metric_name]
                logger.info(f"Loaded metric: {metric_name}")
            else:
                logger.warning(f"Metric not available: {metric_name}")
        
        return metrics
    
    async def evaluate_response(
        self,
        query: str,
        response: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EvaluationResult:
        """Evaluate a single RAG response.
        
        Args:
            query: Original query
            response: Generated response
            contexts: List of context strings used for generation
            ground_truth: Ground truth answer (optional)
            metadata: Additional metadata
            
        Returns:
            EvaluationResult with metric scores
        """
        start_time = time.time()
        
        if not self.metrics:
            logger.warning("No metrics available for evaluation")
            return EvaluationResult(
                query=query,
                response=response,
                ground_truth=ground_truth,
                evaluation_time=time.time() - start_time,
                metadata=metadata or {},
            )
        
        try:
            # Prepare dataset for RAGAs
            eval_data = {
                "question": [query],
                "answer": [response],
                "contexts": [contexts],
            }
            
            # Add ground truth if available
            if ground_truth:
                eval_data["ground_truth"] = [ground_truth]
            
            # Create dataset
            dataset = Dataset.from_dict(eval_data)
            
            # Filter metrics based on available data
            applicable_metrics = self._filter_applicable_metrics(ground_truth is not None)
            
            if not applicable_metrics:
                logger.warning("No applicable metrics for evaluation")
                return EvaluationResult(
                    query=query,
                    response=response,
                    ground_truth=ground_truth,
                    evaluation_time=time.time() - start_time,
                    metadata=metadata or {},
                )
            
            # Run evaluation
            results = evaluate(dataset, metrics=list(applicable_metrics.values()))
            
            # Extract scores
            result = EvaluationResult(
                query=query,
                response=response,
                ground_truth=ground_truth,
                evaluation_time=time.time() - start_time,
                metadata=metadata or {},
            )
            
            # Populate metric scores
            for metric_name in applicable_metrics.keys():
                if metric_name in results:
                    score = float(results[metric_name][0]) if results[metric_name] else None
                    setattr(result, metric_name, score)
            
            # Compute overall score
            result.compute_overall_score()
            
            # Update statistics
            self._update_stats(result, success=True)
            
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating response: {e}")
            
            # Update statistics
            evaluation_time = time.time() - start_time
            self._update_stats(None, success=False, evaluation_time=evaluation_time)
            
            return EvaluationResult(
                query=query,
                response=response,
                ground_truth=ground_truth,
                evaluation_time=evaluation_time,
                metadata={(metadata or {}).update({"error": str(e)})[0]},
            )
    
    def _filter_applicable_metrics(self, has_ground_truth: bool) -> Dict[str, Any]:
        """Filter metrics based on available data.
        
        Args:
            has_ground_truth: Whether ground truth is available
            
        Returns:
            Dictionary of applicable metrics
        """
        applicable = {}
        
        for name, metric in self.metrics.items():
            # Some metrics require ground truth
            if name in ["context_recall", "answer_similarity"] and not has_ground_truth:
                continue
            applicable[name] = metric
        
        return applicable
    
    async def evaluate_batch(
        self,
        evaluation_data: List[Dict[str, Any]],
        show_progress: bool = True,
    ) -> List[EvaluationResult]:
        """Evaluate a batch of RAG responses.
        
        Args:
            evaluation_data: List of evaluation data dictionaries
            show_progress: Show progress bar
            
        Returns:
            List of EvaluationResult objects
        """
        if not evaluation_data:
            return []
        
        logger.info(f"Evaluating batch of {len(evaluation_data)} responses")
        
        results = []
        
        # Process in batches to manage memory
        for i in range(0, len(evaluation_data), self.batch_size):
            batch_data = evaluation_data[i:i + self.batch_size]
            
            # Create tasks for concurrent evaluation
            tasks = []
            for data in batch_data:
                task = self.evaluate_response(
                    query=data["query"],
                    response=data["response"],
                    contexts=data["contexts"],
                    ground_truth=data.get("ground_truth"),
                    metadata=data.get("metadata"),
                )
                tasks.append(task)
            
            # Execute batch concurrently
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for result in batch_results:
                if isinstance(result, EvaluationResult):
                    results.append(result)
                elif isinstance(result, Exception):
                    logger.error(f"Evaluation failed: {result}")
                    # Create placeholder result
                    results.append(EvaluationResult(
                        query="",
                        response="",
                        evaluation_time=0.0,
                        metadata={"error": str(result)},
                    ))
            
            if show_progress:
                logger.info(f"Completed batch {i//self.batch_size + 1}/{(len(evaluation_data)-1)//self.batch_size + 1}")
        
        logger.info(f"Batch evaluation completed: {len(results)} results")
        return results
    
    async def evaluate_rag_pipeline(
        self,
        test_queries: List[str],
        rag_pipeline,  # RAGPipeline instance
        ground_truths: Optional[List[str]] = None,
        show_progress: bool = True,
    ) -> List[EvaluationResult]:
        """Evaluate a RAG pipeline end-to-end.
        
        Args:
            test_queries: List of test queries
            rag_pipeline: RAG pipeline to evaluate
            ground_truths: Optional ground truth answers
            show_progress: Show progress bar
            
        Returns:
            List of evaluation results
        """
        logger.info(f"Evaluating RAG pipeline with {len(test_queries)} queries")
        
        evaluation_data = []
        
        # Generate responses for all queries
        for i, query in enumerate(test_queries):
            try:
                # Get response from RAG pipeline
                rag_response = await rag_pipeline.process_query(query)
                
                # Extract contexts from retrieval results
                contexts = []
                if rag_response.retrieval_results:
                    contexts = [chunk.content for chunk in rag_response.retrieval_results.chunks]
                
                # Prepare evaluation data
                eval_item = {
                    "query": query,
                    "response": rag_response.answer,
                    "contexts": contexts,
                    "metadata": {
                        "response_time": rag_response.response_time,
                        "confidence_score": rag_response.confidence_score,
                        "num_sources": len(rag_response.sources),
                    },
                }
                
                # Add ground truth if available
                if ground_truths and i < len(ground_truths):
                    eval_item["ground_truth"] = ground_truths[i]
                
                evaluation_data.append(eval_item)
                
            except Exception as e:
                logger.error(f"Error processing query '{query}': {e}")
                continue
        
        # Evaluate all responses
        return await self.evaluate_batch(evaluation_data, show_progress)
    
    def generate_evaluation_report(
        self,
        results: List[EvaluationResult],
        include_individual: bool = False,
    ) -> Dict[str, Any]:
        """Generate comprehensive evaluation report.
        
        Args:
            results: List of evaluation results
            include_individual: Include individual result details
            
        Returns:
            Dictionary with evaluation report
        """
        if not results:
            return {"error": "No evaluation results provided"}
        
        # Calculate aggregate metrics
        valid_results = [r for r in results if r.overall_score is not None]
        
        if not valid_results:
            return {"error": "No valid evaluation results"}
        
        report = {
            "summary": {
                "total_evaluations": len(results),
                "valid_evaluations": len(valid_results),
                "success_rate": len(valid_results) / len(results),
            },
            "metrics": {},
            "performance": {
                "avg_evaluation_time": sum(r.evaluation_time for r in results) / len(results),
                "total_evaluation_time": sum(r.evaluation_time for r in results),
            },
        }
        
        # Calculate metric averages
        metrics_to_average = [
            "faithfulness",
            "answer_relevancy", 
            "context_precision",
            "context_recall",
            "overall_score",
        ]
        
        for metric in metrics_to_average:
            values = [getattr(r, metric) for r in valid_results if getattr(r, metric) is not None]
            if values:
                report["metrics"][metric] = {
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values),
                }
        
        # Add distribution analysis
        if "overall_score" in report["metrics"]:
            overall_scores = [r.overall_score for r in valid_results if r.overall_score is not None]
            if overall_scores:
                # Quality thresholds
                excellent = sum(1 for s in overall_scores if s >= 0.8) / len(overall_scores)
                good = sum(1 for s in overall_scores if 0.6 <= s < 0.8) / len(overall_scores)
                fair = sum(1 for s in overall_scores if 0.4 <= s < 0.6) / len(overall_scores)
                poor = sum(1 for s in overall_scores if s < 0.4) / len(overall_scores)
                
                report["quality_distribution"] = {
                    "excellent (>=0.8)": excellent,
                    "good (0.6-0.8)": good,
                    "fair (0.4-0.6)": fair,
                    "poor (<0.4)": poor,
                }
        
        # Include individual results if requested
        if include_individual:
            report["individual_results"] = [
                {
                    "query": r.query[:100] + "..." if len(r.query) > 100 else r.query,
                    "faithfulness": r.faithfulness,
                    "answer_relevancy": r.answer_relevancy,
                    "context_precision": r.context_precision,
                    "context_recall": r.context_recall,
                    "overall_score": r.overall_score,
                    "evaluation_time": r.evaluation_time,
                }
                for r in valid_results
            ]
        
        return report
    
    def _update_stats(
        self,
        result: Optional[EvaluationResult],
        success: bool,
        evaluation_time: float = 0.0,
    ) -> None:
        """Update evaluation statistics.
        
        Args:
            result: Evaluation result (if successful)
            success: Whether evaluation was successful
            evaluation_time: Time taken for evaluation
        """
        self.stats["total_evaluations"] += 1
        
        if success and result:
            self.stats["successful_evaluations"] += 1
            self.stats["total_evaluation_time"] += result.evaluation_time
            
            # Update running averages
            successful_count = self.stats["successful_evaluations"]
            
            if result.faithfulness is not None:
                current_avg = self.stats["avg_faithfulness"]
                self.stats["avg_faithfulness"] = (
                    (current_avg * (successful_count - 1) + result.faithfulness) / successful_count
                )
            
            if result.answer_relevancy is not None:
                current_avg = self.stats["avg_answer_relevancy"]
                self.stats["avg_answer_relevancy"] = (
                    (current_avg * (successful_count - 1) + result.answer_relevancy) / successful_count
                )
            
            if result.context_precision is not None:
                current_avg = self.stats["avg_context_precision"]
                self.stats["avg_context_precision"] = (
                    (current_avg * (successful_count - 1) + result.context_precision) / successful_count
                )
            
            if result.context_recall is not None:
                current_avg = self.stats["avg_context_recall"]
                self.stats["avg_context_recall"] = (
                    (current_avg * (successful_count - 1) + result.context_recall) / successful_count
                )
        else:
            self.stats["failed_evaluations"] += 1
            self.stats["total_evaluation_time"] += evaluation_time
    
    def get_evaluation_stats(self) -> Dict[str, Any]:
        """Get evaluation statistics.
        
        Returns:
            Dictionary with evaluation statistics
        """
        stats = self.stats.copy()
        
        if stats["total_evaluations"] > 0:
            stats["success_rate"] = stats["successful_evaluations"] / stats["total_evaluations"]
            stats["failure_rate"] = stats["failed_evaluations"] / stats["total_evaluations"]
            stats["avg_evaluation_time"] = stats["total_evaluation_time"] / stats["total_evaluations"]
        
        return stats
    
    def reset_stats(self) -> None:
        """Reset evaluation statistics."""
        self.stats = {
            "total_evaluations": 0,
            "successful_evaluations": 0,
            "failed_evaluations": 0,
            "total_evaluation_time": 0.0,
            "avg_faithfulness": 0.0,
            "avg_answer_relevancy": 0.0,
            "avg_context_precision": 0.0,
            "avg_context_recall": 0.0,
        }


class TestDatasetGenerator:
    """Generate test datasets for RAG evaluation."""
    
    def __init__(self, documents: List[str]):
        """Initialize test dataset generator.
        
        Args:
            documents: List of document texts
        """
        self.documents = documents
        
        if not evaluate:
            logger.error("RAGAs not available, test generation disabled")
            self.generator = None
        else:
            try:
                self.generator = TestsetGenerator.from_langchain(
                    generator_llm=None,  # Use default
                    critic_llm=None,     # Use default
                )
            except Exception as e:
                logger.error(f"Failed to initialize test generator: {e}")
                self.generator = None
    
    async def generate_test_dataset(
        self,
        test_size: int = 10,
        distributions: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        """Generate test dataset from documents.
        
        Args:
            test_size: Number of test cases to generate
            distributions: Distribution of question types
            
        Returns:
            List of test cases
        """
        if not self.generator:
            logger.error("Test generator not available")
            return []
        
        try:
            # Default distributions
            if not distributions:
                distributions = {
                    simple: 0.5,
                    reasoning: 0.3,
                    multi_context: 0.2,
                }
            
            # Generate test dataset
            testset = self.generator.generate_with_langchain_docs(
                documents=self.documents,
                test_size=test_size,
                distributions=distributions,
            )
            
            # Convert to list of dictionaries
            test_cases = []
            for item in testset:
                test_cases.append({
                    "query": item.question,
                    "ground_truth": item.ground_truth,
                    "contexts": item.contexts,
                    "evolution_type": item.evolution_type,
                    "metadata": item.metadata,
                })
            
            logger.info(f"Generated {len(test_cases)} test cases")
            return test_cases
            
        except Exception as e:
            logger.error(f"Error generating test dataset: {e}")
            return []


# Convenience functions
def create_evaluator(**kwargs) -> RAGEvaluator:
    """Create RAG evaluator with default settings.
    
    Args:
        **kwargs: Arguments to override defaults
        
    Returns:
        RAGEvaluator instance
    """
    return RAGEvaluator(**kwargs)


async def quick_evaluate(
    query: str,
    response: str,
    contexts: List[str],
    ground_truth: Optional[str] = None,
) -> EvaluationResult:
    """Quick evaluation of a single response.
    
    Args:
        query: Original query
        response: Generated response
        contexts: Context strings
        ground_truth: Ground truth answer (optional)
        
    Returns:
        EvaluationResult
    """
    evaluator = create_evaluator()
    return await evaluator.evaluate_response(
        query=query,
        response=response,
        contexts=contexts,
        ground_truth=ground_truth,
    )
