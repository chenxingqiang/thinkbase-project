"""Test inference instructions."""

import pytest
import asyncio
from typing import Dict, Any, List
import json
import numpy as np

from agentflow.core.instructions.inference import (
    TextClassificationInstruction,
    SentimentAnalysisInstruction,
    NERInstruction,
    TextSimilarityInstruction,
    QuestionAnsweringInstruction,
    CausalInferenceInstruction,
    LogicalReasoningInstruction,
    KnowledgeGraphInstruction
)
from agentflow.core.instructions.base import InstructionStatus

@pytest.fixture
def text_classifier():
    """Create TextClassificationInstruction instance."""
    return TextClassificationInstruction()

@pytest.fixture
def sentiment_analyzer():
    """Create SentimentAnalysisInstruction instance."""
    return SentimentAnalysisInstruction()

@pytest.fixture
def ner_extractor():
    """Create NERInstruction instance."""
    return NERInstruction()

@pytest.fixture
def text_similarity():
    """Create TextSimilarityInstruction instance."""
    return TextSimilarityInstruction()

@pytest.fixture
def qa_system():
    """Create QuestionAnsweringInstruction instance."""
    return QuestionAnsweringInstruction()

@pytest.fixture
def causal_inference():
    """Create CausalInferenceInstruction instance."""
    return CausalInferenceInstruction()

@pytest.fixture
def logical_reasoning():
    """Create LogicalReasoningInstruction instance."""
    return LogicalReasoningInstruction()

@pytest.fixture
def knowledge_graph():
    """Create KnowledgeGraphInstruction instance."""
    return KnowledgeGraphInstruction()

@pytest.fixture
def sample_texts():
    """Create sample texts for testing."""
    return [
        "The product is excellent and I love it!",
        "This service is terrible and I want a refund.",
        "The weather is nice today.",
        "Python is a great programming language for AI development.",
        "Breaking: Major tech company announces new AI chip."
    ]

@pytest.fixture
def sample_qa_pairs():
    """Create sample QA pairs for testing."""
    return [
        {
            "context": "Python is a high-level programming language known for its simplicity and readability.",
            "question": "What kind of language is Python?",
            "answer": "a high-level programming language"
        },
        {
            "context": "The Earth completes one rotation around its axis in approximately 24 hours.",
            "question": "How long does it take for Earth to rotate once?",
            "answer": "approximately 24 hours"
        }
    ]

@pytest.mark.asyncio
async def test_text_classification_basic(text_classifier, sample_texts):
    """Test basic text classification functionality."""
    context = {
        "texts": sample_texts,
        "labels": ["positive", "negative", "neutral", "technical", "news"],
        "multi_label": False
    }
    
    result = await text_classifier.execute(context)
    assert result.status == InstructionStatus.COMPLETED
    assert "predictions" in result.data
    assert len(result.data["predictions"]) == len(sample_texts)
    assert "confidence_scores" in result.data
    assert "metrics" in result.data

@pytest.mark.asyncio
async def test_sentiment_analysis_basic(sentiment_analyzer, sample_texts):
    """Test basic sentiment analysis functionality."""
    context = {
        "texts": sample_texts,
        "include_neutral": True,
        "include_aspects": False
    }
    
    result = await sentiment_analyzer.execute(context)
    assert result.status == InstructionStatus.COMPLETED
    assert "sentiments" in result.data
    assert len(result.data["sentiments"]) == len(sample_texts)
    assert all("score" in s for s in result.data["sentiments"])
    assert "metrics" in result.data

@pytest.mark.asyncio
async def test_ner_basic(ner_extractor, sample_texts):
    """Test basic named entity recognition functionality."""
    context = {
        "texts": sample_texts,
        "entity_types": ["PERSON", "ORG", "LOC", "PRODUCT"],
        "include_spans": True
    }
    
    result = await ner_extractor.execute(context)
    assert result.status == InstructionStatus.COMPLETED
    assert "entities" in result.data
    assert len(result.data["entities"]) == len(sample_texts)
    assert "metrics" in result.data

@pytest.mark.asyncio
async def test_text_similarity_basic(text_similarity, sample_texts):
    """Test basic text similarity functionality."""
    context = {
        "text1": sample_texts[0],
        "text2": sample_texts[1],
        "metrics": ["cosine", "jaccard"],
        "use_embeddings": True
    }
    
    result = await text_similarity.execute(context)
    assert result.status == InstructionStatus.COMPLETED
    assert "similarity_scores" in result.data
    assert all(m in result.data["similarity_scores"] for m in context["metrics"])
    assert "metrics" in result.data

@pytest.mark.asyncio
async def test_qa_basic(qa_system, sample_qa_pairs):
    """Test basic question answering functionality."""
    context = {
        "qa_pairs": sample_qa_pairs,
        "max_answer_length": 50,
        "include_confidence": True
    }
    
    result = await qa_system.execute(context)
    assert result.status == InstructionStatus.COMPLETED
    assert "answers" in result.data
    assert len(result.data["answers"]) == len(sample_qa_pairs)
    assert all("answer" in a for a in result.data["answers"])
    assert all("confidence" in a for a in result.data["answers"])
    assert "metrics" in result.data

@pytest.mark.asyncio
async def test_causal_inference_basic(causal_inference):
    """Test basic causal inference functionality."""
    context = {
        "data": {
            "treatment": [1, 0, 1, 0, 1],
            "outcome": [5.2, 3.1, 4.8, 3.3, 5.5],
            "covariates": {
                "age": [25, 30, 35, 40, 45],
                "gender": ["M", "F", "M", "F", "M"]
            }
        },
        "method": "matching",
        "treatment_col": "treatment",
        "outcome_col": "outcome"
    }
    
    result = await causal_inference.execute(context)
    assert result.status == InstructionStatus.COMPLETED
    assert "ate" in result.data  # Average Treatment Effect
    assert "confidence_interval" in result.data
    assert "metrics" in result.data

@pytest.mark.asyncio
async def test_logical_reasoning_basic(logical_reasoning):
    """Test basic logical reasoning functionality."""
    context = {
        "premises": [
            "All birds can fly",
            "Penguins are birds",
            "Penguins cannot fly"
        ],
        "query": "Is there a contradiction?",
        "reasoning_type": "deductive"
    }
    
    result = await logical_reasoning.execute(context)
    assert result.status == InstructionStatus.COMPLETED
    assert "conclusion" in result.data
    assert "explanation" in result.data
    assert "confidence" in result.data
    assert "metrics" in result.data

@pytest.mark.asyncio
async def test_knowledge_graph_basic(knowledge_graph):
    """Test basic knowledge graph functionality."""
    context = {
        "triples": [
            ("Python", "is_a", "programming_language"),
            ("Python", "created_by", "Guido_van_Rossum"),
            ("programming_language", "used_for", "software_development")
        ],
        "query": ("Python", "used_for", "?"),
        "inference_depth": 2
    }
    
    result = await knowledge_graph.execute(context)
    assert result.status == InstructionStatus.COMPLETED
    assert "inferred_triples" in result.data
    assert "confidence_scores" in result.data
    assert "metrics" in result.data

@pytest.mark.asyncio
async def test_batch_processing(text_classifier, sample_texts):
    """Test batch processing capability."""
    batch_size = 2
    total_texts = len(sample_texts)
    
    context = {
        "texts": sample_texts,
        "labels": ["positive", "negative", "neutral", "technical", "news"],
        "batch_size": batch_size
    }
    
    result = await text_classifier.execute(context)
    assert result.status == InstructionStatus.COMPLETED
    assert len(result.data["predictions"]) == total_texts
    assert "batch_metrics" in result.data
    assert len(result.data["batch_metrics"]) == (total_texts + batch_size - 1) // batch_size

@pytest.mark.asyncio
async def test_error_handling(text_classifier):
    """Test error handling."""
    # Test missing required parameters
    context = {
        "texts": []  # Empty texts
    }
    
    result = await text_classifier.execute(context)
    assert result.status == InstructionStatus.FAILED
    assert result.error is not None
    
    # Test invalid input format
    context = {
        "texts": "not a list",  # Invalid input type
        "labels": ["positive", "negative"]
    }
    
    result = await text_classifier.execute(context)
    assert result.status == InstructionStatus.FAILED
    assert result.error is not None

@pytest.mark.asyncio
async def test_concurrent_execution(text_classifier, sentiment_analyzer, sample_texts):
    """Test concurrent execution of multiple instructions."""
    tasks = []
    
    # Create multiple classification and sentiment analysis tasks
    for _ in range(3):
        tasks.append(text_classifier.execute({
            "texts": sample_texts,
            "labels": ["positive", "negative", "neutral"]
        }))
        tasks.append(sentiment_analyzer.execute({
            "texts": sample_texts
        }))
    
    # Execute tasks concurrently
    results = await asyncio.gather(*tasks)
    
    # Verify all tasks completed successfully
    assert all(r.status == InstructionStatus.COMPLETED for r in results)
    assert len(results) == 6  # 3 classification + 3 sentiment analysis tasks

@pytest.mark.asyncio
async def test_model_caching(text_classifier, sample_texts):
    """Test model caching functionality."""
    context = {
        "texts": sample_texts,
        "labels": ["positive", "negative", "neutral"],
        "use_cache": True
    }
    
    # First execution (cold start)
    start_time = asyncio.get_event_loop().time()
    result1 = await text_classifier.execute(context)
    cold_start_time = asyncio.get_event_loop().time() - start_time
    
    # Second execution (warm start)
    start_time = asyncio.get_event_loop().time()
    result2 = await text_classifier.execute(context)
    warm_start_time = asyncio.get_event_loop().time() - start_time
    
    assert result1.status == InstructionStatus.COMPLETED
    assert result2.status == InstructionStatus.COMPLETED
    assert warm_start_time < cold_start_time  # Warm start should be faster

@pytest.mark.asyncio
async def test_result_validation(text_classifier, sample_texts):
    """Test result validation functionality."""
    context = {
        "texts": sample_texts,
        "labels": ["positive", "negative", "neutral"],
        "validation_rules": {
            "min_confidence": 0.5,
            "allowed_labels": ["positive", "negative", "neutral"]
        }
    }
    
    result = await text_classifier.execute(context)
    assert result.status == InstructionStatus.COMPLETED
    assert "validation_results" in result.data
    assert "validation_passed" in result.data["validation_results"]
    assert "validation_errors" in result.data["validation_results"]

if __name__ == "__main__":
    pytest.main([__file__]) 