"""Test cases for NLP, MLOps, and QA specialized instructions."""

import pytest
import asyncio
from typing import Dict, Any, List
import time
import numpy as np

from agentflow.core.instructions.specialized import (
    NLPProcessingInstruction,
    MLOpsInstruction,
    QualityAssuranceInstruction
)
from agentflow.core.instructions.base import InstructionStatus

@pytest.fixture
def nlp_processor():
    """Create NLPProcessingInstruction instance."""
    return NLPProcessingInstruction.remote()

@pytest.fixture
def mlops_manager():
    """Create MLOpsInstruction instance."""
    return MLOpsInstruction.remote()

@pytest.fixture
def qa_tester():
    """Create QualityAssuranceInstruction instance."""
    return QualityAssuranceInstruction.remote()

@pytest.fixture
def sample_texts():
    """Create sample texts for testing."""
    return [
        "The product is excellent and I love it!",
        "This service is terrible and I want a refund.",
        "Python is a great programming language.",
        "Breaking: Major tech company announces new AI chip."
    ]

@pytest.mark.asyncio
async def test_nlp_sentiment_analysis(nlp_processor, sample_texts):
    """Test sentiment analysis functionality."""
    context = {
        "text": sample_texts[0],
        "task": "sentiment",
        "model_name": "distilbert-base-uncased-finetuned-sst-2-english"
    }
    
    result = await nlp_processor.execute.remote(context)
    assert result.status == InstructionStatus.COMPLETED
    assert "sentiment" in result.data
    assert "confidence" in result.data
    assert "processing_time" in result.data

@pytest.mark.asyncio
async def test_nlp_text_classification(nlp_processor, sample_texts):
    """Test text classification functionality."""
    context = {
        "text": sample_texts[2],
        "task": "classify",
        "model_name": "bert-base-uncased"
    }
    
    result = await nlp_processor.execute.remote(context)
    assert result.status == InstructionStatus.COMPLETED
    assert "category" in result.data
    assert "confidence" in result.data
    assert "processing_time" in result.data

@pytest.mark.asyncio
async def test_nlp_text_summarization(nlp_processor):
    """Test text summarization functionality."""
    long_text = " ".join(["This is a long text that needs to be summarized."] * 10)
    context = {
        "text": long_text,
        "task": "summarize",
        "model_name": "facebook/bart-large-cnn"
    }
    
    result = await nlp_processor.execute.remote(context)
    assert result.status == InstructionStatus.COMPLETED
    assert "summary" in result.data
    assert "processing_time" in result.data

@pytest.mark.asyncio
async def test_nlp_translation(nlp_processor):
    """Test translation functionality."""
    context = {
        "text": "Hello, how are you?",
        "task": "translate",
        "model_name": "Helsinki-NLP/opus-mt-en-fr"
    }
    
    result = await nlp_processor.execute.remote(context)
    assert result.status == InstructionStatus.COMPLETED
    assert "translation" in result.data
    assert "processing_time" in result.data

@pytest.mark.asyncio
async def test_mlops_model_deployment(mlops_manager):
    """Test model deployment functionality."""
    context = {
        "task": "deploy",
        "model_id": "test-model-123",
        "context": {
            "environment": "staging",
            "version": "1.0.0",
            "resources": {
                "cpu": "2",
                "memory": "4Gi"
            }
        }
    }
    
    result = await mlops_manager.execute.remote(context)
    assert result.status == InstructionStatus.COMPLETED
    assert "deployment_id" in result.data
    assert "status" in result.data
    assert "metrics" in result.data

@pytest.mark.asyncio
async def test_mlops_model_monitoring(mlops_manager):
    """Test model monitoring functionality."""
    context = {
        "task": "monitor",
        "model_id": "test-model-123"
    }
    
    result = await mlops_manager.execute.remote(context)
    assert result.status == InstructionStatus.COMPLETED
    assert "metrics" in result.data
    assert "alerts" in result.data
    assert "status" in result.data

@pytest.mark.asyncio
async def test_mlops_model_rollback(mlops_manager):
    """Test model rollback functionality."""
    context = {
        "task": "rollback",
        "model_id": "test-model-123"
    }
    
    result = await mlops_manager.execute.remote(context)
    assert result.status == InstructionStatus.COMPLETED
    assert "rollback_status" in result.data
    assert "previous_version" in result.data

@pytest.mark.asyncio
async def test_mlops_deployment_scaling(mlops_manager):
    """Test deployment scaling functionality."""
    context = {
        "task": "scale",
        "model_id": "test-model-123",
        "context": {
            "replicas": 3,
            "resources": {
                "cpu": "4",
                "memory": "8Gi"
            }
        }
    }
    
    result = await mlops_manager.execute.remote(context)
    assert result.status == InstructionStatus.COMPLETED
    assert "scaling_status" in result.data
    assert "current_replicas" in result.data

@pytest.mark.asyncio
async def test_qa_run_tests(qa_tester):
    """Test running test suite functionality."""
    context = {
        "task": "run_tests",
        "target": "./tests",
        "context": {
            "test_type": "unit",
            "pattern": "test_*.py",
            "parallel": True
        }
    }
    
    result = await qa_tester.execute.remote(context)
    assert result.status == InstructionStatus.COMPLETED
    assert "test_results" in result.data
    assert "coverage" in result.data
    assert "execution_time" in result.data

@pytest.mark.asyncio
async def test_qa_code_linting(qa_tester):
    """Test code linting functionality."""
    context = {
        "task": "lint",
        "target": "./src",
        "context": {
            "linters": ["pylint", "flake8"],
            "config": {
                "max-line-length": 100
            }
        }
    }
    
    result = await qa_tester.execute.remote(context)
    assert result.status == InstructionStatus.COMPLETED
    assert "lint_results" in result.data
    assert "issues_found" in result.data
    assert "quality_score" in result.data

@pytest.mark.asyncio
async def test_qa_coverage_analysis(qa_tester):
    """Test coverage analysis functionality."""
    context = {
        "task": "coverage",
        "target": "./src",
        "context": {
            "include_branches": True,
            "min_coverage": 80
        }
    }
    
    result = await qa_tester.execute.remote(context)
    assert result.status == InstructionStatus.COMPLETED
    assert "coverage_data" in result.data
    assert "total_coverage" in result.data
    assert "uncovered_lines" in result.data

@pytest.mark.asyncio
async def test_qa_security_scan(qa_tester):
    """Test security scanning functionality."""
    context = {
        "task": "security_scan",
        "target": "./src",
        "context": {
            "scan_type": "dependency",
            "severity_threshold": "medium"
        }
    }
    
    result = await qa_tester.execute.remote(context)
    assert result.status == InstructionStatus.COMPLETED
    assert "vulnerabilities" in result.data
    assert "risk_score" in result.data
    assert "recommendations" in result.data

@pytest.mark.asyncio
async def test_error_handling():
    """Test error handling across all instructions."""
    nlp = NLPProcessingInstruction.remote()
    mlops = MLOpsInstruction.remote()
    qa = QualityAssuranceInstruction.remote()
    
    # Test NLP with invalid task
    result = await nlp.execute.remote({"task": "invalid_task"})
    assert result.status == InstructionStatus.FAILED
    assert result.error is not None
    
    # Test MLOps with invalid model ID
    result = await mlops.execute.remote({
        "task": "deploy",
        "model_id": ""
    })
    assert result.status == InstructionStatus.FAILED
    assert result.error is not None
    
    # Test QA with invalid target
    result = await qa.execute.remote({
        "task": "run_tests",
        "target": "/nonexistent/path"
    })
    assert result.status == InstructionStatus.FAILED
    assert result.error is not None

@pytest.mark.asyncio
async def test_concurrent_execution():
    """Test concurrent execution of multiple instructions."""
    nlp = NLPProcessingInstruction.remote()
    mlops = MLOpsInstruction.remote()
    qa = QualityAssuranceInstruction.remote()
    
    tasks = [
        nlp.execute.remote({
            "text": "Test text",
            "task": "sentiment"
        }),
        mlops.execute.remote({
            "task": "monitor",
            "model_id": "test-model"
        }),
        qa.execute.remote({
            "task": "lint",
            "target": "./src"
        })
    ]
    
    results = await asyncio.gather(*tasks)
    assert all(r.status in [InstructionStatus.COMPLETED, InstructionStatus.FAILED] for r in results)

if __name__ == "__main__":
    pytest.main([__file__]) 