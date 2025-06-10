"""Tests for advanced instruction functionality."""
import pytest
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from agentflow.core.instructions.base import (
    InstructionStatus, 
    InstructionResult, 
    InstructionMetrics, 
    BaseInstruction, 
    ValidationResult,
    CompositeInstruction,
    ParallelInstruction
)
import asyncio
import time
import logging
from pathlib import Path
import tempfile
import os

# Set up logging
logger = logging.getLogger(__name__)

# Test fixtures
@pytest.fixture
def temp_dir():
    """Create a temporary directory for file operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def sample_file(temp_dir):
    """Create a sample file for testing."""
    file_path = temp_dir / "test.txt"
    with open(file_path, "w") as f:
        f.write("test content")
    return file_path

@pytest.fixture
def sample_params():
    """Create sample instruction parameters."""
    return {
        "name": "test_instruction",
        "description": "Test instruction",
        "timeout": 10,
        "retry_count": 3,
        "validation_rules": []
    }

@pytest.fixture
def sample_context():
    """Create sample execution context."""
    return {
        "input_data": {"key": "value"},
        "params": {"param1": 1, "param2": "test"},
        "metadata": {"timestamp": time.time()}
    }

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return pd.DataFrame({
        "feature1": np.random.rand(100),
        "feature2": np.random.rand(100),
        "target": np.random.randint(0, 2, 100)
    })

# Test implementations
class SimpleInstruction(BaseInstruction):
    """Simple test instruction implementation."""
    
    def __init__(self, name: str, description: str, **kwargs):
        super().__init__(name, description)
        self.validation_rules = []
        if 'timeout' in kwargs:
            self.timeout = kwargs['timeout']
        if 'retry_count' in kwargs:
            self.retry_count = kwargs['retry_count']

    def add_validation_rule(self, rule: callable):
        """Add validation rule."""
        self.validation_rules.append(rule)

    async def _validate(self, context: Dict[str, Any]) -> ValidationResult:
        """Validate context against rules."""
        for rule in self.validation_rules:
            result = rule(context) if not asyncio.iscoroutinefunction(rule) else await rule(context)
            if not result:
                return ValidationResult(is_valid=False, message="Validation failed")
        return ValidationResult(is_valid=True, message="Validation passed")

    async def _execute_impl(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute instruction implementation."""
        await asyncio.sleep(0.1)  # Simulate work
        return InstructionResult(
            status="completed",
            data={"result": "success", "data": context},
            metrics={}
        )

class SimpleCompositeInstruction(CompositeInstruction):
    """Simple test composite instruction implementation."""
    
    def __init__(self, name: str, description: str, **kwargs):
        super().__init__(name, description)
        if 'timeout' in kwargs:
            self.timeout = kwargs['timeout']

    async def _execute_impl(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute all instructions in sequence."""
        results = []
        for instruction in self.instructions:
            result = await instruction.execute(context)
            results.append(result)
        return InstructionResult(
            status="completed",
            data=self._combine_results(results),
            metrics={}
        )

class SimpleParallelInstruction(ParallelInstruction):
    """Simple test parallel instruction implementation."""
    
    def __init__(self, name: str, description: str, **kwargs):
        super().__init__(name, description)
        if 'timeout' in kwargs:
            self.timeout = kwargs['timeout']

    async def _execute_impl(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute all instructions in parallel."""
        tasks = [instruction.execute(context) for instruction in self.instructions]
        results = await asyncio.gather(*tasks)
        return InstructionResult(
            status="completed",
            data=self._combine_results(results),
            metrics={}
        )

class DataProcessingInstruction(BaseInstruction):
    """Simple test data processing instruction."""
    
    def __init__(self, name: str, description: str, **kwargs):
        super().__init__(name, description)
        if 'timeout' in kwargs:
            self.timeout = kwargs['timeout']

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data."""
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        return data.fillna(0)

    def reduce_dimensions(self, data: pd.DataFrame, n_components: int = 2) -> pd.DataFrame:
        """Reduce data dimensions."""
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components)
        return pd.DataFrame(pca.fit_transform(data))

    def cluster_data(self, data: pd.DataFrame, n_clusters: int = 3) -> pd.DataFrame:
        """Cluster data."""
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters)
        data['cluster'] = kmeans.fit_predict(data)
        return data

    async def _execute_impl(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data processing pipeline."""
        try:
            data = context.get('data')
            if not isinstance(data, pd.DataFrame):
                raise ValueError("Input must be a pandas DataFrame")
                
            processed_data = self.preprocess(data)
            reduced_data = self.reduce_dimensions(processed_data)
            clustered_data = self.cluster_data(reduced_data)
            
            return InstructionResult(
                status="completed",
                data={
                    "data": clustered_data.to_dict(),
                    "metrics": {
                        "rows": len(clustered_data),
                        "columns": len(clustered_data.columns)
                    }
                },
                metrics={}
            )
        except Exception as e:
            logger.error(f"Data processing failed: {str(e)}")
            return InstructionResult(
                status="failed",
                data={"error": str(e)},
                metrics={}
            )

# Test classes
@pytest.mark.asyncio
class TestAdvancedInstruction:
    """Test advanced instruction functionality."""
    
    async def test_initialization(self, sample_params):
        """Test instruction initialization."""
        instruction = SimpleInstruction(**sample_params)
        assert instruction.name == sample_params["name"]
        assert instruction.description == sample_params["description"]
        assert instruction.status == "pending"
        
    async def test_validation(self, sample_params):
        """Test parameter validation."""
        instruction = SimpleInstruction(**sample_params)
        
        # Test valid case
        async def validation_rule(ctx):
            return True
        
        instruction.add_validation_rule(validation_rule)
        result = await instruction._validate({"test": "data"})
        assert result.is_valid
        
        # Test invalid case
        async def failing_rule(ctx):
            return False
            
        instruction.add_validation_rule(failing_rule)
        result = await instruction._validate({"test": "data"})
        assert not result.is_valid
        
    async def test_execution(self, sample_params, sample_context):
        """Test instruction execution."""
        instruction = SimpleInstruction(**sample_params)
        result = await instruction.execute(sample_context)
        assert isinstance(result, InstructionResult)
        assert result.status == "completed"
        assert "result" in result.data
        assert result.data["result"] == "success"
        assert instruction.metrics is not None
        assert instruction.metrics.execution_time > 0

@pytest.mark.asyncio
class TestCompositeInstruction:
    """Test composite instruction functionality."""
    
    async def test_composition(self, sample_params):
        """Test instruction composition."""
        composite = SimpleCompositeInstruction(**sample_params)
        sub_instruction1 = SimpleInstruction("sub1", "Sub instruction 1")
        sub_instruction2 = SimpleInstruction("sub2", "Sub instruction 2")
        
        composite.add_instruction(sub_instruction1)
        composite.add_instruction(sub_instruction2)
        
        assert len(composite.instructions) == 2
        
    async def test_execution_order(self, sample_params, sample_context):
        """Test execution order of composite instructions."""
        composite = SimpleCompositeInstruction(**sample_params)
        execution_order = []
        
        class OrderedInstruction(SimpleInstruction):
            async def _execute_impl(self, context):
                execution_order.append(self.name)
                await asyncio.sleep(0.1)
                return InstructionResult(
                    status="completed",
                    data={"result": "success", "name": self.name},
                    metrics={}
                )
        
        inst1 = OrderedInstruction("first", "First instruction")
        inst2 = OrderedInstruction("second", "Second instruction")
        
        composite.add_instruction(inst1)
        composite.add_instruction(inst2)
        
        result = await composite.execute(sample_context)
        assert execution_order == ["first", "second"]
        assert isinstance(result, InstructionResult)
        assert result.status == "completed"
        assert len(result.data["results"]) == 2

@pytest.mark.asyncio
class TestParallelInstruction:
    """Test parallel instruction functionality."""
    
    async def test_parallel_execution(self, sample_params, sample_context):
        """Test parallel execution of instructions."""
        parallel = SimpleParallelInstruction(**sample_params)
        execution_times = []
        
        class TimedInstruction(SimpleInstruction):
            async def _execute_impl(self, context):
                start = time.time()
                await asyncio.sleep(0.1)
                execution_times.append(time.time() - start)
                return InstructionResult(
                    status="completed",
                    data={"result": "success", "execution_time": execution_times[-1]},
                    metrics={}
                )
        
        for i in range(3):
            parallel.add_instruction(TimedInstruction(f"timed_{i}", f"Timed instruction {i}"))
        
        start = time.time()
        result = await parallel.execute(sample_context)
        total_time = time.time() - start
        
        # Verify parallel execution
        assert total_time < sum(execution_times)
        assert isinstance(result, InstructionResult)
        assert result.status == "completed"
        assert len(result.data["results"]) == 3

@pytest.mark.asyncio
class TestDataProcessingInstruction:
    """Test data processing instruction functionality."""
    
    async def test_preprocessing(self, sample_data):
        """Test data preprocessing."""
        instruction = DataProcessingInstruction("test", "Test preprocessing")
        processed_data = instruction.preprocess(sample_data)
        assert isinstance(processed_data, pd.DataFrame)
        assert not processed_data.isnull().any().any()
        
    async def test_dimension_reduction(self, sample_data):
        """Test dimension reduction."""
        instruction = DataProcessingInstruction("test", "Test dimension reduction")
        reduced_data = instruction.reduce_dimensions(sample_data, n_components=2)
        assert isinstance(reduced_data, pd.DataFrame)
        assert reduced_data.shape[1] == 2
        
    @pytest.mark.parametrize("n_clusters", [2, 3, 4])
    async def test_clustering(self, sample_data, n_clusters):
        """Test data clustering with different numbers of clusters."""
        instruction = DataProcessingInstruction("test", "Test clustering")
        clusters = instruction.cluster_data(sample_data, n_clusters=n_clusters)
        assert len(np.unique(clusters['cluster'])) == n_clusters
        
    async def test_execution(self, sample_data):
        """Test full execution pipeline."""
        instruction = DataProcessingInstruction("test", "Test execution")
        result = await instruction.execute({
            "data": sample_data,
            "params": {
                "n_clusters": 3
            }
        })
        assert isinstance(result, InstructionResult)
        assert result.status == "completed"
        assert "data" in result.data
        assert "metrics" in result.data
        assert instruction.metrics is not None
        assert instruction.metrics.execution_time > 0
        
    async def test_error_handling(self):
        """Test error handling with invalid input."""
        instruction = DataProcessingInstruction("test", "Test error handling")
        with pytest.raises(ValueError):
            instruction.preprocess({"invalid": "data"})
            
    @pytest.mark.parametrize("data_size", [10, 100, 1000])
    async def test_optimization(self, data_size):
        """Test optimization with different data sizes."""
        data = pd.DataFrame(np.random.rand(data_size, 5))
        instruction = DataProcessingInstruction("test", "Test optimization")
        result = await instruction.execute({"data": data})
        assert result.status == "completed"
        assert instruction.metrics is not None
        assert instruction.metrics.execution_time > 0
