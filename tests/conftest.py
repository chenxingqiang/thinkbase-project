import pytest
from pathlib import Path
import json
import tempfile
import shutil
import os
import logging

@pytest.fixture
def test_data_dir() -> Path:
    """Get path to test data directory"""
    return Path(__file__).parent / 'tests' / 'data'

@pytest.fixture
def temp_dir():
    """Create temporary directory for test files"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir

@pytest.fixture
def test_config(test_data_dir):
    """Get test configuration file path"""
    return str(test_data_dir / 'config.json')

@pytest.fixture
def test_workflow(test_data_dir):
    """Get test workflow file path"""
    return str(test_data_dir / 'agent.json')

@pytest.fixture(autouse=True)
def setup_logging():
    """Setup logging for tests"""
    logging.basicConfig(level=logging.INFO)
    yield

@pytest.fixture
def agent_config():
    return {
        "AGENT": "Test_Agent",
        "CONTEXT": "Test context for unit testing",
        "ENVIRONMENT": {
            "INPUT": ["test_input_1", "test_input_2"],
            "OUTPUT": ["test_output_1", "test_output_2"]
        },
        "WORKFLOW": [
            {
                "step": 1,
                "title": "Test Step 1",
                "input": ["test_input_1"],
                "output": {
                    "type": "test",
                    "format": "text"
                }
            }
        ]
    }

@pytest.fixture
def workflow_config():
    return {
        "WORKFLOW": [
            {
                "step": 1,
                "title": "Test Step",
                "input": ["required_input"],
                "output": {"type": "test"}
            }
        ]
    }

@pytest.fixture
def temp_config_file(temp_dir, agent_config):
    """Create a temporary config file"""
    config_path = os.path.join(temp_dir, 'test_config.json')
    with open(config_path, 'w') as f:
        json.dump(agent_config, f)
    return config_path

@pytest.fixture
def mock_llm_response():
    """Mock LLM response for testing"""
    return {
        "response": "Test response",
        "metadata": {
            "model": "test-model",
            "tokens": 10
        }
    }

@pytest.fixture(scope="session", autouse=True)
def setup_ray_session():
    """Share one Ray cluster across the whole test session.

    Per-test ``ray.init()``/``ray.shutdown()`` costs ~10-13s each (the
    slowest single item in the suite by far); a single session-scoped
    cluster eliminates that entirely.  ``ignore_reinit_error=True`` keeps
    module-level ``ray.init`` calls harmless no-ops, and each module's
    teardown ``ray.shutdown()`` is idempotent, so the shared cluster is
    only torn down once at session end.

    The server spawned by tests/test_workflow_api.py is isolated with
    ``RAY_ADDRESS=local``, so it does not try to join this cluster.
    """
    import ray
    if not ray.is_initialized():
        ray.init(
            num_cpus=os.cpu_count() or 4,
            ignore_reinit_error=True,
            log_to_driver=False,
        )
    yield
    if ray.is_initialized():
        ray.shutdown()
