import pytest
import requests
import subprocess
import time
import os
import sys
import signal
import logging
import socket
import atexit
from requests.exceptions import HTTPError, ConnectionError
import json

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Find an available port
def find_free_port():
    """Find a free port for testing"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

class ServerManager:
    """Manage server startup and teardown for tests"""
    def __init__(self):
        self.port = find_free_port()
        self.process = None
        self.base_url = f"http://localhost:{self.port}"
        
    def start_server(self):
        """Start the workflow server in a separate process"""
        import subprocess
        import sys
        import time
        import os
        import traceback

        # Construct the command to run the server
        server_script = os.path.join(os.path.dirname(__file__), '..', 'agentflow', 'api', 'workflow_server.py')

        # Start the server process.  stdout/stderr go to a log file
        # rather than captured pipes: the server logs at DEBUG level
        # (ray/uvicorn startup), so a 64KB pipe would fill and block the
        # server mid-request, hanging tests that await responses.  A log
        # file keeps the server unblocked and the output debuggable.
        server_log = os.path.join(
            os.path.dirname(__file__), '..', 'logs', 'test_workflow_server.log'
        )
        os.makedirs(os.path.dirname(server_log), exist_ok=True)
        log_handle = open(server_log, 'a', buffering=1)
        self.log_handle = log_handle
        # Force the server subprocess to start its own Ray cluster
        # (RAY_ADDRESS=local) instead of auto-connecting to whatever
        # cluster the parent pytest process left running: connecting to
        # a stale GCS times out for 60s, pushing server startup past any
        # sane wait bound.
        server_env = dict(os.environ)
        server_env["RAY_ADDRESS"] = "local"
        self.process = subprocess.Popen(
            [sys.executable, server_script, str(self.port)],
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=server_env
        )

        # Wait for the server to start.  Ray/uvicorn startup is slow
        # under disk pressure (ray's file-system monitor warns past 95%
        # capacity and worker spawn slows), so keep generous headroom
        # before declaring failure.
        max_wait_time = 60
        start_time = time.time()

        while time.time() - start_time < max_wait_time:
            try:
                # Try to connect to the server
                response = requests.get(f"{self.base_url}/docs", timeout=2)
                if response.status_code == 200:
                    logger.info(f"Server started successfully on port {self.port}")
                    return
            except (requests.ConnectionError, requests.Timeout) as e:
                # Print out any connection errors
                logger.warning(f"Connection attempt failed: {e}")
                time.sleep(1)

        # Print out process output for debugging.  communicate() must be
        # bounded: the server process is likely still alive (slow start),
        # so an unbounded wait would hang the test forever.
        try:
            stdout, stderr = self.process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            stdout, stderr = self.process.communicate()
        logger.error(f"Server stdout: {stdout}")
        logger.error(f"Server stderr: {stderr}")

        # If we get here, server didn't start
        raise RuntimeError(f"Server failed to start within {max_wait_time} seconds")
    
    def stop_server(self):
        """Stop the server process"""
        if self.process:
            self.process.terminate()
            try:
                stdout, stderr = self.process.communicate(timeout=5)
                logger.debug(f"Server stdout: {stdout}")
                logger.debug(f"Server stderr: {stderr}")
            except subprocess.TimeoutExpired:
                self.process.kill()
            if getattr(self, 'log_handle', None):
                self.log_handle.close()
            self.process = None

@pytest.fixture(scope="module")
def server():
    """Pytest fixture to manage server lifecycle"""
    server_manager = ServerManager()
    try:
        server_manager.start_server()
        yield server_manager
    finally:
        server_manager.stop_server()

# Global variable to store the base URL
BASE_URL = None

@pytest.fixture(scope="session", autouse=True)
def configure_base_url(request):
    """Configure the base URL for tests"""
    global BASE_URL
    server_manager = ServerManager()
    BASE_URL = server_manager.base_url
    return BASE_URL

def test_sync_workflow_execution(server):
    """Test synchronous workflow execution"""
    url = f"{server.base_url}/workflow/execute"
    
    request_data = {
        "workflow": {
            "id": "test-workflow-1",
            "name": "Test Research Workflow",
            "steps": [
                {
                    "id": "step-1",
                    "type": "agent",
                    "name": "Research Step",
                    "description": "Perform research on the given topic",
                    "dependencies": [],
                    "required": True,
                    "optional": False,
                    "is_distributed": False,
                    "config": {
                        "strategy": "standard",
                        "params": {
                            "input": ["STUDENT_NEEDS", "LANGUAGE", "TEMPLATE"],
                            "output": {
                                "type": "research_findings",
                                "format": "structured_data"
                            }
                        },
                        "retry_delay": 1.0,
                        "retry_backoff": 2.0,
                        "max_retries": 3,
                        "timeout": 30.0
                    }
                }
            ]
        },
        "config": {
            "max_retries": 3,
            "retry_backoff": 2.0,
            "retry_delay": 0.1,
            "step_1_config": {
                "max_retries": 3,
                "timeout": 30,
                "preprocessors": [],
                "postprocessors": []
            },
            "execution": {
                "parallel": False,
                "max_retries": 3
            },
            "distributed": False,
            "timeout": 300,
            "logging_level": "INFO"
        },
        "input_data": {
            "STUDENT_NEEDS": {
                "RESEARCH_TOPIC": "API Testing in Distributed Systems",
                "DEADLINE": "2024-05-15",
                "ACADEMIC_LEVEL": "Master"
            },
            "LANGUAGE": {
                "TYPE": "English",
                "STYLE": "Academic"
            },
            "TEMPLATE": "Research Paper"
        }
    }
    
    response = requests.post(url, json=request_data, timeout=30)
    
    if response.status_code != 200:
        logger.error(f"Full error response: {response.text}")
        raise requests.exceptions.HTTPError(
            f"Request failed with status {response.status_code}: {response.text}"
        )
    
    result = response.json()
    assert result is not None
    assert "steps" in result
    assert isinstance(result["steps"], list)
    assert len(result["steps"]) > 0
    step = result["steps"][0]
    assert step["id"] == "step-1"
    assert step["status"] == "success"
    assert "result" in step
    assert isinstance(step["result"], dict)
    assert "content" in step["result"]

def test_async_workflow_execution(server):
    """Test asynchronous workflow execution"""
    execute_url = f"{server.base_url}/workflow/execute_async"
    
    request_data = {
        "workflow": {
            "id": "test-workflow-2",
            "name": "Test Async Research Workflow",
            "steps": [
                {
                    "id": "step-1",
                    "type": "agent",
                    "name": "Research Step",
                    "description": "Perform research on the given topic",
                    "dependencies": [],
                    "required": True,
                    "optional": False,
                    "is_distributed": True,
                    "config": {
                        "strategy": "standard",
                        "params": {
                            "input": ["STUDENT_NEEDS", "LANGUAGE", "TEMPLATE"],
                            "output": {
                                "type": "research_findings",
                                "format": "structured_data"
                            }
                        },
                        "retry_delay": 1.0,
                        "retry_backoff": 2.0,
                        "max_retries": 3,
                        "timeout": 30.0
                    }
                },
                {
                    "id": "step-2",
                    "type": "agent",
                    "name": "Document Generation Step",
                    "description": "Generate document from research findings",
                    "dependencies": ["step-1"],
                    "required": True,
                    "optional": False,
                    "is_distributed": True,
                    "config": {
                        "strategy": "standard",
                        "params": {
                            "input": ["WORKFLOW.1.output"],
                            "output": {
                                "type": "document",
                                "format": "Markdown with LaTeX"
                            }
                        },
                        "retry_delay": 1.0,
                        "retry_backoff": 2.0,
                        "max_retries": 3,
                        "timeout": 30.0
                    }
                }
            ]
        },
        "config": {
            "max_iterations": 3,
            "logging_level": "INFO",
            "distributed": True,
            "timeout": 300,
            "execution": {
                "parallel": True,
                "max_retries": 3
            },
            "agents": {
                "research": {
                    "provider": "openai",
                    "model": "gpt-3.5-turbo",
                    "temperature": 0.7
                },
                "document": {
                    "provider": "openai",
                    "model": "gpt-3.5-turbo",
                    "temperature": 0.7
                }
            }
        },
        "input_data": {
            "STUDENT_NEEDS": {
                "RESEARCH_TOPIC": "Distributed Computing Systems",
                "DEADLINE": "2024-05-15",
                "ACADEMIC_LEVEL": "Master"
            },
            "LANGUAGE": {
                "TYPE": "English",
                "STYLE": "Academic"
            },
            "TEMPLATE": "Research Paper"
        }
    }
    
    async_response = requests.post(execute_url, json=request_data, timeout=30)
    
    if async_response.status_code != 200:
        logger.error(f"Full async error response: {async_response.text}")
        raise requests.exceptions.HTTPError(
            f"Async request failed with status {async_response.status_code}: {async_response.text}"
        )
    
    result = async_response.json()
    assert result["status"] == "pending"
    assert result["task_id"] is not None

def test_invalid_workflow(server):
    """Test handling of invalid workflow configuration"""
    url = f"{server.base_url}/workflow/execute"

    invalid_workflow_config = {
        "workflow": {
            # Intentionally missing required fields and empty steps
            "steps": []
        },
        "input_data": {}
    }

    try:
        response = requests.post(url, json=invalid_workflow_config, timeout=30)
        logger.debug(f"Invalid Workflow Response Status: {response.status_code}")
        logger.debug(f"Invalid Workflow Response Content: {response.text}")

        # Verify error response
        assert response.status_code == 422  # Validation error status code
        error_data = response.json()
        assert "detail" in error_data
        
        # Check for empty steps error
        error_details = error_data.get("detail", [])
        assert any("Workflow must have at least one step" in str(detail) for detail in error_details), \
            f"No 'Workflow must have at least one step' error found in {error_details}"
    except Exception as e:
        logger.error(f"Invalid workflow request failed: {e}")
        raise

if __name__ == "__main__":
    pytest.main([__file__])
