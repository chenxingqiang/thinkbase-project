"""Shared fixtures for instruction tests."""

import pytest
import os
import tempfile
import shutil
from typing import Generator

@pytest.fixture
def test_dir() -> Generator[str, None, None]:
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def test_file(test_dir: str) -> Generator[str, None, None]:
    """Create a test file with sample content."""
    file_path = os.path.join(test_dir, "test.txt")
    with open(file_path, "w") as f:
        f.write("Line 1\nLine 2\nLine 3\nLine 4\nLine 5\n")
    yield file_path
    if os.path.exists(file_path):
        os.unlink(file_path) 