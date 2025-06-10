"""Test cases for the transformer module."""

import pytest
from unittest.mock import MagicMock, patch
from .transformer import AgenticTransformer
from .instructions.response_parsing import ResponseParsingInstruction, ContentType, ContentBlock
import asyncio

@pytest.fixture
def mock_response_parser():
    """Create a mock response parser."""
    parser = MagicMock(spec=ResponseParsingInstruction)
    # Make execute return a coroutine
    async def mock_execute(*args, **kwargs):
        return parser.execute.return_value
    parser.execute = MagicMock(side_effect=mock_execute)
    return parser

@pytest.fixture
def transformer(mock_response_parser):
    """Create a transformer instance for testing."""
    with patch('agentflow.core.transformer.ResponseParsingInstruction', return_value=mock_response_parser):
        return AgenticTransformer()

def test_process_text_empty(transformer):
    """Test processing empty text."""
    assert transformer.process_text("") == ""

def test_process_text_code(transformer, mock_response_parser):
    """Test processing code text."""
    code = "def test():\n    return 42"
    mock_response_parser.execute.return_value = [
        ContentBlock(type=ContentType.CODE, content=code, metadata={})
    ]
    processed = transformer.process_text(code)
    assert "def test" in processed
    assert "return 42" in processed

def test_process_text_natural_language(transformer, mock_response_parser):
    """Test processing natural language text."""
    text = "This is a test."
    mock_response_parser.execute.return_value = [
        ContentBlock(type=ContentType.TEXT, content=text, metadata={})
    ]
    processed = transformer.process_text(text)
    assert processed == text.strip()

def test_process_text_mixed(transformer, mock_response_parser):
    """Test processing mixed text."""
    mixed = "Text\ndef test():\n    pass"
    mock_response_parser.execute.return_value = [
        ContentBlock(type=ContentType.TEXT, content="Text", metadata={}),
        ContentBlock(type=ContentType.CODE, content="def test():\n    pass", metadata={})
    ]
    processed = transformer.process_text(mixed)
    assert "Text" in processed
    assert "def test" in processed

def test_process_text_preserves_symbols(transformer, mock_response_parser):
    """Test that processing preserves important symbols."""
    code = "x = 1 + 2 * 3"
    mock_response_parser.execute.return_value = [
        ContentBlock(type=ContentType.CODE, content=code, metadata={})
    ]
    processed = transformer.process_text(code)
    assert "+" in processed
    assert "*" in processed

def test_process_text_multiline(transformer, mock_response_parser):
    """Test processing multiline text."""
    text = """First line
    Second line
    Third line
    x = 5
    y = x + 1"""
    mock_response_parser.execute.return_value = [
        ContentBlock(type=ContentType.TEXT, content=text, metadata={})
    ]
    processed = transformer.process_text(text)
    assert "First line" in processed
    assert "Second line" in processed
    assert "Third line" in processed
    assert "x = 5" in processed
    assert "y = x + 1" in processed

@pytest.mark.asyncio
async def test_extract_patterns_code(transformer, mock_response_parser):
    """Test extracting code patterns."""
    code = """```python
    def test_function(x: int) -> int:
        '''Test docstring.'''
        return x + 1
    ```"""
    mock_response_parser.execute.return_value = [
        ContentBlock(type=ContentType.CODE, content=code, metadata={})
    ]
    patterns = await transformer.extract_patterns(code)
    assert any(p["type"] == "code" for p in patterns)
    assert any("def" in p["text"] for p in patterns)
    assert any("return" in p["text"] for p in patterns)

@pytest.mark.asyncio
async def test_extract_patterns_documentation(transformer, mock_response_parser):
    """Test extracting documentation patterns."""
    doc = '''```python
    def test():
        """This is a test docstring."""
        pass
    ```'''
    mock_response_parser.execute.return_value = [
        ContentBlock(type=ContentType.CODE, content=doc, metadata={})
    ]
    patterns = await transformer.extract_patterns(doc)
    assert any(p["type"] == "code" for p in patterns)
    assert any("docstring" in p["text"] for p in patterns)

@pytest.mark.asyncio
async def test_extract_patterns_structure(transformer, mock_response_parser):
    """Test extracting structure patterns."""
    code = """```python
    x = [1, 2, 3]
    y = {'a': 1}
    ```"""
    mock_response_parser.execute.return_value = [
        ContentBlock(type=ContentType.CODE, content=code, metadata={})
    ]
    patterns = await transformer.extract_patterns(code)
    assert any(p["type"] == "structure" for p in patterns)
    assert any("[" in p["text"] for p in patterns)
    assert any("{" in p["text"] for p in patterns)

@pytest.mark.asyncio
async def test_extract_patterns_operation(transformer, mock_response_parser):
    """Test extracting operation patterns."""
    code = """```python
    x = a + b
    y *= 2
    ```"""
    mock_response_parser.execute.return_value = [
        ContentBlock(type=ContentType.CODE, content=code, metadata={})
    ]
    patterns = await transformer.extract_patterns(code)
    assert any(p["type"] == "operation" for p in patterns)
    assert any("+" in p["text"] for p in patterns)
    assert any("*" in p["text"] for p in patterns)

@pytest.mark.asyncio
async def test_extract_patterns_empty(transformer, mock_response_parser):
    """Test extracting patterns from empty text."""
    mock_response_parser.execute.return_value = []
    patterns = await transformer.extract_patterns("")
    assert patterns == []

@pytest.mark.asyncio
async def test_extract_patterns_whitespace(transformer, mock_response_parser):
    """Test extracting patterns from whitespace."""
    mock_response_parser.execute.return_value = []
    patterns = await transformer.extract_patterns("   \n   ")
    assert patterns == []

@pytest.mark.asyncio
async def test_extract_patterns_complex(transformer, mock_response_parser):
    """Test extracting patterns from complex input."""
    code = """```python
    def calculate_sum(numbers: List[int]) -> int:
        '''Calculate the sum of a list of numbers.
        
        Args:
            numbers: List of numbers to sum
            
        Returns:
            Sum of the numbers
        '''
        if not numbers:
            return 0
        return sum(numbers)
    ```"""
    mock_response_parser.execute.return_value = [
        ContentBlock(type=ContentType.CODE, content=code, metadata={})
    ]
    patterns = await transformer.extract_patterns(code)
    
    # Check for code patterns
    assert any(p["type"] == "code" and "def" in p["text"] for p in patterns)
    assert any(p["type"] == "code" and "return" in p["text"] for p in patterns)
    assert any(p["type"] == "code" and "if" in p["text"] for p in patterns)
    
    # Check for structure patterns
    assert any(p["type"] == "structure" and "(" in p["text"] for p in patterns)
    assert any(p["type"] == "structure" and ")" in p["text"] for p in patterns)
    assert any(p["type"] == "structure" and "[" in p["text"] for p in patterns)
    assert any(p["type"] == "structure" and "]" in p["text"] for p in patterns)

if __name__ == '__main__':
    pytest.main(['-v'])
