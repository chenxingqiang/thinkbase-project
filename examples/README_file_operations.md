# File Operations Example

This example demonstrates comprehensive file operations using the AgentFlow framework. It showcases various file manipulation capabilities including creation, reading, updating, deletion, searching, and advanced editing operations.

## Features

1. **Basic File Operations**
   - Create files with content
   - Read file contents
   - Update file contents
   - Delete files
   - Search files by pattern
   - Get file metadata

2. **Advanced Edit Operations**
   - Insert text at specific positions
   - Delete lines
   - Change text case
   - Format text
   - Auto-indent code
   - Sort lines
   - Join lines
   - Block editing
   - Text analysis

3. **Error Handling**
   - Comprehensive error reporting
   - Operation status tracking
   - Detailed error messages
   - Metrics collection

## Usage

### Basic Operations

```python
from examples.file_operations_example import FileOperationsDemo

# Initialize with workspace directory
demo = FileOperationsDemo("./workspace")

# Create a file
result = await demo.create_file("example.txt", "Hello, World!")
print(f"Create result: {result.message}")

# Read file content
result = await demo.read_file("example.txt")
if result.success:
    print(f"Content: {result.data['content']}")

# Update file
result = await demo.update_file("example.txt", "Updated content")
print(f"Update result: {result.message}")

# Delete file
result = await demo.delete_file("example.txt")
print(f"Delete result: {result.message}")
```

### Advanced Editing

```python
# Perform multiple edit operations
edit_operations = [
    {
        "type": "insert_text",
        "position": 1,
        "text": "New Line\n"
    },
    {
        "type": "delete_lines",
        "start": 3,
        "end": 3
    },
    {
        "type": "change_case",
        "start": 1,
        "end": 2,
        "mode": "upper"
    }
]

result = await demo.edit_file_content("example.txt", edit_operations)
print(f"Edit result: {result.message}")
```

### File Search

```python
# Search for files matching pattern
result = await demo.search_files("*.txt")
if result.success:
    print("Found files:", result.data["matches"])
```

### Metadata Retrieval

```python
# Get file metadata
result = await demo.get_file_metadata("example.txt")
if result.success:
    print("Metadata:", result.data)
```

## Testing

The example includes a comprehensive test suite that verifies all functionality:

```bash
# Run tests
pytest tests/test_file_operations.py -v
```

## Error Handling

All operations return a `FileOperationResult` object containing:
- `success`: Boolean indicating operation success
- `message`: Descriptive message about the operation
- `data`: Operation-specific data (if any)
- `metrics`: Performance metrics (if available)

Example error handling:

```python
result = await demo.read_file("nonexistent.txt")
if not result.success:
    print(f"Error: {result.message}")
```

## Project Structure

```
examples/
├── file_operations_example.py  # Main implementation
├── README_file_operations.md   # This documentation
└── test_workspace/            # Default workspace directory
    └── ...                    # Created files

tests/
└── test_file_operations.py    # Test suite
```

## Dependencies

- Python 3.8+
- asyncio
- pathlib
- AgentFlow framework

## Contributing

Feel free to contribute by:
1. Adding new file operations
2. Improving error handling
3. Adding more test cases
4. Enhancing documentation
5. Optimizing performance

## License

This example is part of the AgentFlow project and is licensed under the same terms. 