"""Instructions for parsing LLM responses and recognizing content types."""

from typing import Dict, Any, List, Optional, Union, Pattern
from dataclasses import dataclass, field
from enum import Enum
import re
import json
import base64
from pathlib import Path
import logging
from xml.etree.ElementTree import Element

from agentflow.core.instructions.base import BaseInstruction, InstructionResult, InstructionStatus

logger = logging.getLogger(__name__)

class ContentType(Enum):
    """Types of content in responses."""
    TEXT = "text"
    CODE = "code"
    IMAGE = "image"
    JSON = "json"
    MARKDOWN = "markdown"
    TABLE = "table"
    MIXED = "mixed"
    XML = "xml"
    CSV = "csv"
    YAML = "yaml"
    MATH = "math"
    DIAGRAM = "diagram"
    PSEUDOCODE = "pseudocode"

@dataclass
class ContentBlock:
    """Block of content with type information."""
    type: ContentType
    content: str
    metadata: Dict[str, Any]
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.type.value,
            "content": self.content,
            "metadata": self.metadata,
            "line_start": self.line_start,
            "line_end": self.line_end
        }

@dataclass
class ResponseMetrics:
    """Metrics for response parsing."""
    total_blocks: int = 0
    blocks_by_type: Dict[str, int] = field(default_factory=dict)
    parsing_time: float = 0.0
    total_lines: int = 0
    error_count: int = 0
    
    def __post_init__(self):
        """Initialize blocks_by_type if None."""
        if self.blocks_by_type is None:
            self.blocks_by_type = {}

class ResponseParsingInstruction(BaseInstruction):
    """Instruction for parsing LLM responses."""
    
    def __init__(self):
        super().__init__(
            name="response_parsing",
            description="Parse and analyze LLM responses"
        )
        self._metrics = ResponseMetrics()
        self._patterns = {
            "code": re.compile(r"```(\w+)?\n(.*?)\n```", re.DOTALL),
            "image": re.compile(r"!\[([^\]]*)\]\(([^)]+)\)"),
            "table": re.compile(r"\|.*\|[\s\S]*?\|.*\|"),
            "json": re.compile(r"\{[\s\S]*?\}|\[[\s\S]*?\]"),
            "xml": re.compile(r"<\?xml.*?\?>.*?</\w+>|<\w+>.*?</\w+>", re.DOTALL),
            "csv": re.compile(r"(?:\w+,)+\w+(?:\n(?:\w+,)+\w+)*"),
            "yaml": re.compile(r"---\n[\s\S]*?(?:\n\.{3}|\Z)"),
            "math": re.compile(r"\$\$(.*?)\$\$|\$(.*?)\$", re.DOTALL),
            "diagram": re.compile(r"```(?:mermaid|plantuml|dot)\n(.*?)\n```", re.DOTALL),
            "pseudocode": re.compile(r"```pseudocode\n(.*?)\n```", re.DOTALL)
        }
        
    async def execute(self, context: Dict[str, Any]) -> InstructionResult:
        """Execute the instruction."""
        return await self._execute_impl(context)

    async def _execute_impl(self, context: Dict[str, Any]) -> InstructionResult:
        """Execute response parsing."""
        try:
            # Get response text
            response = context.get("response")
            if not response:
                return InstructionResult(
                    status=InstructionStatus.FAILED,
                    error="No response provided"
                )
            
            # Parse content blocks
            blocks = self._parse_content_blocks(response)
            
            # Process each block based on type
            processed_blocks = []
            for block in blocks:
                processed_block = await self._process_block(block)
                processed_blocks.append(processed_block)
            
            return InstructionResult(
                status=InstructionStatus.COMPLETED,
                data={
                    "blocks": [block.to_dict() for block in processed_blocks],
                    "metrics": self._metrics.__dict__
                }
            )
            
        except Exception as e:
            logger.error(f"Error in response parsing: {str(e)}")
            return InstructionResult(
                status=InstructionStatus.FAILED,
                error=str(e)
            )
    
    def _parse_content_blocks(self, response: str) -> List[ContentBlock]:
        """Parse response into content blocks."""
        blocks = []
        current_pos = 0
        
        # Find all content blocks
        while current_pos < len(response):
            block = self._find_next_block(response, current_pos)
            if block:
                blocks.append(block)
                current_pos = block.line_end or (current_pos + 1)
            else:
                # Handle remaining text
                remaining = response[current_pos:].strip()
                if remaining:
                    blocks.append(ContentBlock(
                        type=ContentType.TEXT,
                        content=remaining,
                        metadata={},
                        line_start=current_pos,
                        line_end=len(response)
                    ))
                break
        
        # Update metrics
        self._metrics.total_blocks = len(blocks)
        for block in blocks:
            self._metrics.blocks_by_type[block.type.value] = \
                self._metrics.blocks_by_type.get(block.type.value, 0) + 1
        
        return blocks
    
    def _find_next_block(self, text: str, start_pos: int) -> Optional[ContentBlock]:
        """Find next content block in text."""
        # Try each pattern
        best_match = None
        best_start = float('inf')
        
        for content_type, pattern in self._patterns.items():
            match = pattern.search(text, start_pos)
            if match and match.start() < best_start:
                best_match = (content_type, match)
                best_start = match.start()
        
        if best_match:
            content_type, match = best_match
            return self._create_block_from_match(content_type, match)
        
        return None
    
    def _create_block_from_match(
        self,
        content_type: str,
        match: re.Match
    ) -> ContentBlock:
        """Create content block from regex match."""
        if content_type == "code":
            language = match.group(1) or "text"
            content = match.group(2)
            metadata = {"language": language}
        elif content_type == "image":
            alt_text = match.group(1)
            url = match.group(2)
            content = match.group(0)
            metadata = {"alt_text": alt_text, "url": url}
        else:
            content = match.group(0)
            metadata = {}
        
        return ContentBlock(
            type=ContentType(content_type),
            content=content,
            metadata=metadata,
            line_start=match.start(),
            line_end=match.end()
        )
    
    async def _process_block(self, block: ContentBlock) -> ContentBlock:
        """Process content block based on type."""
        if block.type == ContentType.CODE:
            return await self._process_code_block(block)
        elif block.type == ContentType.IMAGE:
            return await self._process_image_block(block)
        elif block.type == ContentType.JSON:
            return await self._process_json_block(block)
        elif block.type == ContentType.TABLE:
            return await self._process_table_block(block)
        elif block.type == ContentType.XML:
            return await self._process_xml_block(block)
        elif block.type == ContentType.CSV:
            return await self._process_csv_block(block)
        elif block.type == ContentType.YAML:
            return await self._process_yaml_block(block)
        elif block.type == ContentType.MATH:
            return await self._process_math_block(block)
        elif block.type == ContentType.DIAGRAM:
            return await self._process_diagram_block(block)
        elif block.type == ContentType.PSEUDOCODE:
            return await self._process_pseudocode_block(block)
        else:
            return block
    
    async def _process_code_block(self, block: ContentBlock) -> ContentBlock:
        """Process code block."""
        try:
            # Add code analysis metadata
            language = block.metadata.get("language", "text")
            lines = block.content.count('\n') + 1
            block.metadata.update({
                "line_count": lines,
                "char_count": len(block.content),
                "language": language,
                "complexity": self._calculate_complexity(block.content, language),
                "dependencies": self._extract_dependencies(block.content, language),
                "documentation_ratio": self._calculate_doc_ratio(block.content, language),
                "patterns": self._detect_code_patterns(block.content, language)
            })
            
            # Enhanced code structure analysis
            if language in ["python", "javascript", "java"]:
                block.metadata["functions"] = self._extract_functions(block.content, language)
                block.metadata["classes"] = self._extract_classes(block.content, language)
                block.metadata["imports"] = self._extract_imports(block.content, language)
                block.metadata["variables"] = self._extract_variables(block.content, language)
            
            return block
        except Exception as e:
            logger.error(f"Error processing code block: {e}")
            self._metrics.error_count += 1
            return block
    
    async def _process_image_block(self, block: ContentBlock) -> ContentBlock:
        """Process image block."""
        try:
            url = block.metadata.get("url")
            if url and url.startswith("data:image"):
                # Handle base64 encoded images
                block.metadata["encoding"] = "base64"
                block.metadata["mime_type"] = url.split(";")[0].split(":")[1]
            else:
                block.metadata["encoding"] = "url"
            
            return block
        except Exception as e:
            logger.error(f"Error processing image block: {e}")
            self._metrics.error_count += 1
            return block
    
    async def _process_json_block(self, block: ContentBlock) -> ContentBlock:
        """Process JSON block."""
        try:
            # Parse JSON and add structure info
            data = json.loads(block.content)
            block.metadata["structure"] = self._analyze_json_structure(data)
            return block
        except Exception as e:
            logger.error(f"Error processing JSON block: {e}")
            self._metrics.error_count += 1
            return block
    
    async def _process_table_block(self, block: ContentBlock) -> ContentBlock:
        """Process table block."""
        try:
            # Parse markdown table
            lines = block.content.strip().split('\n')
            headers = [col.strip() for col in lines[0].split('|')[1:-1]]
            block.metadata["headers"] = headers
            block.metadata["row_count"] = len(lines) - 2  # Exclude header and separator
            return block
        except Exception as e:
            logger.error(f"Error processing table block: {e}")
            self._metrics.error_count += 1
            return block
    
    def _calculate_complexity(self, code: str, language: str) -> Dict[str, Any]:
        """Calculate code complexity metrics."""
        metrics = {
            "cyclomatic": 0,
            "cognitive": 0,
            "halstead": {
                "vocabulary": 0,
                "length": 0,
                "difficulty": 0
            }
        }
        
        # Basic cyclomatic complexity
        control_structures = ["if", "for", "while", "case", "catch", "&&", "||"]
        metrics["cyclomatic"] = sum(code.count(kw) for kw in control_structures)
        
        # Basic cognitive complexity
        nesting_level = 0
        for line in code.split('\n'):
            if any(kw in line for kw in ["if", "for", "while", "def", "class"]):
                nesting_level += 1
            metrics["cognitive"] += nesting_level
        
        return metrics

    def _extract_dependencies(self, code: str, language: str) -> List[str]:
        """Extract code dependencies."""
        dependencies = []
        
        if language == "python":
            # Match import statements
            import_pattern = r"(?:from\s+(\S+)\s+import|\bimport\s+([^;\n]+))"
            for match in re.finditer(import_pattern, code):
                dep = match.group(1) or match.group(2)
                dependencies.extend(d.strip() for d in dep.split(","))
        elif language == "javascript":
            # Match require and import statements
            js_pattern = r"(?:require\(['\"]([^'\"]+)['\"]\)|import.*?from\s+['\"]([^'\"]+)['\"])"
            for match in re.finditer(js_pattern, code):
                dep = match.group(1) or match.group(2)
                dependencies.append(dep)
                
        return list(set(dependencies))

    def _calculate_doc_ratio(self, code: str, language: str) -> float:
        """Calculate documentation to code ratio."""
        doc_lines = 0
        code_lines = 0
        
        for line in code.split('\n'):
            line = line.strip()
            if line.startswith(('#', '//', '/*', '*', '"""', "'''")):
                doc_lines += 1
            elif line:
                code_lines += 1
                
        return doc_lines / max(code_lines, 1)

    def _detect_code_patterns(self, code: str, language: str) -> List[str]:
        """Detect common code patterns."""
        patterns = []
        
        # Design patterns
        if re.search(r"class\s+\w+Factory", code):
            patterns.append("factory_pattern")
        if re.search(r"class\s+\w+Singleton", code):
            patterns.append("singleton_pattern")
        if re.search(r"class\s+\w+Observer|subscribe|notify", code):
            patterns.append("observer_pattern")
            
        # Architecture patterns
        if re.search(r"class\s+\w+Controller", code):
            patterns.append("mvc_pattern")
        if re.search(r"class\s+\w+Service", code):
            patterns.append("service_pattern")
            
        return patterns

    async def _process_xml_block(self, block: ContentBlock) -> ContentBlock:
        """Process XML block."""
        try:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(block.content)
            
            # Extract namespaces from the XML string directly
            ns_pattern = r'xmlns:(\w+)=["\']([^"\']+)["\']'
            namespaces = dict(re.findall(ns_pattern, block.content))
            
            block.metadata.update({
                "root_tag": root.tag,
                "namespaces": list(namespaces.keys()),
                "child_elements": len(list(root)),
                "attributes": len(root.attrib),
                "structure": self._analyze_xml_structure(root)
            })
            
            return block
        except Exception as e:
            logger.error(f"Error processing XML block: {e}")
            self._metrics.error_count += 1
            return block

    async def _process_csv_block(self, block: ContentBlock) -> ContentBlock:
        """Process CSV block."""
        try:
            import csv
            from io import StringIO
            
            # Parse CSV content
            f = StringIO(block.content)
            reader = csv.reader(f)
            rows = list(reader)
            
            if rows:
                headers = rows[0]
                data_rows = rows[1:]
                
                # Analyze column types
                column_types = []
                for col_idx in range(len(headers)):
                    col_values = [row[col_idx] for row in data_rows if len(row) > col_idx]
                    col_type = self._infer_column_type(col_values)
                    column_types.append(col_type)
                
                block.metadata.update({
                    "headers": headers,
                    "row_count": len(data_rows),
                    "column_count": len(headers),
                    "column_types": dict(zip(headers, column_types)),
                    "statistics": self._calculate_csv_statistics(data_rows, headers, column_types)
                })
            
            return block
        except Exception as e:
            logger.error(f"Error processing CSV block: {e}")
            self._metrics.error_count += 1
            return block

    def _infer_column_type(self, values: List[str]) -> str:
        """Infer data type of a column."""
        try:
            # Try numeric
            if all(self._is_numeric(v) for v in values if v):
                if any('.' in v for v in values if v):
                    return "float"
                return "integer"
            # Try date
            elif all(self._is_date(v) for v in values if v):
                return "date"
            # Default to string
            return "string"
        except:
            return "string"

    def _calculate_csv_statistics(
        self,
        data_rows: List[List[str]],
        headers: List[str],
        column_types: List[str]
    ) -> Dict[str, Any]:
        """Calculate statistics for CSV data."""
        stats = {}
        
        for idx, (header, col_type) in enumerate(zip(headers, column_types)):
            col_stats = {
                "type": col_type,
                "null_count": 0,
                "unique_values": set()
            }
            
            values = [row[idx] for row in data_rows if len(row) > idx]
            col_stats["null_count"] = sum(1 for v in values if not v)
            col_stats["unique_values"] = len(set(values))
            
            if col_type in ["integer", "float"]:
                numeric_values = [float(v) for v in values if v and self._is_numeric(v)]
                if numeric_values:
                    col_stats.update({
                        "min": min(numeric_values),
                        "max": max(numeric_values),
                        "mean": sum(numeric_values) / len(numeric_values),
                        "median": sorted(numeric_values)[len(numeric_values)//2]
                    })
            
            stats[header] = col_stats
            
        return stats

    async def _process_math_block(self, block: ContentBlock) -> ContentBlock:
        """Process math/LaTeX block."""
        try:
            content = block.content.strip('$')
            
            block.metadata.update({
                "expressions": self._parse_math_expressions(content),
                "symbols": self._extract_math_symbols(content),
                "complexity": self._analyze_math_complexity(content)
            })
            
            return block
        except Exception as e:
            logger.error(f"Error processing math block: {e}")
            self._metrics.error_count += 1
            return block

    def _parse_math_expressions(self, content: str) -> List[Dict[str, Any]]:
        """Parse mathematical expressions."""
        expressions = []
        
        # Basic expression parsing
        expr_pattern = r'\\begin\{equation\}(.*?)\\end\{equation\}|\\[(.*?)\\]'
        for match in re.finditer(expr_pattern, content, re.DOTALL):
            expr = match.group(1) or match.group(2)
            expressions.append({
                "expression": expr.strip(),
                "type": self._determine_math_type(expr)
            })
            
        return expressions

    def _determine_math_type(self, expr: str) -> str:
        """Determine type of mathematical expression."""
        if '\\sum' in expr or '\\prod' in expr:
            return "series"
        elif '\\int' in expr:
            return "integral"
        elif '\\frac' in expr:
            return "fraction"
        elif '\\lim' in expr:
            return "limit"
        elif '\\matrix' in expr or '\\begin{matrix}' in expr:
            return "matrix"
        else:
            return "algebraic"

    async def _process_diagram_block(self, block: ContentBlock) -> ContentBlock:
        """Process diagram block."""
        try:
            diagram_type = block.content.split('\n')[0].strip('`')
            content = '\n'.join(block.content.split('\n')[1:-1])
            
            block.metadata.update({
                "diagram_type": diagram_type,
                "elements": self._analyze_diagram_elements(content, diagram_type),
                "connections": self._analyze_diagram_connections(content, diagram_type),
                "complexity": self._calculate_diagram_complexity(content, diagram_type)
            })
            
            return block
        except Exception as e:
            logger.error(f"Error processing diagram block: {e}")
            self._metrics.error_count += 1
            return block

    def _analyze_diagram_elements(self, content: str, diagram_type: str) -> List[Dict[str, Any]]:
        """Analyze diagram elements."""
        elements = []
        
        if diagram_type == "mermaid":
            # Parse Mermaid diagram elements
            node_pattern = r'(\w+)\[(.*?)\]'
            for match in re.finditer(node_pattern, content):
                elements.append({
                    "id": match.group(1),
                    "label": match.group(2),
                    "type": "node"
                })
        elif diagram_type == "plantuml":
            # Parse PlantUML elements
            class_pattern = r'class\s+(\w+)(?:\s+\{(.*?)\})?'
            for match in re.finditer(class_pattern, content, re.DOTALL):
                elements.append({
                    "name": match.group(1),
                    "attributes": match.group(2).split('\n') if match.group(2) else [],
                    "type": "class"
                })
                
        return elements

    async def _process_pseudocode_block(self, block: ContentBlock) -> ContentBlock:
        """Process pseudocode block."""
        try:
            block.metadata.update({
                "structure": self._analyze_pseudocode_structure(block.content),
                "complexity": self._analyze_pseudocode_complexity(block.content),
                "variables": self._extract_pseudocode_variables(block.content)
            })
            
            return block
        except Exception as e:
            logger.error(f"Error processing pseudocode block: {e}")
            self._metrics.error_count += 1
            return block

    def _analyze_pseudocode_structure(self, content: str) -> Dict[str, Any]:
        """Analyze pseudocode structure."""
        structure = {
            "control_structures": [],
            "functions": [],
            "nesting_depth": 0
        }
        
        # Identify control structures
        control_patterns = {
            "if": r'\bif\b.*?:',
            "loop": r'\b(?:for|while)\b.*?:',
            "function": r'\b(?:function|procedure)\s+(\w+)'
        }
        
        for struct_type, pattern in control_patterns.items():
            matches = re.finditer(pattern, content, re.IGNORECASE)
            if struct_type == "function":
                structure["functions"].extend(m.group(1) for m in matches)
            else:
                structure["control_structures"].extend(struct_type for _ in re.finditer(pattern, content))
        
        # Calculate maximum nesting depth
        current_depth = 0
        max_depth = 0
        for line in content.split('\n'):
            if re.search(r':\s*$', line):
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif line.strip() and not line.startswith(' '):
                current_depth = 0
                
        structure["nesting_depth"] = max_depth
        
        return structure

    def _extract_functions(self, code: str, language: str) -> List[Dict[str, Any]]:
        """Extract function definitions from code."""
        functions = []
        if language == "python":
            pattern = r"def\s+(\w+)\s*\((.*?)\)"
        elif language in ["javascript", "java"]:
            pattern = r"function\s+(\w+)\s*\((.*?)\)"
        else:
            return functions
            
        matches = re.finditer(pattern, code)
        for match in matches:
            functions.append({
                "name": match.group(1),
                "params": [p.strip() for p in match.group(2).split(",") if p.strip()]
            })
            
        return functions
    
    def _extract_classes(self, code: str, language: str) -> List[Dict[str, Any]]:
        """Extract class definitions from code."""
        classes = []
        if language == "python":
            pattern = r"class\s+(\w+)(?:\((.*?)\))?"
        elif language in ["javascript", "java"]:
            pattern = r"class\s+(\w+)(?:\s+extends\s+(\w+))?"
        else:
            return classes
            
        matches = re.finditer(pattern, code)
        for match in matches:
            classes.append({
                "name": match.group(1),
                "parent": match.group(2) if match.group(2) else None
            })
            
        return classes
    
    def _extract_imports(self, code: str, language: str) -> List[str]:
        """Extract import statements from code."""
        imports = []
        if language == "python":
            pattern = r"(?:from\s+(\S+)\s+import|\bimport\s+([^;\n]+))"
        elif language == "javascript":
            pattern = r"(?:require\(['\"]([^'\"]+)['\"]\)|import.*?from\s+['\"]([^'\"]+)['\"])"
        else:
            return imports
            
        matches = re.finditer(pattern, code)
        for match in matches:
            imp = match.group(1) or match.group(2)
            imports.extend(i.strip() for i in imp.split(","))
            
        return list(set(imports))
    
    def _extract_variables(self, code: str, language: str) -> List[Dict[str, Any]]:
        """Extract variable declarations from code."""
        variables = []
        if language == "python":
            pattern = r"(\w+)\s*=\s*([^;\n]+)"
        elif language in ["javascript", "java"]:
            pattern = r"(?:var|let|const)\s+(\w+)\s*=\s*([^;\n]+)"
        else:
            return variables
            
        matches = re.finditer(pattern, code)
        for match in matches:
            variables.append({
                "name": match.group(1),
                "value": match.group(2).strip()
            })
            
        return variables
    
    def _analyze_json_structure(self, data: Any, depth: int = 0) -> Dict[str, Any]:
        """Analyze structure of JSON data."""
        if depth > 10:  # Prevent infinite recursion
            return {"type": "max_depth_reached"}
            
        if isinstance(data, dict):
            return {
                "type": "object",
                "properties": {
                    k: self._analyze_json_structure(v, depth + 1)
                    for k, v in data.items()
                }
            }
        elif isinstance(data, list):
            return {
                "type": "array",
                "length": len(data),
                "sample": self._analyze_json_structure(data[0], depth + 1) if data else None
            }
        else:
            return {"type": type(data).__name__}
    
    def _analyze_xml_structure(self, element: 'Element') -> Dict[str, Any]:
        """Analyze structure of XML element."""
        structure = {
            "tag": element.tag,
            "attributes": dict(element.attrib),
            "children": []
        }
        
        for child in element:
            structure["children"].append(self._analyze_xml_structure(child))
            
        return structure
    
    def _is_numeric(self, value: str) -> bool:
        """Check if a string value is numeric."""
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False
    
    def _is_date(self, value: str) -> bool:
        """Check if a string value is a date."""
        date_patterns = [
            r'^\d{4}-\d{2}-\d{2}$',  # YYYY-MM-DD
            r'^\d{2}/\d{2}/\d{4}$',  # MM/DD/YYYY
            r'^\d{2}-\d{2}-\d{4}$'   # DD-MM-YYYY
        ]
        return any(re.match(pattern, value) for pattern in date_patterns)
    
    async def _process_yaml_block(self, block: ContentBlock) -> ContentBlock:
        """Process YAML block."""
        try:
            import yaml
            data = yaml.safe_load(block.content)
            
            block.metadata.update({
                "structure": self._analyze_yaml_structure(data),
                "document_count": len(list(yaml.safe_load_all(block.content))),
                "keys": self._extract_yaml_keys(data)
            })
            
            return block
        except Exception as e:
            logger.error(f"Error processing YAML block: {e}")
            self._metrics.error_count += 1
            return block
    
    def _analyze_yaml_structure(self, data: Any) -> Dict[str, Any]:
        """Analyze structure of YAML data."""
        if isinstance(data, dict):
            return {
                "type": "mapping",
                "properties": {
                    k: self._analyze_yaml_structure(v)
                    for k, v in data.items()
                }
            }
        elif isinstance(data, list):
            return {
                "type": "sequence",
                "length": len(data),
                "sample": self._analyze_yaml_structure(data[0]) if data else None
            }
        else:
            return {"type": type(data).__name__}
    
    def _extract_yaml_keys(self, data: Any, prefix: str = "") -> List[str]:
        """Extract all keys from YAML data."""
        keys = []
        if isinstance(data, dict):
            for key, value in data.items():
                full_key = f"{prefix}.{key}" if prefix else key
                keys.append(full_key)
                if isinstance(value, (dict, list)):
                    keys.extend(self._extract_yaml_keys(value, full_key))
        elif isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, (dict, list)):
                    keys.extend(self._extract_yaml_keys(item, f"{prefix}[{i}]"))
        return keys

    def _extract_math_symbols(self, content: str) -> List[Dict[str, Any]]:
        """Extract mathematical symbols from content."""
        symbols = []
        
        # Common math symbols and their types
        symbol_patterns = {
            "operator": r'[+\-*/=<>≤≥≠]',
            "greek": r'\\(?:alpha|beta|gamma|delta|epsilon|zeta|eta|theta|iota|kappa|lambda|mu|nu|xi|omicron|pi|rho|sigma|tau|upsilon|phi|chi|psi|omega)',
            "function": r'\\(?:sin|cos|tan|log|ln|exp|lim|sup|inf|max|min)',
            "special": r'\\(?:infty|partial|nabla|exists|forall|in|notin|subset|supset)'
        }
        
        for symbol_type, pattern in symbol_patterns.items():
            for match in re.finditer(pattern, content):
                symbols.append({
                    "symbol": match.group(0),
                    "type": symbol_type,
                    "position": match.start()
                })
        
        return symbols
    
    def _analyze_math_complexity(self, content: str) -> Dict[str, Any]:
        """Analyze complexity of mathematical expressions."""
        complexity = {
            "nesting_depth": 0,
            "operator_count": 0,
            "function_count": 0,
            "variable_count": 0,
            "expression_type": "simple"
        }
        
        # Count nested structures
        brackets = {'{': '}', '[': ']', '(': ')'}
        stack = []
        max_depth = 0
        current_depth = 0
        
        for char in content:
            if char in brackets:
                stack.append(char)
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char in brackets.values():
                if stack and brackets[stack[-1]] == char:
                    stack.pop()
                    current_depth -= 1
        
        complexity["nesting_depth"] = max_depth
        
        # Count operators and functions
        complexity["operator_count"] = len(re.findall(r'[+\-*/=]', content))
        complexity["function_count"] = len(re.findall(r'\\[a-zA-Z]+', content))
        
        # Count variables (single letters not part of commands)
        complexity["variable_count"] = len(re.findall(r'(?<!\\)[a-zA-Z]', content))
        
        # Determine expression type
        if max_depth > 2 or complexity["operator_count"] > 5:
            complexity["expression_type"] = "complex"
        elif max_depth > 1 or complexity["operator_count"] > 3:
            complexity["expression_type"] = "moderate"
        
        return complexity
    
    def _analyze_diagram_connections(self, content: str, diagram_type: str) -> List[Dict[str, Any]]:
        """Analyze connections in diagram."""
        connections = []
        
        if diagram_type == "mermaid":
            # Parse Mermaid connections
            conn_pattern = r'(\w+)\s*(?:-->|---|\.\.\.|\=\=>)\s*(\w+)(?:\s*:?\s*([^:\n]+))?'
            for match in re.finditer(conn_pattern, content):
                connections.append({
                    "from": match.group(1),
                    "to": match.group(2),
                    "label": match.group(3).strip() if match.group(3) else None,
                    "type": "directed" if "-->" in match.group(0) else "undirected"
                })
        elif diagram_type == "plantuml":
            # Parse PlantUML connections
            conn_pattern = r'(\w+)\s*(?:-+>|\.+>|\*+>|\<+\-|\<+\.|\<+\*)\s*(\w+)(?:\s*:?\s*([^:\n]+))?'
            for match in re.finditer(conn_pattern, content):
                connections.append({
                    "from": match.group(1),
                    "to": match.group(2),
                    "label": match.group(3).strip() if match.group(3) else None,
                    "type": "inheritance" if "*" in match.group(0) else "association"
                })
        
        return connections
    
    def _calculate_diagram_complexity(self, content: str, diagram_type: str) -> Dict[str, Any]:
        """Calculate complexity metrics for diagram."""
        complexity = {
            "node_count": 0,
            "edge_count": 0,
            "max_edges_per_node": 0,
            "avg_edges_per_node": 0.0,  # Changed to float
            "complexity_score": 0
        }
        
        # Count nodes and edges
        if diagram_type == "mermaid":
            nodes = set(re.findall(r'(\w+)\[', content))
            edges = re.findall(r'(\w+)\s*(?:-->|---|\.\.\.|\=\=>)\s*(\w+)', content)
        elif diagram_type == "plantuml":
            nodes = set(re.findall(r'(?:class|object|component)\s+(\w+)', content))
            edges = re.findall(r'(\w+)\s*(?:-+>|\.+>|\*+>|\<+\-|\<+\.|\<+\*)\s*(\w+)', content)
        else:
            return complexity
        
        complexity["node_count"] = len(nodes)
        complexity["edge_count"] = len(edges)
        
        # Calculate edge metrics
        if nodes:
            edges_per_node = {}
            for edge in edges:
                edges_per_node[edge[0]] = edges_per_node.get(edge[0], 0) + 1
                edges_per_node[edge[1]] = edges_per_node.get(edge[1], 0) + 1
            
            if edges_per_node:
                complexity["max_edges_per_node"] = max(edges_per_node.values())
                complexity["avg_edges_per_node"] = sum(edges_per_node.values()) / (2 * len(nodes))
        
        # Calculate complexity score (0-100)
        if complexity["node_count"] > 0:
            score = (
                (complexity["edge_count"] * 2) +
                (complexity["max_edges_per_node"] * 5) +
                (complexity["node_count"])
            )
            complexity["complexity_score"] = min(100, int(score))  # Convert to int
        
        return complexity
    
    def _analyze_pseudocode_complexity(self, content: str) -> Dict[str, Any]:
        """Analyze complexity of pseudocode."""
        complexity = {
            "cyclomatic": 0,
            "nesting_depth": 0,
            "statement_count": 0,
            "branch_points": 0,
            "loop_count": 0
        }
        
        # Count control structures
        control_patterns = {
            "if": r'\bif\b',
            "else": r'\belse\b',
            "loop": r'\b(?:for|while|repeat|until)\b',
            "case": r'\b(?:case|switch)\b'
        }
        
        for struct_type, pattern in control_patterns.items():
            matches = len(re.findall(pattern, content, re.IGNORECASE))
            if struct_type in ["if", "case"]:
                complexity["branch_points"] += matches
            elif struct_type == "loop":
                complexity["loop_count"] += matches
        
        # Calculate cyclomatic complexity
        complexity["cyclomatic"] = complexity["branch_points"] + complexity["loop_count"] + 1
        
        # Calculate nesting depth
        current_depth = 0
        max_depth = 0
        for line in content.split('\n'):
            if re.search(r':\s*$', line):
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif line.strip() and not line.startswith(' '):
                current_depth = 0
        
        complexity["nesting_depth"] = max_depth
        
        # Count statements
        complexity["statement_count"] = len([
            line for line in content.split('\n')
            if line.strip() and not line.strip().startswith(('#', '//'))
        ])
        
        return complexity
    
    def _extract_pseudocode_variables(self, content: str) -> List[Dict[str, Any]]:
        """Extract variables from pseudocode."""
        variables = []
        
        # Common variable declaration patterns
        patterns = [
            r'\b(?:let|var|set)\s+(\w+)(?:\s*=\s*([^:\n]+))?',  # let/var/set declarations
            r'\b(\w+)\s*(?:=|←)\s*([^:\n]+)',  # Assignment with = or ←
            r'\b(?:int|float|string|bool|array)\s+(\w+)',  # Type declarations
        ]
        
        for pattern in patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                var_name = match.group(1)
                var_value = match.group(2).strip() if len(match.groups()) > 1 and match.group(2) else None
                
                # Skip if variable is a keyword
                if var_name.lower() not in ["if", "else", "for", "while", "return", "break", "continue"]:
                    variables.append({
                        "name": var_name,
                        "value": var_value,
                        "line": content[:match.start()].count('\n') + 1
                    })
        
        return list({v["name"]: v for v in variables}.values())  # Remove duplicates

class CodeAnalysisInstruction(ResponseParsingInstruction):
    """Specialized instruction for code analysis."""
    
    def __init__(self):
        super().__init__()
        self.name = "code_analysis"
        self.description = "Analyze code blocks in responses"
        
    async def _execute_impl(self, context: Dict[str, Any]) -> InstructionResult:
        """Execute code analysis."""
        result = await super()._execute_impl(context)
        
        if result.status == InstructionStatus.COMPLETED and result.data is not None and "blocks" in result.data:
            # Filter for code blocks
            code_blocks = [
                block for block in result.data["blocks"]
                if block["type"] == ContentType.CODE.value
            ]
            
            # Perform detailed analysis
            analysis = await self._analyze_code_blocks(code_blocks)
            if result.data is not None:  # Extra check to satisfy type checker
                result.data["analysis"] = analysis
            
        return result
    
    async def _analyze_code_blocks(
        self,
        blocks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Perform detailed code analysis."""
        analysis = {
            "total_lines": sum(b["metadata"]["line_count"] for b in blocks),
            "languages": {},
            "complexity": {},
            "patterns": self._identify_patterns(blocks)
        }
        
        # Analyze by language
        for block in blocks:
            lang = block["metadata"]["language"]
            if lang not in analysis["languages"]:
                analysis["languages"][lang] = {
                    "blocks": 0,
                    "lines": 0,
                    "functions": 0,
                    "classes": 0
                }
            
            lang_stats = analysis["languages"][lang]
            lang_stats["blocks"] += 1
            lang_stats["lines"] += block["metadata"]["line_count"]
            lang_stats["functions"] += len(block["metadata"].get("functions", []))
            lang_stats["classes"] += len(block["metadata"].get("classes", []))
        
        return analysis
    
    def _identify_patterns(self, blocks: List[Dict[str, Any]]) -> List[str]:
        """Identify common code patterns."""
        patterns = []
        
        # Look for common patterns
        for block in blocks:
            content = block["content"].lower()
            
            if "try" in content and "except" in content:
                patterns.append("error_handling")
            if "async" in content or "await" in content:
                patterns.append("asynchronous")
            if "class" in content:
                patterns.append("object_oriented")
            if "import" in content or "require" in content:
                patterns.append("module_based")
            
        return list(set(patterns))  # Remove duplicates

class ImageAnalysisInstruction(ResponseParsingInstruction):
    """Specialized instruction for image analysis."""
    
    def __init__(self):
        super().__init__()
        self.name = "image_analysis"
        self.description = "Analyze image content in responses"
        
    async def _execute_impl(self, context: Dict[str, Any]) -> InstructionResult:
        """Execute image analysis."""
        result = await super()._execute_impl(context)
        
        if result.status == InstructionStatus.COMPLETED and result.data is not None and "blocks" in result.data:
            # Filter for image blocks
            image_blocks = [
                block for block in result.data["blocks"]
                if block["type"] == ContentType.IMAGE.value
            ]
            
            # Analyze images
            analysis = await self._analyze_image_blocks(image_blocks)
            if result.data is not None:  # Extra check to satisfy type checker
                result.data["analysis"] = analysis
            
        return result
    
    async def _analyze_image_blocks(
        self,
        blocks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze image blocks."""
        analysis = {
            "total_images": len(blocks),
            "formats": {},
            "sources": {},
            "has_alt_text": 0
        }
        
        for block in blocks:
            # Count formats
            if "mime_type" in block["metadata"]:
                format = block["metadata"]["mime_type"]
                analysis["formats"][format] = analysis["formats"].get(format, 0) + 1
            
            # Count source types
            encoding = block["metadata"].get("encoding", "unknown")
            analysis["sources"][encoding] = analysis["sources"].get(encoding, 0) + 1
            
            # Check alt text
            if block["metadata"].get("alt_text"):
                analysis["has_alt_text"] += 1
        
        return analysis 