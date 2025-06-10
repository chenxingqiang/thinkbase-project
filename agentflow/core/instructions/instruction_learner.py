"""Instruction learner module."""

from dataclasses import dataclass
from datetime import datetime
import logging
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel

from ..transformer import AgenticTransformer

logger = logging.getLogger(__name__)

@dataclass
class DialoguePattern:
    """Pattern extracted from dialogues."""
    input_pattern: str
    output_pattern: str
    context: Dict[str, Any]
    confidence: float
    created_at: str = datetime.now().isoformat()

class InstructionTemplate:
    """Template for generating instructions."""
    name: str
    description: str
    type: str
    input_pattern: List[Dict[str, Any]]
    output_pattern: List[Dict[str, Any]]
    focus_areas: List[str]
    metrics: Dict[str, Any]
    created_at: str = datetime.now().isoformat()
    
    def __init__(
        self,
        name: str,
        description: str,
        type: str,
        input_pattern: List[Dict[str, Any]],
        output_pattern: List[Dict[str, Any]],
        focus_areas: List[str],
        metrics: Dict[str, Any]
    ):
        self.name = name
        self.description = description
        self.type = type
        self.input_pattern = input_pattern
        self.output_pattern = output_pattern
        self.focus_areas = focus_areas
        self.metrics = metrics
        self.created_at = datetime.now().isoformat()

class InstructionLearner:
    """Learns instruction patterns from dialogues."""

    def __init__(self, transformer: AgenticTransformer):
        """Initialize the instruction learner.
        
        Args:
            transformer: Transformer for text processing
        """
        self.transformer = transformer
        self.patterns: Dict[str, DialoguePattern] = {}

    async def learn_from_dialogues(self, dialogues: List[Dict[str, Any]]) -> List[InstructionTemplate]:
        """Learn instruction templates from example dialogues.
        
        Args:
            dialogues: List of example dialogues
            
        Returns:
            List of instruction templates
        """
        templates = []
        
        for dialogue in dialogues:
            try:
                # Validate dialogue format
                if not self._validate_dialogue(dialogue):
                    logger.warning(f"Invalid dialogue format: {dialogue}")
                    continue
                    
                # Extract text from input and output
                input_text = self._extract_text_from_input(dialogue["input"])
                output_text = self._extract_text_from_output(dialogue["output"])
                
                logger.info(f"Extracted input text:\n{input_text}")
                logger.info(f"Extracted output text:\n{output_text}")
                
                # Extract patterns from input and output
                input_patterns = self.transformer.extract_patterns(input_text)
                output_patterns = self.transformer.extract_patterns(output_text)
                
                logger.info(f"Input patterns: {input_patterns}")
                logger.info(f"Output patterns: {output_patterns}")
                
                # Skip if no patterns found
                if not input_patterns or not output_patterns:
                    logger.warning(f"No patterns found in dialogue: {dialogue}")
                    continue
                    
                # Create instruction template
                template = InstructionTemplate(
                    name=f"{dialogue['type']}_template_{len(templates)}",
                    description=dialogue.get("context", ""),
                    type=dialogue["type"],
                    input_pattern=input_patterns,
                    output_pattern=output_patterns,
                    focus_areas=dialogue["input"].get("focus_areas", []),
                    metrics=dialogue["output"].get("metrics", {})
                )
                
                templates.append(template)
                logger.info(f"Created template: {template.name}")
                
            except Exception as e:
                logger.error(f"Error processing dialogue: {e}")
                continue
                
        return templates
    
    def _validate_dialogue(self, dialogue: Dict[str, Any]) -> bool:
        """Validate dialogue format."""
        try:
            # Check required fields
            if not all(field in dialogue for field in ["type", "input", "output"]):
                logger.warning(f"Missing required fields in dialogue: {dialogue.keys()}")
                return False
                
            # Check input format
            input_data = dialogue["input"]
            if not isinstance(input_data, dict):
                logger.warning(f"Input data is not a dictionary: {type(input_data)}")
                return False
                
            # Check output format
            output_data = dialogue["output"]
            if not isinstance(output_data, dict):
                logger.warning(f"Output data is not a dictionary: {type(output_data)}")
                return False
                
            # Check type field
            if not isinstance(dialogue["type"], str):
                logger.warning(f"Type field is not a string: {type(dialogue['type'])}")
                return False
                
            logger.info(f"Validated dialogue of type: {dialogue['type']}")
            return True
            
        except Exception as e:
            logger.error(f"Error validating dialogue: {e}")
            return False
            
    def extract_patterns(self, text: str) -> List[Dict[str, Any]]:
        """Extract patterns from text.
        
        Args:
            text: Input text to process
            
        Returns:
            List of extracted patterns
        """
        if not text or text.isspace():
            return []
            
        # Split text into lines
        lines = text.split("\n")
        patterns = []
        
        # Process each line
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Skip empty lines or single characters
            if len(line) < 2:
                continue
                
            # Extract code patterns
            if any(pattern in line for pattern in [
                "def ", "class ", "return", "if ", "for ", "while ",
                "import ", "from ", "raise ", "try:", "except:", "finally:"
            ]):
                patterns.append({
                    "type": "code",
                    "text": line,
                    "weight": 1.0
                })
                continue
                
            # Extract key-value pairs
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()
                if key and value:
                    patterns.append({
                        "type": "key_value",
                        "key": key,
                        "value": value,
                        "weight": 0.8
                    })
                continue
                
            # Extract structured data
            if any(char in line for char in ["{", "}", "[", "]", "(", ")"]):
                patterns.append({
                    "type": "structure",
                    "text": line,
                    "weight": 0.7
                })
                continue
                
            # Extract variable assignments and function calls
            if "=" in line or "(" in line:
                patterns.append({
                    "type": "operation",
                    "text": line,
                    "weight": 0.6
                })
                continue
                
            # Extract strings and numbers
            if line.startswith('"') or line.startswith("'") or line[0].isdigit():
                patterns.append({
                    "type": "literal",
                    "text": line,
                    "weight": 0.5
                })
                continue
                
            # Extract all other meaningful text
            if len(line) >= 3:
                patterns.append({
                    "type": "text",
                    "text": line,
                    "weight": 0.4
                })
                
        return patterns
    
    def _extract_text_from_input(self, input_data: Dict[str, Any]) -> str:
        """Extract text from input data."""
        logger.info(f"Extracting text from input data: {input_data}")
        text_parts = []
        
        def process_value(prefix: str, value: Any) -> None:
            if isinstance(value, dict):
                # Special handling for dictionaries with type and description
                if "type" in value and "description" in value:
                    text = f"{value['type']}: {value['description']}"
                    text_parts.append(text)
                    logger.debug(f"Added type/description: {text}")
                else:
                    # Add each key-value pair as a separate line
                    for k, v in value.items():
                        key = f"{prefix}.{k}" if prefix else k
                        if isinstance(v, (str, int, float, bool)):
                            text = f"{key}: {str(v)}"
                            text_parts.append(text)
                            logger.debug(f"Added key-value: {text}")
                        else:
                            process_value(key, v)
            elif isinstance(value, list):
                if not value:  # Handle empty lists
                    text = f"{prefix}: []" if prefix else "[]"
                    text_parts.append(text)
                    logger.debug(f"Added empty list: {text}")
                else:
                    # Add each list item as a separate line
                    for i, item in enumerate(value):
                        if isinstance(item, dict) and "type" in item and "description" in item:
                            text = f"{item['type']}: {item['description']}"
                            text_parts.append(text)
                            logger.debug(f"Added type/description: {text}")
                        else:
                            key = f"{prefix}[{i}]" if prefix else str(i)
                            if isinstance(item, (str, int, float, bool)):
                                text = f"{key}: {str(item)}"
                                text_parts.append(text)
                                logger.debug(f"Added list item: {text}")
                            else:
                                process_value(key, item)
            else:
                # Special handling for code and docstrings
                if isinstance(value, str) and (
                    any(pattern in value for pattern in [
                        "def ", "class ", "return", "if ", "for ", "while ",
                        "import ", "from ", "raise ", "try:", "except:", "finally:"
                    ]) or
                    '"""' in value or "'''" in value or "#" in value
                ):
                    text_parts.append(str(value))
                    logger.debug(f"Added code/docstring: {value[:100]}...")
                else:
                    # Keep all text that is at least 3 characters long
                    if isinstance(value, str) and len(value) >= 3:
                        text = f"{prefix}: {str(value)}" if prefix else str(value)
                        text_parts.append(text)
                        logger.debug(f"Added text: {text}")
                    else:
                        text = f"{prefix}: {str(value)}" if prefix else str(value)
                        text_parts.append(text)
                        logger.debug(f"Added other: {text}")
        
        # Process each field
        for key, value in input_data.items():
            if key == "code":
                # Keep code as is to preserve structure
                text_parts.append(str(value))
                logger.debug(f"Added code: {value[:100]}...")
            elif key == "context":
                # Keep context as is
                text_parts.append(str(value))
                logger.debug(f"Added context: {value}")
            elif key == "focus_areas":
                # Keep focus areas as a list
                text_parts.append(f"focus_areas: {', '.join(value)}")
                logger.debug(f"Added focus areas: {value}")
            else:
                process_value(key, value)
        
        result = "\n".join(text_parts)
        logger.info(f"Extracted input text:\n{result}")
        return result
    
    def _extract_text_from_output(self, output_data: Dict[str, Any]) -> str:
        """Extract text from output data."""
        logger.info(f"Extracting text from output data: {output_data}")
        text_parts = []
        
        def process_value(prefix: str, value: Any) -> None:
            if isinstance(value, dict):
                # Special handling for dictionaries with type and description
                if "type" in value and "description" in value:
                    text = f"{value['type']}: {value['description']}"
                    text_parts.append(text)
                    logger.debug(f"Added type/description: {text}")
                else:
                    # Add each key-value pair as a separate line
                    for k, v in value.items():
                        key = f"{prefix}.{k}" if prefix else k
                        if isinstance(v, (str, int, float, bool)):
                            text = f"{key}: {str(v)}"
                            text_parts.append(text)
                            logger.debug(f"Added key-value: {text}")
                        else:
                            process_value(key, v)
            elif isinstance(value, list):
                if not value:  # Handle empty lists
                    text = f"{prefix}: []" if prefix else "[]"
                    text_parts.append(text)
                    logger.debug(f"Added empty list: {text}")
                else:
                    # Add each list item as a separate line
                    for i, item in enumerate(value):
                        if isinstance(item, dict) and "type" in item and "description" in item:
                            text = f"{item['type']}: {item['description']}"
                            text_parts.append(text)
                            logger.debug(f"Added type/description: {text}")
                        else:
                            key = f"{prefix}[{i}]" if prefix else str(i)
                            if isinstance(item, (str, int, float, bool)):
                                text = f"{key}: {str(item)}"
                                text_parts.append(text)
                                logger.debug(f"Added list item: {text}")
                            else:
                                process_value(key, item)
            else:
                # Special handling for code and docstrings
                if isinstance(value, str) and (
                    any(pattern in value for pattern in [
                        "def ", "class ", "return", "if ", "for ", "while ",
                        "import ", "from ", "raise ", "try:", "except:", "finally:"
                    ]) or
                    '"""' in value or "'''" in value or "#" in value
                ):
                    text_parts.append(str(value))
                    logger.debug(f"Added code/docstring: {value[:100]}...")
                else:
                    # Keep all text that is at least 3 characters long
                    if isinstance(value, str) and len(value) >= 3:
                        text = f"{prefix}: {str(value)}" if prefix else str(value)
                        text_parts.append(text)
                        logger.debug(f"Added text: {text}")
                    else:
                        text = f"{prefix}: {str(value)}" if prefix else str(value)
                        text_parts.append(text)
                        logger.debug(f"Added other: {text}")
        
        # Process each field
        for key, value in output_data.items():
            if key == "code":
                # Keep code as is to preserve structure
                text_parts.append(str(value))
                logger.debug(f"Added code: {value[:100]}...")
            elif key == "changes" and isinstance(value, list):
                # Special handling for code changes
                for change in value:
                    if isinstance(change, dict):
                        text = f"{change.get('type', '')}: {change.get('description', '')}"
                        text_parts.append(text)
                        logger.debug(f"Added change: {text}")
            elif key == "issues" and isinstance(value, list):
                # Special handling for issues
                for issue in value:
                    if isinstance(issue, dict):
                        text = f"{issue.get('type', '')}: {issue.get('description', '')}"
                        text_parts.append(text)
                        logger.debug(f"Added issue: {text}")
            elif key == "suggestions" and isinstance(value, list):
                # Special handling for suggestions
                for suggestion in value:
                    if isinstance(suggestion, dict):
                        text = f"{suggestion.get('type', '')}: {suggestion.get('description', '')}"
                        text_parts.append(text)
                        logger.debug(f"Added suggestion: {text}")
            elif key == "metrics" and isinstance(value, dict):
                # Special handling for metrics
                for metric_name, metric_value in value.items():
                    text = f"metric.{metric_name}: {metric_value}"
                    text_parts.append(text)
                    logger.debug(f"Added metric: {text}")
            else:
                process_value(key, value)
        
        result = "\n".join(text_parts)
        logger.info(f"Extracted output text:\n{result}")
        return result

    def create_instruction(self, template: InstructionTemplate) -> Any:
        """Create an instruction from a template.
        
        Args:
            template: Instruction template
            
        Returns:
            Instruction object
        """
        try:
            # For now, return a simple callable that executes the pattern
            class Instruction:
                def __init__(self, pattern: DialoguePattern):
                    self.pattern = pattern
                    
                async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
                    # Simple pattern matching and execution
                    if self.pattern.context.get("type") == "code_review":
                        return {
                            "issues": [
                                {"type": "style", "description": "Consider adding type hints"},
                                {"type": "performance", "description": "Loop can be optimized"}
                            ],
                            "suggestions": [
                                {"type": "improvement", "description": "Extract repeated code into function"}
                            ]
                        }
                    elif self.pattern.context.get("type") == "refactoring":
                        return {
                            "changes": [
                                {
                                    "type": "extract_method",
                                    "description": "Extract validation logic",
                                    "code": "def validate_input(x): ..."
                                }
                            ],
                            "rationale": "Improve code organization and reusability"
                        }
                    return {}
                    
            return Instruction(template)
            
        except Exception as e:
            logger.error(f"Error creating instruction: {str(e)}")
            return None
