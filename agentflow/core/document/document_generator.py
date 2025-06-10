import os
import json
import re
from typing import Dict, Any, Optional, Union, List, Callable
from jinja2 import Environment, FileSystemLoader, select_autoescape
import markdown2
import pdfkit
import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator, ConfigDict

class DocumentSchema(BaseModel):
    """Pydantic model for document validation"""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='ignore'
    )
    
    title: str = Field(min_length=1, max_length=200)
    summary: Optional[str] = Field(default=None, max_length=1000)
    author: Optional[str] = Field(default=None, max_length=100)
    date: Optional[str] = Field(default=None)
    sections: Optional[Dict[str, str]] = None
    tags: Optional[List[str]] = None
    research_questions: Optional[List[str]] = None
    
    @field_validator('tags', mode='before')
    @classmethod
    def validate_tags(cls, tags):
        """Validate tags"""
        if tags:
            # Ensure tags are non-empty and non-whitespace
            validated_tags = [tag.strip() for tag in tags if tag.strip()]
            if not validated_tags:
                raise ValueError("Tags must be non-empty")
            return validated_tags
        return tags
    
    @field_validator('title')
    @classmethod
    def validate_title(cls, title):
        """Validate title"""
        if not title or not title.strip():
            raise ValueError("Title must not be empty")
        return title.strip()

class ContentParser:
    """Advanced content parsing utility with validation"""
    
    @classmethod
    def validate_document(cls, 
                          content: Dict[str, Any], 
                          schema: type[BaseModel] = DocumentSchema) -> Dict[str, Any]:
        """Validate document content against a schema
        
        Args:
            content: Content to validate
            schema: Pydantic schema to validate against
        
        Returns:
            Validated content
        """
        try:
            return schema(**content).model_dump(exclude_unset=True)
        except ValidationError as e:
            # Detailed error handling
            error_details = {}
            for error in e.errors():
                loc = '.'.join(str(x) for x in error['loc'])
                error_details[loc] = error['msg']
            
            raise ValueError(f"Document validation failed: {error_details}")
    
    @staticmethod
    def parse_markdown(content: str) -> Dict[str, Any]:
        """Parse markdown-like content into structured data
        
        Args:
            content: Input markdown-like content
        
        Returns:
            Parsed content dictionary
        """
        # Split content into sections
        sections = {}
        current_section = None
        parsed_content = {}
        
        for line in content.split('\n'):
            # Check for section headers
            header_match = re.match(r'^##\s*(.+)$', line)
            if header_match:
                current_section = header_match.group(1).strip()
                sections[current_section] = []
                continue
            
            # Check for key-value pairs
            kv_match = re.match(r'^(\w+):\s*(.+)$', line)
            if kv_match:
                key = kv_match.group(1)
                value = kv_match.group(2)
                
                # Special handling for lists
                if value.startswith('[') and value.endswith(']'):
                    # Parse list
                    value = [v.strip().strip("'\"") for v in value[1:-1].split(',')]
                
                parsed_content[key] = value
                continue
            
            # Add content to current section
            if current_section and line.strip():
                sections[current_section].append(line.strip())
        
        # Convert section lists to strings
        parsed_content['sections'] = {
            section: '\n'.join(content) 
            for section, content in sections.items()
        }
        
        return parsed_content
    
    @staticmethod
    def parse_yaml(content: str) -> Dict[str, Any]:
        """Parse YAML content
        
        Args:
            content: Input YAML content
        
        Returns:
            Parsed content dictionary
        """
        try:
            return yaml.safe_load(content)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML content: {e}")
    
    @staticmethod
    def parse_json(content: str) -> Dict[str, Any]:
        """Parse JSON content
        
        Args:
            content: Input JSON content
        
        Returns:
            Parsed content dictionary
        """
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON content: {e}")
    
    @classmethod
    def extract_metadata(cls, 
                        content: Union[str, Dict[str, Any]], 
                        keys: Optional[List[str]] = None) -> Dict[str, Any]:
        """Extract specific metadata from content
        
        Args:
            content: Content to extract metadata from
            keys: Optional list of keys to extract
        
        Returns:
            Extracted metadata
        """
        # If content is a string, parse it first
        if isinstance(content, str):
            # Try parsing methods
            parse_methods = [
                cls.parse_markdown,
                cls.parse_yaml,
                cls.parse_json
            ]
            
            for method in parse_methods:
                try:
                    content = method(content)
                    break
                except ValueError:
                    continue
            else:
                raise ValueError("Unable to parse content")
        
        # Extract metadata
        if keys:
            return {k: content.get(k) for k in keys if k in content}
        
        return content

class DocumentGenerator:
    """Advanced document generation utility with multiple format support"""
    
    SUPPORTED_FORMATS = ['markdown', 'html', 'pdf', 'json', 'txt', 'yaml']
    
    def __init__(self, 
                 template_dir: Optional[str] = None, 
                 default_template: str = 'default.j2'):
        """Initialize document generator
        
        Args:
            template_dir: Directory containing Jinja2 templates
            default_template: Default template to use if not specified
        """
        self.template_dir = template_dir or os.path.join(
            os.path.dirname(__file__), 'templates'
        )
        self.default_template = default_template
        
        # Setup Jinja2 environment
        self.jinja_env = Environment(
            loader=FileSystemLoader(self.template_dir),
            autoescape=select_autoescape(['html', 'xml'])
        )
        
        # Content parser
        self.content_parser = ContentParser()
    
    def _parse_content(self, 
                       content: Union[Dict[str, Any], str], 
                       parse_method: Optional[str] = None) -> Dict[str, Any]:
        """Parse input content
        
        Args:
            content: Content to parse
            parse_method: Optional parsing method (markdown, yaml, json)
        
        Returns:
            Parsed content dictionary
        """
        # If already a dictionary, return as-is
        if isinstance(content, dict):
            return content
        
        # Determine parsing method if not specified
        if not parse_method:
            # Try to guess parsing method
            content_str = str(content)
            if content_str.startswith('{') and content_str.endswith('}'):
                parse_method = 'json'
            elif content_str.startswith('---\n'):  # YAML front matter
                parse_method = 'yaml'
            else:
                parse_method = 'markdown'
        
        # Parse content based on method
        parse_methods = {
            'markdown': self.content_parser.parse_markdown,
            'yaml': self.content_parser.parse_yaml,
            'json': self.content_parser.parse_json
        }
        
        if parse_method not in parse_methods:
            raise ValueError(f"Unsupported parsing method: {parse_method}")
        
        return parse_methods[parse_method](content)
    
    def _render_template(self, 
                         content: Dict[str, Any], 
                         template_name: Optional[str] = None) -> str:
        """Render content using Jinja2 template
        
        Args:
            content: Dictionary of content to render
            template_name: Optional template name
        
        Returns:
            Rendered template string
        """
        template_name = template_name or self.default_template
        try:
            template = self.jinja_env.get_template(template_name)
            return template.render(**content)
        except Exception as e:
            # Fallback to basic rendering if template fails
            return '\n'.join(f"{k}: {v}" for k, v in content.items())
    
    def generate(self, 
                 content: Union[Dict[str, Any], str], 
                 format: str = 'markdown', 
                 output_path: Optional[str] = None,
                 template: Optional[str] = None,
                 parse_method: Optional[str] = None) -> Union[str, None]:
        """Generate document in specified format
        
        Args:
            content: Content to generate document from
            format: Output document format
            output_path: Optional path to save document
            template: Optional custom template to use
            parse_method: Optional parsing method for string content
            
        Returns:
            Generated document content or path
        """
        # Validate format
        format = format.lower()
        if format not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {format}. Supported formats: {self.SUPPORTED_FORMATS}")
        
        # Parse content
        parsed_content = self._parse_content(content, parse_method)
        
        # Validate content
        validated_content = self.content_parser.validate_document(parsed_content)
        
        # Render content
        rendered_content = self._render_template(validated_content, template)
        
        # Convert to specific format
        formatted_content = self._convert_format(rendered_content, format)
        
        # Output handling
        if output_path:
            self._save_document(formatted_content, output_path, format)
            return output_path
        
        return formatted_content
    
    def _convert_format(self, content: str, format: str) -> str:
        """Convert content to specified format
        
        Args:
            content: Input content
            format: Desired output format
        
        Returns:
            Formatted content
        """
        if format == 'markdown':
            return content
        elif format == 'html':
            return markdown2.markdown(content)
        elif format == 'json':
            # Convert markdown-like content to JSON
            return json.dumps({
                'content': content.split('\n'),
                'metadata': {'format': 'markdown'}
            }, indent=2)
        elif format == 'txt':
            return content
        elif format == 'yaml':
            # Convert content to YAML
            return yaml.safe_dump({
                'content': content.split('\n')
            }, default_flow_style=False)
        elif format == 'pdf':
            # Requires wkhtmltopdf to be installed
            pdf_content = pdfkit.from_string(markdown2.markdown(content), False)
            return pdf_content.decode('utf-8') if isinstance(pdf_content, bytes) else str(pdf_content)
        
        return content
    
    def _save_document(self, content: str, path: str, format: str):
        """Save document to specified path
        
        Args:
            content: Document content
            path: Output file path
            format: Document format
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Write content
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    @classmethod
    def list_templates(cls, template_dir: Optional[str] = None) -> list:
        """List available templates
        
        Args:
            template_dir: Optional directory to search for templates
        
        Returns:
            List of available template names
        """
        if not template_dir:
            template_dir = os.path.join(
                os.path.dirname(__file__), 'templates'
            )
        
        return [f for f in os.listdir(template_dir) if f.endswith('.j2')]
