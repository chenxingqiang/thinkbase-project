"""Transformer module for text processing."""

import logging
from typing import List, Dict, Any, Optional, Union
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from collections import Counter
import asyncio

from agentflow.core.instructions.response_parsing import ContentType, ContentBlock, ResponseParsingInstruction

logger = logging.getLogger(__name__)

class AgenticTransformer:
    """Transformer for text processing."""
    
    def __init__(self):
        """Initialize the transformer."""
        self.vectorizer = TfidfVectorizer(
            max_df=1.0,  # Keep all terms
            min_df=1,    # Keep terms that appear in at least 1 doc
            stop_words=None,  # Don't use stop words - let the code handle them
            ngram_range=(1, 3),  # Use 1-3 word phrases
            max_features=None,   # Don't limit vocab size
            token_pattern=r'(?u)\b\w[\w\.]+\b',  # Include dots for method names
            tokenizer=self._create_code_tokenizer()  # Use our custom tokenizer
        )
        self.response_parser = ResponseParsingInstruction()
        self._code_tokenizer = self._create_code_tokenizer()
        self.code_patterns = {
            'function_def': r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(',
            'class_def': r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[:\(]',
            'variable_def': r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=',
            'import_stmt': r'import\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            'comment': r'#\s*(.*?)$'
        }
        
    def _create_code_tokenizer(self):
        """Create a custom tokenizer for code."""
        def tokenize(text: str) -> List[str]:
            # Split on whitespace and special characters
            tokens = re.findall(r'\b\w+\b|[^\w\s]', text)
            return [t for t in tokens if t]
        return tokenize

    def process_text(self, text: str) -> str:
        """Process text to extract meaningful tokens while preserving structure.
        
        Args:
            text: Input text to process
            
        Returns:
            Processed text
        """
        if not text or text.isspace():
            logger.warning("Empty or whitespace text")
            return ""
            
        # Process text while preserving structure
        processed_text = text.strip()
        return processed_text

    async def extract_patterns(self, text: str) -> List[Dict[str, Any]]:
        """Extract patterns from text.
        
        Args:
            text: Input text to process
            
        Returns:
            List of extracted patterns
        """
        logger.info(f"Extracting patterns from text:\n{text}")
        
        if not text or text.isspace():
            logger.warning("Empty or whitespace text")
            return []
            
        # Parse content using response parser
        context = {"response": text}
        blocks = await self.response_parser.execute(context)
            
        patterns = []
        
        # Process each content block
        for block in blocks:
            block_type = block.type.value
            block_content = block.content
            
            # Skip empty blocks
            if not block_content:
                continue
                
            # Preprocess block content
            tokens = self._code_tokenizer(block_content)
            if not tokens:
                continue
                
            # Join tokens back into text for vectorizer
            processed_text = " ".join(tokens)
            
            try:
                # Fit and transform the text
                X = self.vectorizer.fit_transform([processed_text])
                feature_names = self.vectorizer.get_feature_names_out()
                
                # Get token weights
                token_weights = {}
                for i, weight in enumerate(X.toarray()[0]):
                    if weight > 0:
                        token_weights[feature_names[i]] = weight
                        
                logger.debug(f"Found {len(token_weights)} tokens with weights")
                
                # Convert to patterns based on content type
                for token, weight in token_weights.items():
                    # Skip very short tokens
                    if len(token) < 2:
                        continue
                        
                    # Determine pattern type based on content type
                    if block_type == ContentType.CODE.value:
                        if any(pattern in token for pattern in [
                            "def", "class", "return", "if", "for", "while",
                            "import", "from", "raise", "try", "except", "finally"
                        ]):
                            pattern_type = "code"
                        elif any(char in token for char in "()[]{}"):
                            pattern_type = "structure"
                        elif any(char in token for char in "+-*/%=<>!&|^~"):
                            pattern_type = "operation"
                        else:
                            pattern_type = "code"
                    elif block_type == ContentType.MARKDOWN.value:
                        if any(token.startswith(s) for s in ['"""', "'''", "#"]):
                            pattern_type = "documentation"
                        else:
                            pattern_type = "text"
                    elif block_type == ContentType.JSON.value:
                        pattern_type = "structure"
                    else:
                        pattern_type = "text"
                        
                    patterns.append({
                        "type": pattern_type,
                        "text": token,
                        "weight": float(weight),
                        "content_type": block_type
                    })
                    logger.debug(f"Added pattern: {pattern_type} - {token[:100]}")
                    
            except Exception as e:
                logger.error(f"Error extracting patterns from block: {e}")
                continue
                
        logger.info(f"Extracted {len(patterns)} patterns")
        return patterns

    def generate_text(self, pattern: Dict[str, Any]) -> str:
        """Generate text from pattern.
        
        Args:
            pattern: Pattern to generate from
            
        Returns:
            Generated text
        """
        text = []
        
        # Generate code elements
        if 'code_elements' in pattern:
            for element in pattern['code_elements']:
                if element['type'] == 'function':
                    text.append(f"def {element['name']}({element.get('params', '')}):")
                    if 'body' in element:
                        text.extend([f"    {line}" for line in element['body'].split('\n')])
                elif element['type'] == 'class':
                    text.append(f"class {element['name']}:")
                    if 'body' in element:
                        text.extend([f"    {line}" for line in element['body'].split('\n')])
                        
        return '\n'.join(text)
