# Tutorial 4: Building Real-World Applications

This tutorial demonstrates how to build production-ready applications using the AgentISA framework, focusing on practical use cases and best practices.

## Prerequisites

- Completed Tutorials 1-3
- Understanding of production deployment concepts
- Familiarity with async web frameworks

## 1. Building a Smart Document Processor

Let's build a document processing system that can handle various file formats, extract information, and learn from processing patterns.

### Core Components

```python
from agentflow.core.isa import (
    Instruction, 
    InstructionType, 
    TypeConstraint,
    CompositeInstruction,
    AgentISA,
    AgentISAConfig
)
from typing import Dict, Any, List, Optional
import asyncio
import json
from pathlib import Path
import mimetypes

class DocumentMetadata:
    """Document metadata container."""
    def __init__(self, path: str):
        self.path = Path(path)
        self.mime_type = mimetypes.guess_type(path)[0]
        self.size = self.path.stat().st_size
        self.extension = self.path.suffix.lower()

class FileReaderInstruction(Instruction):
    """Read and validate document files."""
    
    def __init__(self):
        super().__init__(
            id="file_reader",
            name="file_reader",
            type=InstructionType.RESOURCE_MANAGEMENT,
            params={
                "file_path": TypeConstraint(base_type=str),
                "max_size_mb": TypeConstraint(base_type=float)
            },
            description="Read and validate document files"
        )
        
    async def execute(self, input_params: Dict[str, Any]) -> Dict[str, Any]:
        file_path = input_params["file_path"]
        max_size_mb = input_params["max_size_mb"]
        
        # Create metadata
        metadata = DocumentMetadata(file_path)
        
        # Validate file
        if metadata.size > max_size_mb * 1024 * 1024:
            raise ValueError(f"File too large: {metadata.size / 1024 / 1024:.2f}MB")
            
        if not metadata.mime_type:
            raise ValueError(f"Unknown file type: {metadata.path}")
        
        # Read file content
        content = await self._read_file(metadata)
        
        return {
            "content": content,
            "metadata": {
                "path": str(metadata.path),
                "mime_type": metadata.mime_type,
                "size": metadata.size,
                "extension": metadata.extension
            }
        }
    
    async def _read_file(self, metadata: DocumentMetadata) -> str:
        """Read file content based on type."""
        if metadata.mime_type.startswith('text/'):
            async with aiofiles.open(metadata.path, 'r') as f:
                return await f.read()
        else:
            # Handle binary files
            raise NotImplementedError(f"Unsupported mime type: {metadata.mime_type}")

class ContentExtractorInstruction(Instruction):
    """Extract structured information from document content."""
    
    def __init__(self):
        super().__init__(
            id="content_extractor",
            name="content_extractor",
            type=InstructionType.LLM_INTERACTION,
            params={
                "content": TypeConstraint(base_type=str),
                "metadata": TypeConstraint(base_type=dict),
                "extraction_rules": TypeConstraint(base_type=dict)
            },
            description="Extract structured information from documents"
        )
        
    async def execute(self, input_params: Dict[str, Any]) -> Dict[str, Any]:
        content = input_params["content"]
        metadata = input_params["metadata"]
        rules = input_params["extraction_rules"]
        
        # Extract information based on rules
        extracted = {}
        for field, rule in rules.items():
            if isinstance(rule, str):
                # Regex-based extraction
                matches = re.findall(rule, content)
                extracted[field] = matches[0] if matches else None
            elif isinstance(rule, dict):
                # Custom extraction logic
                extracted[field] = await self._custom_extract(
                    content, rule, metadata
                )
        
        return {
            "extracted": extracted,
            "confidence": self._calculate_confidence(extracted)
        }
    
    async def _custom_extract(self, content: str, rule: Dict[str, Any],
                            metadata: Dict[str, Any]) -> Any:
        """Apply custom extraction rules."""
        rule_type = rule.get("type", "default")
        
        if rule_type == "llm":
            # Use LLM for extraction
            prompt = rule["prompt"].format(content=content)
            # Implement LLM extraction
            return "extracted_value"
        
        return None
    
    def _calculate_confidence(self, extracted: Dict[str, Any]) -> float:
        """Calculate confidence score for extraction."""
        non_null_fields = sum(1 for v in extracted.values() if v is not None)
        return non_null_fields / len(extracted) if extracted else 0.0

class DocumentProcessor(CompositeInstruction):
    """Complete document processing pipeline."""
    
    def __init__(self, extraction_rules: Dict[str, Any]):
        # Create component instructions
        reader = FileReaderInstruction()
        extractor = ContentExtractorInstruction()
        
        super().__init__(
            id="document_processor",
            name="document_processor",
            components=[reader, extractor],
            composition_type="sequential"
        )
        self.extraction_rules = extraction_rules
    
    async def execute(self, input_params: Dict[str, Any]) -> Dict[str, Any]:
        # Execute reader
        reader_result = await self.components[0].execute({
            "file_path": input_params["file_path"],
            "max_size_mb": input_params.get("max_size_mb", 10.0)
        })
        
        # Execute extractor
        extractor_result = await self.components[1].execute({
            "content": reader_result["content"],
            "metadata": reader_result["metadata"],
            "extraction_rules": self.extraction_rules
        })
        
        return {
            "file_info": reader_result["metadata"],
            "extracted_data": extractor_result["extracted"],
            "confidence": extractor_result["confidence"]
        }
```

### Application Setup

```python
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import uvicorn

class ExtractionRule(BaseModel):
    field: str
    rule: Dict[str, Any]

class ProcessingResult(BaseModel):
    file_info: Dict[str, Any]
    extracted_data: Dict[str, Any]
    confidence: float

class DocumentProcessingApp:
    def __init__(self):
        self.app = FastAPI(title="Smart Document Processor")
        self.agent_isa = None
        self.processor_id = None
        self.setup_routes()
    
    async def initialize(self):
        """Initialize AgentISA and document processor."""
        config = AgentISAConfig(
            enable_learning=True,
            enable_optimization=True,
            min_pattern_support=2,
            min_success_rate=0.8
        )
        self.agent_isa = AgentISA(config)
        
        # Create and register processor
        processor = DocumentProcessor(
            extraction_rules={
                "title": r"Title:\s*(.+)",
                "date": {
                    "type": "llm",
                    "prompt": "Extract the date from: {content}"
                },
                "author": {
                    "type": "llm",
                    "prompt": "Extract the author from: {content}"
                }
            }
        )
        self.processor_id = await self.agent_isa.register_instruction(processor)
    
    def setup_routes(self):
        @self.app.post("/process", response_model=ProcessingResult)
        async def process_document(file: UploadFile):
            if not self.agent_isa:
                await self.initialize()
            
            # Save uploaded file
            file_path = f"uploads/{file.filename}"
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            
            # Process document
            result = await self.agent_isa.execute_instruction(
                self.processor_id,
                {
                    "file_path": file_path,
                    "max_size_mb": 10.0
                }
            )
            
            return ProcessingResult(**result)
        
        @self.app.get("/stats")
        async def get_stats():
            if not self.agent_isa:
                await self.initialize()
            return self.agent_isa.get_instruction_stats()
    
    def run(self):
        uvicorn.run(self.app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    app = DocumentProcessingApp()
    app.run()
```

## 2. Building a Conversational Agent

Let's create a chatbot that learns from interactions and optimizes its responses.

### Core Components

```python
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Message:
    content: str
    role: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None

class ChatContext:
    """Maintains conversation context."""
    
    def __init__(self):
        self.messages: List[Message] = []
        self.user_profile: Dict[str, Any] = {}
        self.current_topic: Optional[str] = None
    
    def add_message(self, message: Message):
        self.messages.append(message)
    
    def get_recent_messages(self, count: int = 5) -> List[Message]:
        return self.messages[-count:]
    
    def update_topic(self, topic: str):
        self.current_topic = topic

class MessageProcessorInstruction(Instruction):
    """Process and understand user messages."""
    
    def __init__(self):
        super().__init__(
            id="message_processor",
            name="message_processor",
            type=InstructionType.LLM_INTERACTION,
            params={
                "message": TypeConstraint(base_type=str),
                "context": TypeConstraint(base_type=ChatContext)
            }
        )
    
    async def execute(self, input_params: Dict[str, Any]) -> Dict[str, Any]:
        message = input_params["message"]
        context = input_params["context"]
        
        # Process message
        intent = await self._detect_intent(message, context)
        entities = await self._extract_entities(message)
        sentiment = await self._analyze_sentiment(message)
        
        # Update context
        new_message = Message(
            content=message,
            role="user",
            timestamp=datetime.now()
        )
        context.add_message(new_message)
        
        return {
            "intent": intent,
            "entities": entities,
            "sentiment": sentiment,
            "requires_clarification": self._needs_clarification(message, intent)
        }
    
    async def _detect_intent(self, message: str, 
                           context: ChatContext) -> Dict[str, Any]:
        """Detect user intent."""
        # Implement intent detection
        return {"type": "query", "confidence": 0.9}
    
    async def _extract_entities(self, message: str) -> List[Dict[str, Any]]:
        """Extract named entities."""
        # Implement entity extraction
        return []
    
    async def _analyze_sentiment(self, message: str) -> Dict[str, float]:
        """Analyze message sentiment."""
        # Implement sentiment analysis
        return {"positive": 0.7, "negative": 0.1, "neutral": 0.2}

class ResponseGeneratorInstruction(Instruction):
    """Generate contextual responses."""
    
    def __init__(self):
        super().__init__(
            id="response_generator",
            name="response_generator",
            type=InstructionType.LLM_INTERACTION,
            params={
                "processed_message": TypeConstraint(base_type=dict),
                "context": TypeConstraint(base_type=ChatContext)
            }
        )
    
    async def execute(self, input_params: Dict[str, Any]) -> Dict[str, Any]:
        processed = input_params["processed_message"]
        context = input_params["context"]
        
        if processed["requires_clarification"]:
            response = await self._generate_clarification(processed)
        else:
            response = await self._generate_response(processed, context)
        
        # Update context
        new_message = Message(
            content=response["text"],
            role="assistant",
            timestamp=datetime.now(),
            metadata={"confidence": response["confidence"]}
        )
        context.add_message(new_message)
        
        return response
    
    async def _generate_clarification(self, 
                                    processed: Dict[str, Any]) -> Dict[str, Any]:
        """Generate clarification question."""
        return {
            "text": "Could you please clarify what you mean?",
            "type": "clarification",
            "confidence": 1.0
        }
    
    async def _generate_response(self, processed: Dict[str, Any],
                               context: ChatContext) -> Dict[str, Any]:
        """Generate contextual response."""
        # Implement response generation
        return {
            "text": "I understand you want to...",
            "type": "response",
            "confidence": 0.85
        }

class ChatbotAgent:
    """Main chatbot agent."""
    
    def __init__(self):
        config = AgentISAConfig(
            enable_learning=True,
            enable_optimization=True,
            min_pattern_support=2,
            min_success_rate=0.8
        )
        self.agent_isa = AgentISA(config)
        self.context = ChatContext()
        
        # Initialize instructions
        asyncio.create_task(self._init_instructions())
    
    async def _init_instructions(self):
        """Initialize and register instructions."""
        self.processor = MessageProcessorInstruction()
        self.generator = ResponseGeneratorInstruction()
        
        self.processor_id = await self.agent_isa.register_instruction(
            self.processor
        )
        self.generator_id = await self.agent_isa.register_instruction(
            self.generator
        )
    
    async def handle_message(self, message: str) -> str:
        """Process user message and generate response."""
        try:
            # Process message
            processed = await self.agent_isa.execute_instruction(
                self.processor_id,
                {
                    "message": message,
                    "context": self.context
                }
            )
            
            # Generate response
            response = await self.agent_isa.execute_instruction(
                self.generator_id,
                {
                    "processed_message": processed,
                    "context": self.context
                }
            )
            
            # Learn from interaction
            await self.agent_isa.learn_from_history()
            
            return response["text"]
            
        except Exception as e:
            logger.error(f"Error handling message: {str(e)}")
            return "I apologize, but I encountered an error processing your message."
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return self.agent_isa.get_instruction_stats()
```

### Application Setup

```python
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

class ChatApp:
    def __init__(self):
        self.app = FastAPI(title="Smart Chatbot")
        self.agent = ChatbotAgent()
        self.setup_routes()
        self.setup_middleware()
    
    def setup_middleware(self):
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def setup_routes(self):
        @self.app.websocket("/chat")
        async def chat_endpoint(websocket: WebSocket):
            await websocket.accept()
            
            try:
                while True:
                    message = await websocket.receive_text()
                    response = await self.agent.handle_message(message)
                    await websocket.send_text(response)
            except Exception as e:
                logger.error(f"WebSocket error: {str(e)}")
            finally:
                await websocket.close()
        
        @self.app.get("/stats")
        async def get_stats():
            return self.agent.get_stats()
    
    def run(self):
        uvicorn.run(self.app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    app = ChatApp()
    app.run()
```

## 3. Production Considerations

### Error Handling and Logging

```python
import logging
from logging.handlers import RotatingFileHandler

def setup_logging():
    """Configure logging for production."""
    logger = logging.getLogger("agentisa")
    logger.setLevel(logging.INFO)
    
    # File handler
    file_handler = RotatingFileHandler(
        "logs/agentisa.log",
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    
    # Stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(
        '%(levelname)s: %(message)s'
    ))
    
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    return logger

logger = setup_logging()
```

### Performance Monitoring

```python
from dataclasses import dataclass
from time import perf_counter
import psutil

@dataclass
class PerformanceMetrics:
    execution_time: float
    memory_usage: float
    cpu_usage: float
    instruction_count: int

class PerformanceMonitor:
    def __init__(self):
        self.metrics: List[PerformanceMetrics] = []
    
    async def monitor_execution(self, instruction_id: str,
                              agent_isa: AgentISA,
                              input_params: Dict[str, Any]) -> Any:
        """Monitor instruction execution."""
        start_time = perf_counter()
        start_memory = psutil.Process().memory_info().rss
        start_cpu = psutil.Process().cpu_percent()
        
        try:
            result = await agent_isa.execute_instruction(
                instruction_id,
                input_params
            )
            
            # Record metrics
            end_time = perf_counter()
            end_memory = psutil.Process().memory_info().rss
            end_cpu = psutil.Process().cpu_percent()
            
            metrics = PerformanceMetrics(
                execution_time=end_time - start_time,
                memory_usage=(end_memory - start_memory) / 1024 / 1024,  # MB
                cpu_usage=end_cpu - start_cpu,
                instruction_count=1
            )
            self.metrics.append(metrics)
            
            return result
            
        except Exception as e:
            logger.error(f"Execution error: {str(e)}")
            raise
    
    def get_average_metrics(self) -> Dict[str, float]:
        """Calculate average performance metrics."""
        if not self.metrics:
            return {}
        
        return {
            "avg_execution_time": sum(m.execution_time for m in self.metrics) / len(self.metrics),
            "avg_memory_usage": sum(m.memory_usage for m in self.metrics) / len(self.metrics),
            "avg_cpu_usage": sum(m.cpu_usage for m in self.metrics) / len(self.metrics)
        }
```

### Health Checks

```python
from fastapi import HTTPException
import aiohttp

class HealthCheck:
    def __init__(self, app: FastAPI):
        self.app = app
        self.setup_routes()
    
    def setup_routes(self):
        @self.app.get("/health")
        async def health_check():
            try:
                # Check system health
                memory_usage = psutil.Process().memory_percent()
                cpu_usage = psutil.Process().cpu_percent()
                
                if memory_usage > 90 or cpu_usage > 90:
                    raise HTTPException(
                        status_code=503,
                        detail="System resources critical"
                    )
                
                # Check external dependencies
                async with aiohttp.ClientSession() as session:
                    for dependency in self.get_dependencies():
                        try:
                            async with session.get(dependency["url"]) as response:
                                if response.status != 200:
                                    raise HTTPException(
                                        status_code=503,
                                        detail=f"Dependency {dependency['name']} unavailable"
                                    )
                        except Exception as e:
                            logger.error(f"Dependency check failed: {str(e)}")
                            raise HTTPException(
                                status_code=503,
                                detail="Dependency check failed"
                            )
                
                return {
                    "status": "healthy",
                    "memory_usage": memory_usage,
                    "cpu_usage": cpu_usage
                }
                
            except Exception as e:
                logger.error(f"Health check failed: {str(e)}")
                raise HTTPException(
                    status_code=503,
                    detail="Service unhealthy"
                )
    
    def get_dependencies(self) -> List[Dict[str, str]]:
        """Get list of dependencies to check."""
        return [
            {"name": "database", "url": "http://localhost:5432/health"},
            {"name": "cache", "url": "http://localhost:6379/health"}
        ]
```

### Deployment Configuration

```python
from pydantic import BaseSettings

class Settings(BaseSettings):
    """Application settings."""
    app_name: str = "AgentISA Application"
    debug: bool = False
    environment: str = "production"
    
    # AgentISA settings
    enable_learning: bool = True
    enable_optimization: bool = True
    min_pattern_support: int = 3
    min_success_rate: float = 0.7
    
    # Resource limits
    max_memory_mb: float = 1024.0
    max_cpu_percent: float = 80.0
    
    # External services
    database_url: str
    cache_url: str
    
    class Config:
        env_file = ".env"

def get_settings() -> Settings:
    return Settings()
```

## Next Steps

1. Implement authentication and authorization
2. Add rate limiting and request validation
3. Set up monitoring and alerting
4. Create deployment pipelines
5. Implement backup and recovery strategies

Check out the deployment guide for detailed information about deploying your application to production. 