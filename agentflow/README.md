# AgentFlow Directory Structure

## Overview

AgentFlow is organized into several key modules and packages that provide different functionalities for AI agent workflow management.

## Directory Structure

```
agentflow/
├── agents/                 # Agent implementations
│   ├── agent.py           # Base agent class
│   └── agent_types.py     # Agent type definitions
│
├── api/                   # API endpoints and interfaces
│   └── workflow_api.py    # Workflow API endpoints
│
├── applications/          # Application-specific implementations
│   └── research/         # Research application modules
│
├── configs/              # Configuration management
│   └── default/         # Default configuration files
│
├── core/                 # Core framework components
│   ├── base_types.py    # Base type definitions
│   ├── config.py        # Configuration management
│   ├── workflow.py      # Workflow engine
│   └── workflow_executor.py # Workflow execution
│
├── ell/               # ELL integration
│   ├── integration.py   # Integration module
│   └── types/          # Message and content types
│
├── models/              # Model implementations
│   └── transformers/   # Transformer models
│
├── monitoring/          # System monitoring
│   └── metrics.py      # Metrics collection
│
├── services/           # Service implementations
│   └── registry.py    # Service registry
│
├── strategies/         # Strategy implementations
│   └── transform/     # Transform strategies
│
├── transformations/    # Data transformation tools
│   └── text.py        # Text processing utilities
│
├── __init__.py        # Package initialization
├── errors.py          # Error definitions
├── monitoring.py      # Monitoring utilities
├── services.py        # Service utilities
└── version.py         # Version information
```

## Module Descriptions

### Core Modules

1. **agents/**
   - Base agent implementations
   - Agent type definitions and factories
   - Agent state management

2. **core/**
   - Framework fundamentals
   - Workflow engine and executor
   - Configuration management
   - Base type definitions

3. **ell/**
   - ELL integration components
   - Message type definitions
   - Communication protocols

### Data Processing

4. **transformations/**
   - Data transformation tools
   - Feature engineering utilities
   - Text processing functions

5. **models/**
   - Model implementations
   - Transformer architectures
   - Model utilities

### Infrastructure

6. **api/**
   - REST API endpoints
   - API utilities
   - Interface definitions

7. **services/**
   - Service implementations
   - Service registry
   - Service utilities

8. **monitoring/**
   - System monitoring
   - Metrics collection
   - Performance tracking

### Configuration & Management

9. **configs/**
   - Configuration files
   - Default settings
   - Environment configurations

10. **strategies/**
    - Strategy implementations
    - Transform strategies
    - Execution strategies

### Applications

11. **applications/**
    - Application-specific code
    - Research applications
    - Custom implementations

## Key Files

- **__init__.py**: Package initialization and exports
- **errors.py**: Error definitions and handling
- **monitoring.py**: Monitoring utilities
- **services.py**: Service utilities
- **version.py**: Version information

## Version Information

Current version: v0.1.1
- Enhanced workflow transform functions
- Added new transformation tools
- Improved testing and documentation

## Usage

```python
from agentflow import Agent, AgentConfig, WorkflowConfig
from agentflow.transformations import feature_engineering_transform

# Create and configure agent
agent = Agent(
    config=AgentConfig(
        name="example_agent",
        type="data_science",
        workflow=WorkflowConfig(...)
    )
)

# Execute workflow
result = await agent.execute({"data": your_data})
```

## Development

1. Core Development:
   - Work in core/ for framework changes
   - Update tests in tests/
   - Follow type hints and documentation

2. Adding Features:
   - Add new modules in appropriate directories
   - Update __init__.py for exports
   - Add tests and documentation

3. Testing:
   - Run unit tests: `pytest tests/unit/`
   - Run integration tests: `pytest tests/integration/`
   - Run performance tests: `pytest tests/performance/` 