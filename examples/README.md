# DeepSeek GitHub Agent

An AI-powered agent for GitHub projects, built with the DeepSeek language model. This agent automates various aspects of software development, from code generation to Git operations, following Test-Driven Development (TDD) principles.

## Features

- **Repository Analysis**: Analyzes repository structure, dependencies, and development patterns
- **Requirements Analysis**: Processes and analyzes project requirements
- **Test-Driven Development**: Designs and generates test cases before implementation
- **Code Generation**: Generates code based on test specifications
- **Code Review**: Reviews code for quality, style, and potential issues
- **Git Operations**: Automates Git operations like commits, pushes, and pull requests
- **GitHub Integration**: Seamless integration with GitHub repositories

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/deepseek-github-agent.git
cd deepseek-github-agent
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure the agent:
- Copy `examples/deepseek_github_agent_config.json` to your project
- Update the configuration with your GitHub credentials and preferences
- Set your DeepSeek API key in the configuration

## Usage

### Basic Usage

```python
from agentflow.core.memory import WorkingMemory
from agentflow.providers.deepseek import DeepSeekProvider
from examples.deepseek_github_agent import DeepSeekGitHubAgent

# Initialize components
memory = WorkingMemory()
provider = DeepSeekProvider(api_key="your-api-key")

# Create agent
agent = DeepSeekGitHubAgent(
    provider=provider,
    memory=memory,
    workspace="path/to/workspace",
    github_config={
        "username": "your-username",
        "repository_url": "https://github.com/your/repo.git",
        "default_branch": "main",
        "commit_style": "conventional",
        "auto_push": False,
        "create_pr": False
    }
)

# Initialize agent
await agent.initialize()

# Use agent features
analysis = await agent.analyze_repository()
test_cases = await agent.design_tests(analysis)
code = await agent.implement_code(test_cases)
review = await agent.review_code(code)

# Commit changes
commit_hash = await agent.commit_changes(
    files=["new_file.py"],
    message="feat: add new feature",
    scope="core"
)
```

### Configuration

The agent can be configured through a JSON configuration file. Key configuration options include:

- **Provider Settings**: Model selection, temperature, and API configuration
- **GitHub Settings**: Repository settings, commit style, and automation preferences
- **Instruction Settings**: Configuration for each instruction type (analysis, testing, etc.)
- **Metrics Settings**: Enable/disable various metrics tracking

See `examples/deepseek_github_agent_config.json` for a complete configuration example.

## Components

### Git Handler

The `GitHandler` class provides Git operations functionality:

- Repository initialization
- File staging
- Commit creation
- Branch management
- Push operations
- Pull request creation

### Instructions

The agent supports various instruction types:

- `RepositoryAnalysisInstruction`: Analyzes repository structure
- `RequirementAnalysisInstruction`: Analyzes requirements
- `TestDesignInstruction`: Designs test cases
- `CodeImplementationInstruction`: Implements code
- `CodeReviewInstruction`: Reviews code
- `GitOperationsInstruction`: Handles Git operations

## Development

### Running Tests

```bash
pytest tests/
```

### Adding New Features

1. Create new instruction types in `deepseek_instructions.py`
2. Implement the instruction logic
3. Register the instruction in `INSTRUCTION_REGISTRY`
4. Add corresponding methods to `DeepSeekGitHubAgent`
5. Add tests in `tests/test_deepseek_github_agent.py`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a pull request

## License

MIT License - see LICENSE file for details
