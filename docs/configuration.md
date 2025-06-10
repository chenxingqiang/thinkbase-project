# Configuration Management

## Overview

The AgentFlow configuration management system provides a flexible and secure way to manage settings across different environments. It supports multiple LLM providers and models through ell integration, combining configuration files, environment variables, and runtime settings in a unified interface.

## Directory Structure

```bash
agentflow/
├── config/
│   ├── __init__.py          # Configuration management implementation
│   ├── default.ini          # Default configuration
│   ├── development.ini      # Development environment settings
│   └── production.ini       # Production environment settings
```

## Configuration Files

### Default Configuration (default.ini)

```ini
[api_keys]
# OpenAI
openai = 

# Anthropic
anthropic = 

# Mistral
mistral = 

# AI21
ai21 = 

# Amazon Bedrock
aws_access_key_id = 
aws_secret_access_key = 
aws_region = 

# Cohere
cohere = 

[model_settings]
default_model = claude-3-haiku-20240307
temperature = 0.7
max_tokens = 1000

[available_models]
# OpenAI Models
openai_models = [
    "gpt-4-1106-preview",
    "gpt-4-32k-0314",
    "gpt-4-0125-preview",
    "gpt-4-turbo-preview",
    "gpt-4",
    "gpt-4-0314",
    "gpt-4-0613",
    "gpt-4-turbo",
    "gpt-4-turbo-2024-04-09",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0301",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-16k",
    "gpt-3.5-turbo-16k-0613",
    "gpt-3.5-turbo-0125",
    "gpt-3.5-turbo-1106"
]

# Anthropic Models
anthropic_models = [
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
    "claude-3-5-sonnet-20240620",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-sonnet-latest"
]

# Mistral Models
mistral_models = [
    "mistral.mistral-7b-instruct-v0:2",
    "mistral.mixtral-8x7b-instruct-v0:1",
    "mistral.mistral-large-2402-v1:0",
    "mistral.mistral-small-2402-v1:0"
]

# AI21 Models
ai21_models = [
    "ai21.jamba-instruct-v1:0",
    "ai21.j2-ultra-v1",
    "ai21.j2-mid-v1"
]

# Amazon Titan Models
amazon_models = [
    "amazon.titan-text-lite-v1",
    "amazon.titan-text-express-v1",
    "amazon.titan-embed-text-v1",
    "amazon.titan-image-generator-v1",
    "amazon.titan-image-generator-v2:0"
]

# Cohere Models
cohere_models = [
    "cohere.command-r-plus-v1:0",
    "cohere.command-r-v1:0",
    "cohere.command-text-v14",
    "cohere.embed-english-v3",
    "cohere.embed-multilingual-v3"
]

# Meta Llama Models
meta_models = [
    "meta.llama3-8b-instruct-v1:0",
    "meta.llama3-70b-instruct-v1:0",
    "meta.llama2-13b-chat-v1",
    "meta.llama2-70b-chat-v1"
]
```

## Model-Specific Configuration

### Model Parameters

```ini
[model_parameters]
# Common parameters
temperature = 0.7
max_tokens = 1000
top_p = 1.0
frequency_penalty = 0.0
presence_penalty = 0.0

# Model-specific parameters
[model_parameters.gpt-4]
temperature = 0.7
max_tokens = 8192

[model_parameters.claude-3-opus]
temperature = 0.7
max_tokens = 4096

[model_parameters.mistral-large]
temperature = 0.7
max_tokens = 4096
```

### Rate Limits

```ini
[rate_limits]
max_retries = 3
retry_delay = 1
requests_per_minute = 60

# Provider-specific rate limits
[rate_limits.openai]
requests_per_minute = 60
tokens_per_minute = 90000

[rate_limits.anthropic]
requests_per_minute = 50
tokens_per_minute = 100000

[rate_limits.mistral]
requests_per_minute = 40
tokens_per_minute = 80000
```

### Model Fallbacks

```ini
[model_fallbacks]
# Fallback chain for each provider
openai = ["gpt-4", "gpt-3.5-turbo"]
anthropic = ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"]
mistral = ["mistral.mistral-large-2402-v1:0", "mistral.mistral-small-2402-v1:0"]

[provider_priorities]
# Order of provider preference
order = ["anthropic", "openai", "mistral", "ai21", "amazon", "cohere", "meta"]
```

## Usage

### Basic Usage

```python
from agentflow.config import config

# Get all available models
all_models = config.get_available_models()

# Get models for specific provider
anthropic_models = config.get_available_models('anthropic')

# Get model parameters
model_params = config.get_model_parameters('claude-3-opus-20240229')

# Get provider rate limits
rate_limits = config.get_provider_rate_limits('openai')

# Get fallback models
fallbacks = config.get_fallback_models('anthropic')

# Get provider priority
priority = config.get_provider_priority()
```

### Environment Variables

```bash
# Set environment
export AGENTFLOW_ENV=development  # or production

# API Keys
export ANTHROPIC_API_KEY=your-key-here
export OPENAI_API_KEY=your-openai-key
export MISTRAL_API_KEY=your-mistral-key
export AI21_API_KEY=your-ai21-key
export AWS_ACCESS_KEY_ID=your-aws-key
export AWS_SECRET_ACCESS_KEY=your-aws-secret
export AWS_REGION=your-aws-region
export COHERE_API_KEY=your-cohere-key
```

### Integration with ell

```python
from agentflow.config import config

@ell.simple(
    model=config.get_model_settings().get('default_model'),
    temperature=float(config.get_model_settings().get('temperature', 0.7))
)
def your_llm_function():
    pass
```

## Model Selection Strategy

### Priority and Fallbacks

The system uses a sophisticated model selection strategy:

1. Provider Priority:
   - Follows the order specified in `provider_priorities`
   - Attempts to use the preferred provider first

2. Model Fallbacks:
   - If primary model is unavailable, tries fallback models in order
   - Supports provider-specific fallback chains

3. Rate Limiting:
   - Respects provider-specific rate limits
   - Automatically switches to fallback when limits are reached

### Example Selection Flow

```python
# Configuration-based model selection
primary_model = config.get_model_settings().get('default_model')
fallbacks = config.get_fallback_models(provider)

try:
    result = use_model(primary_model)
except RateLimitError:
    for fallback_model in fallbacks:
        try:
            result = use_model(fallback_model)
            break
        except:
            continue
```

## Best Practices

### Model Configuration

1. Set appropriate fallback chains for each provider
2. Configure rate limits based on your API tier
3. Adjust model parameters for your use case
4. Monitor token usage across providers

### Security

1. Never commit API keys to version control
2. Use environment variables for sensitive data
3. Regularly rotate API keys
4. Monitor API usage and costs

### Testing

1. Create test configurations for each provider
2. Mock API responses in tests
3. Test fallback scenarios
4. Verify rate limit handling

## Troubleshooting

### Common Issues

1. Model Not Available:

```python
ValueError: "Model not available in current configuration"
```

Solution: Check model name and provider configuration

2. Rate Limit Exceeded:

```python
RateLimitError: "Provider rate limit exceeded"
```

Solution: Check rate limit settings and implement backoff

## References

- [ell Documentation](https://docs.ell.so/)
- [OpenAI API](https://platform.openai.com/docs/api-reference)
- [Anthropic API](https://docs.anthropic.com/claude/reference)
- [Mistral AI](https://docs.mistral.ai/)
- [AI21 Labs](https://docs.ai21.com/)
- [Amazon Bedrock](https://docs.aws.amazon.com/bedrock)
- [Cohere API](https://docs.cohere.com/)
