import random
from agentflow.ell_integration import ELLIntegration

# Get singleton instance
ell = ELLIntegration()

# Initialize ELL
ell.configure({
    "enabled": True,
    "tracking_enabled": True,
    "store": "./ell_logs",
    "verbose": True,
    "autocommit": True,
    "model": "gpt-4",
    "default_model": "gpt-4",
    "temperature": 0.1,
    "mode": "simple",
    "metadata": {
        "type": "text",
        "format": "plain"
    }
})

def get_random_adjective():
    adjectives = ["enthusiastic", "cheerful", "warm", "friendly", "heartfelt", "sincere"]
    return random.choice(adjectives)

def get_random_punctuation():
    return random.choice(["!", "!!", "!!!"])

@ell.with_ell(mode="simple")
async def hello(name: str) -> str:
    """You are a helpful and expressive assistant."""
    adjective = get_random_adjective()
    punctuation = get_random_punctuation()
    return f"Say a {adjective} hello to {name}{punctuation}"

if __name__ == "__main__":
    import asyncio
    greeting = asyncio.run(hello("Sam Altman"))
    print(greeting)