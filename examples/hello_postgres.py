import numpy as np
from agentflow.ell_integration import ELLIntegration
from agentflow.ell.stores.sql import PostgresStore

# Get singleton instance
ell = ELLIntegration()

class MyPrompt:
    x : int

def get_random_length():
    return int(np.random.beta(2, 6) * 1500)

@ell.with_ell(mode="simple")
async def hello(world: str) -> str:
    """Your goal is to be really mean to the other guy while saying hello"""
    name = world.capitalize()
    number_of_chars_in_name = get_random_length()

    return f"Say hello to {name} in {number_of_chars_in_name} characters or more!"

if __name__ == "__main__":
    # Initialize ELL with PostgreSQL store
    ell.configure({
        "enabled": True,
        "tracking_enabled": True,
        "store": PostgresStore('postgresql://postgres:postgres@localhost:5432/ell'),
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

    # Test the hello function
    import asyncio
    greeting = asyncio.run(hello("sam altman"))
    print("Greeting:", greeting)
