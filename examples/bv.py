from functools import lru_cache
from agentflow.ell.integration import ELLIntegration
from agentflow.ell.types.message import Message, MessageRole
from agentflow.ell.stores import SQLStore

# Get singleton instance
ell = ELLIntegration()

CODE_INSTURCTIONS = """

Other Instructions:
- You only respond in code without any commentary (except in the docstrings.) 
- Don't respond in markdown just write code!
- It is extremely important that you don't start you code with ```python <...> """


class Tests:
    pass


test = Tests()

another_serializeable_global = ["asd"]


def get_lmp(z=10):
    y = 13
    y = z

    @ell.with_ell(mode="complex")
    async def write_a_complete_python_class(user_spec: str) -> str:
        """Write a complete Python class based on user specification."""
        # Create system message
        system_message = Message(
            role=MessageRole.SYSTEM,
            content="""You are a Python programmer who writes clean, well-documented code. You should:
1. Write code with proper docstrings and type hints
2. Include appropriate methods and attributes
3. Follow PEP 8 style guidelines
4. Only output the code, no additional text or markdown""",
            metadata={
                "type": "text",
                "format": "code"
            }
        )
        
        # Create user message
        user_message = Message(
            role=MessageRole.USER,
            content=f"""Create a Python class that represents {user_spec}. The class should include:
1. Appropriate instance variables with type hints
2. A constructor with proper parameter validation
3. Methods for common operations
4. Property decorators where appropriate
5. Comprehensive docstrings for the class and all methods""",
            metadata={
                "type": "text",
                "format": "code"
            }
        )
        
        # Process system message
        await ell.process_message(system_message)
        
        # Process user message and get response
        response = await ell.process_message(user_message)
        
        # Return the response
        if isinstance(response, Message):
            content = response.content
            if isinstance(content, list):
                content = content[0] if content else ""
            return str(content)
        elif isinstance(response, dict):
            content = response.get("content", "")
            if isinstance(content, list):
                content = content[0] if content else ""
            return str(content)
        else:
            return str(response)

    return write_a_complete_python_class


if __name__ == "__main__":
    # Initialize ELL
    ell.configure({
        "enabled": True,
        "tracking_enabled": True,
        "store": "./logdir",
        "verbose": True,
        "autocommit": True,
        "model": "gpt-4",
        "default_model": "gpt-4",
        "temperature": 0.1,
        "max_tokens": 1000
    })
    
    # Run the example
    import asyncio
    w = get_lmp(z=20)
    bank_spec = """a bank with the following features:
- Account management (create, close, get balance)
- Transaction handling (deposit, withdraw, transfer)
- Interest calculation for savings accounts
- Account types (checking, savings)
- Transaction history tracking"""
    
    cls_def = asyncio.run(w(bank_spec))
    print("\nGenerated Bank Class:")
    print("-" * 50)
    print(cls_def)
    print("-" * 50)
