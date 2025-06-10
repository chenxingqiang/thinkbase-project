from flask import Flask
from agentflow.ell.integration import ELLIntegration
import os
import openai
import asyncio
from functools import wraps
from asgiref.sync import async_to_sync

# Get OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# Create OpenAI client
client = openai.OpenAI(api_key=api_key)

# Get singleton instance
ell = ELLIntegration()

# Initialize ELL
ell.configure({
    "enabled": True,
    "tracking_enabled": True,
    "store": "./logdir",
    "verbose": True,
    "autocommit": True,
    "model": "gpt-4",
    "default_model": "gpt-4",
    "temperature": 0.7,
    "mode": "simple",
    "metadata": {
        "type": "text",
        "format": "plain"
    }
})

app = Flask(__name__)

def async_route(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        return async_to_sync(f)(*args, **kwargs)
    return wrapped

@ell.with_ell(mode="simple")
async def generate_welcome(name: str) -> str:
    """Generate a welcome message for the given name."""
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are a friendly assistant that generates welcoming messages. Keep responses brief and cheerful."
                },
                {
                    "role": "user",
                    "content": f"Write a welcome message for {name}."
                }
            ],
            temperature=0.7
        )
        
        if not response or not response.choices or not response.choices[0].message:
            return f"Welcome, {name}!"
            
        content = response.choices[0].message.content
        if not content:
            return f"Welcome, {name}!"
            
        return content.strip()
        
    except Exception as e:
        print(f"Error generating welcome message: {str(e)}")
        return f"Welcome, {name}!"

@app.route('/')
@async_route
async def home():
    return await generate_welcome("world")

@app.route('/<name>')
@async_route
async def welcome_user(name):
    return await generate_welcome(name)

if __name__ == '__main__':
    print("\nStarting server on http://localhost:5000")
    print("Available routes:")
    print("  - /           -> Welcome message for 'world'")
    print("  - /<name>     -> Welcome message for the specified name")
    print("\nPress Ctrl+C to stop the server.")
    app.run(debug=True, port=5000)
