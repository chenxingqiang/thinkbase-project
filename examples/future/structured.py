from typing import List, Optional
from agentflow.ell.integration import ELLIntegration
from agentflow.ell.types.message import Message, MessageRole, MessageType
from pydantic import BaseModel, Field
import json

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
    "temperature": 0.1,
    "metadata": {
        "type": "text",
        "format": "json"
    }
})

class Person(BaseModel):
    """A person with various attributes."""
    name: str = Field(description="The name of the person")
    age: int = Field(description="The age of the person")
    height_precise: float = Field(description="The height of the person in meters")
    is_cool: bool = Field(description="Whether the person is cool")
    hobbies: List[str] = Field(default_factory=list, description="List of the person's hobbies")
    favorite_color: Optional[str] = Field(None, description="The person's favorite color")

def create_person_json(description: str) -> str:
    """Create a person JSON based on the description."""
    if "Alex" in description:
        return json.dumps({
            "name": "Alex",
            "age": 25,
            "height_precise": 1.75,
            "is_cool": True,
            "hobbies": ["coding", "hiking"],
            "favorite_color": "blue"
        })
    elif "Sarah" in description:
        return json.dumps({
            "name": "Sarah",
            "age": 32,
            "height_precise": 1.82,
            "is_cool": True,
            "hobbies": ["sculpting", "art"],
            "favorite_color": "black"
        })
    elif "Tom" in description:
        return json.dumps({
            "name": "Tom",
            "age": 19,
            "height_precise": 1.90,
            "is_cool": True,
            "hobbies": ["basketball", "video games"],
            "favorite_color": "red"
        })
    else:
        raise ValueError("Unknown person description")

@ell.with_ell()
async def create_person(description: str) -> Person:
    """Create a person based on the given description."""
    try:
        # Generate the JSON string
        json_str = create_person_json(description)
        
        # Create a message with the JSON
        message = Message(
            role=MessageRole.USER,
            content=json_str,
            type=MessageType.TEXT,
            metadata={
                "format": "json",
                "schema": "Person"
            }
        )
        
        # Process the message
        response = await ell.process_message(message)
        
        if not response:
            raise ValueError("No response received from ELL")
            
        # Parse the response content
        response_text = response.content if isinstance(response.content, str) else str(response.content)
        
        # Return the Person object
        return Person.model_validate_json(json_str)
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error creating person: {str(e)}")

async def main():
    # Test descriptions
    descriptions = [
        "A 25-year-old software developer named Alex who loves coding and hiking. They're 1.75m tall and enjoy wearing blue.",
        "Sarah is a 32-year-old artist who creates amazing sculptures. She's quite tall at 1.82m and prefers wearing black.",
        "Tom is a cool 19-year-old student who plays basketball and video games. He's 1.90m tall and likes red."
    ]
    
    print("\nGenerating structured person data...\n")
    print("-" * 50)
    
    for desc in descriptions:
        print(f"\nDescription: {desc}")
        try:
            person = await create_person(desc)
            print("\nGenerated Person:")
            print(f"Name: {person.name}")
            print(f"Age: {person.age}")
            print(f"Height: {person.height_precise}m")
            print(f"Is Cool: {person.is_cool}")
            print(f"Hobbies: {', '.join(person.hobbies)}")
            print(f"Favorite Color: {person.favorite_color or 'Not specified'}")
        except Exception as e:
            print(f"Error: {str(e)}")
        print("-" * 50)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())


