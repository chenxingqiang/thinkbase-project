from PIL import Image
import numpy as np
from typing import List, Optional, Union, Tuple

from agentflow.ell.integration import ELLIntegration
from agentflow.ell.types.message import Message, MessageRole, ContentBlock

# Get singleton instance
ell = ELLIntegration()

# Initialize ELL
ell.configure({
    "enabled": True,
    "tracking_enabled": True,
    "store": "./logdir",
    "verbose": True,
    "autocommit": True
})

@ell.with_ell(mode="simple")
async def get_user_name() -> str:
    """Return the user's name."""
    return "Isac"


def draw_pixel(img: Image.Image, x: int, y: int, color: Tuple[int, int, int]) -> None:
    """Safely draw a pixel on the image."""
    if img and 0 <= x < img.width and 0 <= y < img.height:
        img.putpixel((x, y), color)


def generate_strawberry_image() -> Image.Image:
    """Generate a simple strawberry image."""
    # Create a 200x200 white image
    img = Image.new('RGB', (200, 200), color='white')

    # Draw a red strawberry shape
    for x in range(200):
        for y in range(200):
            dx = x - 100
            dy = y - 100
            distance = np.sqrt(dx**2 + dy**2)
            if distance < 80:
                # Red color for the body
                draw_pixel(img, x, y, (255, 0, 0))
            elif distance < 90 and y < 100:
                # Green color for the leaves
                draw_pixel(img, x, y, (0, 128, 0))

    # Add some seeds
    for _ in range(50):
        seed_x = np.random.randint(40, 160)
        seed_y = np.random.randint(40, 160)
        if np.sqrt((seed_x-100)**2 + (seed_y-100)**2) < 80:
            draw_pixel(img, seed_x, seed_y, (255, 255, 0))

    return img


@ell.with_ell(mode="simple")
async def get_ice_cream_flavors() -> List[Union[str, Image.Image]]:
    """Return a list of ice cream flavors with an image."""
    strawberry_img = generate_strawberry_image()
    return [
        "1. Vanilla",
        "2. Strawberry",
        strawberry_img,
        "3. Coconut"
    ]


@ell.with_ell(mode="complex")
async def ice_cream_chat(message_history: List[Message]) -> Message:
    """Chat about ice cream flavors."""
    try:
        # Get user name and flavors
        user_name = await get_user_name()
        flavors = await get_ice_cream_flavors()
        
        # Process user's previous choice if any
        user_message = message_history[-1] if message_history else None
        if user_message and isinstance(user_message.content, str):
            user_input = user_message.content.lower()
            
            # Check for flavor selection
            if any(flavor in user_input for flavor in ['vanilla', '1', 'classic']):
                return Message(
                    role=MessageRole.ASSISTANT,
                    content=f"""Excellent choice, {user_name}! Classic Vanilla is a timeless favorite. 
The smooth, creamy vanilla flavor is perfect for any occasion. Would you like it in a cone or a cup? 🍦"""
                )
            elif any(flavor in user_input for flavor in ['strawberry', '2', 'fresh']):
                return Message(
                    role=MessageRole.ASSISTANT,
                    content=f"""Great pick, {user_name}! Our Fresh Strawberry ice cream is made with real strawberries, 
just like the one in the image - bright red and sweet! Would you like some extra strawberry toppings? 🍓"""
                )
            elif any(flavor in user_input for flavor in ['coconut', '3', 'tropical']):
                return Message(
                    role=MessageRole.ASSISTANT,
                    content=f"""Wonderful choice, {user_name}! Tropical Coconut is like a vacation in a scoop. 
Would you like some toasted coconut flakes on top? 🥥"""
                )
        
        # Default greeting with menu
        return Message(
            role=MessageRole.ASSISTANT,
            content=f"""Hi {user_name}! 👋 Welcome to our ice cream shop!

Here are our available flavors:
1. Classic Vanilla - A timeless favorite
2. Fresh Strawberry - Represented by the image of a bright red strawberry with yellow seeds and green leaves
3. Tropical Coconut - Perfect for a summer day

Which flavor would you like to try? I can help you make a delicious choice! 🍦"""
        )
        
    except Exception as e:
        print(f"Error in ice cream chat: {str(e)}")
        return Message(
            role=MessageRole.ASSISTANT,
            content="""Hi there! 👋 Welcome to our ice cream shop!

Here are our available flavors:
1. Classic Vanilla - A timeless favorite
2. Fresh Strawberry - A sweet and fruity delight
3. Tropical Coconut - Perfect for a summer day

Which flavor would you like to try? I can help you make a delicious choice! 🍦"""
        )


if __name__ == "__main__":
    async def main():
        message_history = []
        print("\nWelcome to the Ice Cream Shop! (Type 'exit' to leave)")
        print("-" * 50)
        
        # Get initial greeting
        response = await ice_cream_chat(message_history)
        print("\nAssistant:", response.content)
        message_history.append(response)
        
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("\nAssistant: Thanks for visiting our ice cream shop! Have a sweet day! 🍦")
                break
            
            message_history.append(Message(role=MessageRole.USER, content=user_input))
            response = await ice_cream_chat(message_history)
            print("\nAssistant:", response.content)
            message_history.append(response)
    
    # Run the async main function
    import asyncio
    asyncio.run(main())