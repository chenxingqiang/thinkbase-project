from agentflow.ell.integration import ELLIntegration
from agentflow.ell.types.message import Message, MessageRole
from typing import List
import asyncio

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
    "mode": "simple",
    "metadata": {
        "type": "text",
        "format": "plain"
    }
})

@ell.with_ell(mode="simple")
async def get_user_name() -> str:
    """Return the user's name."""
    return "Isac"

@ell.with_ell(mode="simple")
async def get_ice_cream_flavors() -> List[str]:
    """Return a list of ice cream flavors."""
    return ["Vanilla", "Strawberry", "Coconut"]

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
            
            # Check for serving style choices
            if any(word in user_input for word in ['cone', 'cup', 'bowl']):
                serving = 'cone' if 'cone' in user_input else 'cup'
                return Message(
                    role=MessageRole.ASSISTANT,
                    content=f"""Perfect! I'll serve your ice cream in a {serving}. Your order is ready! 
Enjoy your delicious treat, {user_name}! 🍦✨"""
                )
            
            # Check for topping choices
            if any(word in user_input for word in ['topping', 'sprinkle', 'yes', 'sure', 'yeah']):
                # Check recent message history for flavor context
                recent_content = []
                for msg in message_history[-3:]:
                    if isinstance(msg.content, str):
                        recent_content.append(msg.content.lower())
                recent_text = ' '.join(recent_content)
                
                if 'strawberry' in recent_text:
                    return Message(
                        role=MessageRole.ASSISTANT,
                        content=f"""Wonderful! I'll add some fresh strawberry pieces on top. 
Would you like that in a cone or a cup? 🍓"""
                    )
                elif 'coconut' in recent_text:
                    return Message(
                        role=MessageRole.ASSISTANT,
                        content=f"""Great! I'll sprinkle some toasted coconut flakes on top. 
Would you like that in a cone or a cup? 🥥"""
                    )
            
            # Check for flavor selection
            if any(flavor in user_input for flavor in ['vanilla', '1', 'classic']):
                return Message(
                    role=MessageRole.ASSISTANT,
                    content=f"""Excellent choice, {user_name}! Classic Vanilla is a timeless favorite. 
Would you like it in a cone or a cup? 🍦"""
                )
            elif any(flavor in user_input for flavor in ['strawberry', '2', 'fresh']):
                return Message(
                    role=MessageRole.ASSISTANT,
                    content=f"""Great pick, {user_name}! Our Fresh Strawberry ice cream is made with real strawberries. 
Would you like some extra strawberry pieces on top? 🍓"""
                )
            elif any(flavor in user_input for flavor in ['coconut', '3', 'tropical']):
                return Message(
                    role=MessageRole.ASSISTANT,
                    content=f"""Wonderful choice, {user_name}! Tropical Coconut is like a vacation in a scoop. 
Would you like some toasted coconut flakes on top? 🥥"""
                )
            
            # Handle no/decline for toppings
            if any(word in user_input for word in ['no', 'nope', 'skip']):
                return Message(
                    role=MessageRole.ASSISTANT,
                    content=f"""No problem! Would you like your ice cream in a cone or a cup? 🍦"""
                )
        
        # Default greeting with menu
        return Message(
            role=MessageRole.ASSISTANT,
            content=f"""Hi {user_name}! 👋 Welcome to our ice cream shop!

Here are our available flavors:
1. Classic Vanilla - A timeless favorite
2. Fresh Strawberry - Made with real strawberries
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

if __name__ == "__main__":
    asyncio.run(main())