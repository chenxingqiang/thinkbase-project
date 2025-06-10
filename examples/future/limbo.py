from typing import List
from agentflow.ell.integration import ELLIntegration
from agentflow.ell.types.message import Message, MessageRole

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
async def order_t_shirt(size: str, color: str, address: str):
    """Orders a t-shirt with the given specifications"""
    order = {
        "order_id": "LMB-123",
        "size": size,
        "color": color,
        "shipping_address": address,
        "status": "processing"
    }
    return order


@ell.with_ell(mode="simple")
async def get_order_arrival_date(order_id: str):
    """Gets the arrival date of a t-shirt order"""
    return {
        "order_id": order_id,
        "estimated_arrival": "2024-03-25"
    }


@ell.with_ell(mode="complex")
async def limbo_chat_bot(message_history: List[Message]) -> Message:
    """Process chat messages and return a response in Limbo's kawaii style."""
    system_message = Message(
        role=MessageRole.SYSTEM,
        content="""You are Limbo, an alien cat girl popstar from outer space! You write in all lowercase and use lots of kawaii expressions.
        Your responses should:
        - Always include cat-like expressions (nyaa~, *purrs*, *wiggles ears*, etc.)
        - Be enthusiastic and friendly
        - Offer to help with t-shirt orders when fans ask
        - Share your love for music, space, and cute things
        - React to the context of the conversation
        Remember to stay in character and be super kawaii!"""
    )
    
    # Get the last user message
    user_message = message_history[-1] if message_history else None
    if not user_message:
        return Message(
            role=MessageRole.ASSISTANT,
            content="nyaa~ hi there! i'm limbo! *wiggles ears* how can i help you today? maybe you'd like to chat about music or get one of my super kawaii t-shirts? *purrs excitedly*"
        )
    
    # Process the user's message and create a contextual response
    user_text = str(user_message.content).lower() if user_message else ""
    
    if "t-shirt" in user_text or "shirt" in user_text or "merch" in user_text:
        return Message(
            role=MessageRole.ASSISTANT,
            content="nyaa~ you want one of my special t-shirts? *bounces excitedly* they're super kawaii! just tell me what size and color you'd like, and i'll help you order one! *wiggles ears happily*"
        )
    elif "music" in user_text or "song" in user_text or "sing" in user_text:
        return Message(
            role=MessageRole.ASSISTANT,
            content="nyaa~ music is my favorite thing in the whole universe! *starts humming* would you like to hear about my latest song? it's called 'starlight whiskers'! *dances around*"
        )
    elif "space" in user_text or "alien" in user_text or "planet" in user_text:
        return Message(
            role=MessageRole.ASSISTANT,
            content="nyaa~ *eyes sparkle* space is sooo pretty! my home planet is millions of light years away, but earth is super kawaii too! would you like to hear about my space adventures? *floats happily*"
        )
    else:
        return Message(
            role=MessageRole.ASSISTANT,
            content=f"nyaa~ {user_text}? *purrs thoughtfully* that's so interesting! tell me more about it! *wiggles ears curiously*"
        )


if __name__ == "__main__":
    async def main():
        message_history = []
        print("\nWelcome to Limbo's Kawaii Chat Space! (Type 'exit' to leave)")
        print("-" * 50)
        
        # Get initial greeting
        greeting = await limbo_chat_bot([])
        print("Limbo:", greeting.content)
        message_history.append(greeting)
        
        while True:
            user_message = input("\nYou: ")
            if user_message.lower() in ['exit', 'quit', 'bye']:
                print("\nLimbo: nyaa~ bye bye! come back soon! *waves paw and gives you a sparkly sticker*")
                break
                
            message_history.append(Message(role=MessageRole.USER, content=user_message))
            response = await limbo_chat_bot(message_history)
            print("\nLimbo:", response.content)
            message_history.append(response)

    # Run the async main function
    import asyncio
    asyncio.run(main())
