from typing import List, Union
from pydantic import BaseModel, Field
from agentflow.ell.integration import ELLIntegration
from agentflow.ell.types.message import Message, MessageRole, MessageType, ContentBlock
from agentflow.ell.stores import SQLStore

# Get singleton instance
ell = ELLIntegration()

# Initialize ELL
ell.configure({
    "enabled": True,
    "tracking_enabled": True,
    "store": "./logdir",
    "verbose": True,
    "autocommit": True,
    "model": "claude-3-5-sonnet-20241022",
    "default_model": "claude-3-5-sonnet-20241022",
    "temperature": 0.1,
    "max_tokens": 400,
    "metadata": {
        "type": "text",
        "format": "json"
    }
})

@ell.with_ell()
async def create_claim_draft(claim_details: str, claim_type: str, claim_amount: float, 
                       claim_date : str = Field(description="The date of the claim in the format YYYY-MM-DD.")) -> str:
    """Create a claim draft. Returns the claim id created."""
    print("Create claim draft", claim_details, claim_type, claim_amount, claim_date)
    return "claim_id-123234"

@ell.with_ell()
async def approve_claim(claim_id : str) -> str:
    """Approve a claim"""
    return "approved"

@ell.with_ell()
async def insurance_claim_chatbot(message_history: List[Message]) -> Message:
    """Process insurance claim chat messages."""
    # Create system message
    system_message = Message(
        role=MessageRole.SYSTEM,
        content="""You are a an insurance adjuster AI. You are given a dialogue with a user and have access to various tools to effectuate the insurance claim adjustment process. Ask question until you have enough information to create a claim draft. Then ask for approval.""",
        type=MessageType.TEXT
    )
    
    # Process the message
    response = await ell.process_message(system_message)
    
    # Return the response
    return response

def create_claim_draft_response(details: str) -> str:
    """Create a claim draft based on the provided details."""
    # Create a message with the claim details
    message = Message(
        role=MessageRole.USER,
        content=details,
        type=MessageType.TEXT,
        metadata={
            "type": "claim_draft",
            "status": "pending"
        }
    )
    
    # Process the message and generate a response
    response = Message(
        role=MessageRole.ASSISTANT,
        content=f"I'll help you file an insurance claim based on these details:\n{details}\n\nClaim ID: CLM-2024-001",
        type=MessageType.TEXT,
        metadata={
            "type": "claim_draft",
            "status": "created",
            "claim_id": "CLM-2024-001"
        }
    )
    
    return str(response.content)

def process_insurance_claim(message_history: List[Message]) -> str:
    """Process messages in the insurance claim chatbot."""
    # Create system message if not present
    system_message = Message(
        role=MessageRole.SYSTEM,
        content="You are an insurance claim assistant. Help customers file claims and provide relevant information.",
        type=MessageType.TEXT
    )
    
    if not message_history or message_history[0].role != MessageRole.SYSTEM:
        message_history.insert(0, system_message)
    
    # Get the last user message
    user_message = next((msg for msg in reversed(message_history) if msg.role == MessageRole.USER), None)
    if not user_message:
        return "How can I help you with your insurance claim today?"
    
    # Process the message based on content
    user_content = str(user_message.content)
    if "file" in user_content.lower():
        # Extract claim details from previous messages
        claim_details = []
        for msg in message_history:
            if msg.role == MessageRole.USER and msg != user_message:
                claim_details.append(str(msg.content))
        
        # Create claim draft with accumulated details
        return create_claim_draft_response("\n".join(claim_details))
    else:
        # Generate a helpful response
        response = Message(
            role=MessageRole.ASSISTANT,
            content=f"I understand you're saying: {user_content}\nPlease provide more details about your claim, such as:\n- What happened?\n- When did it occur?\n- Estimated damage cost?",
            type=MessageType.TEXT
        )
        return str(response.content)

if __name__ == "__main__":
    # Initialize message history
    message_history: List[Message] = []
    
    # Main chat loop
    while True:
        # Get user input
        user_input = input("\nUser: ")
        if not user_input:
            break
        
        # Create user message
        user_message = Message(
            role=MessageRole.USER,
            content=user_input,
            type=MessageType.TEXT
        )
        message_history.append(user_message)
        
        # Process message and get response
        response = process_insurance_claim(message_history)
        
        # Create assistant message
        assistant_message = Message(
            role=MessageRole.ASSISTANT,
            content=response,
            type=MessageType.TEXT
        )
        message_history.append(assistant_message)
        
        # Print response
        print(f"Assistant: {response}")
