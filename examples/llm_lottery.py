#!/usr/bin/env python3

# Tip: Most fun when run with `-v` or `-vv` flag!
#
# Small educational example on how to use tools.
# The example provides LLM with two tools:
# * One to buy a lottery ticket.
# * Another to check a lottery ticket.
# 
# Then we run LLM to enjoy watching how it plays the lottery.
#
# Additionally, it provides:
# `loop_llm_and_tools` - an example of a handy boilerplate function
#                        that helps to run tools until the task is completed.
# `main` with argparse and -v|--verbose flag -
#        An example of how to make simple code convenient to switch between
#        different verbosity levels from the command line, e.g. for this script:
#        * `-v`  prints progress information to stderr.
#        * `-vv` also enables verbosity for ell.init.

from agentflow.ell import ELL
from agentflow.ell.types.message import Message, MessageRole
from agentflow.ell.integration import ELLIntegration
import argparse
import sys
from typing import List
from pydantic import Field
import openai
import os
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from openai.types.chat.chat_completion_system_message_param import ChatCompletionSystemMessageParam
from openai.types.chat.chat_completion_user_message_param import ChatCompletionUserMessageParam
from openai.types.chat.chat_completion_assistant_message_param import ChatCompletionAssistantMessageParam

VERBOSE = False
WINNING_NUMBER = 12  # Hardcoded winning number

# Get singleton instance
ell = ELLIntegration()

# Get OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# Create OpenAI client
client = openai.OpenAI(api_key=api_key)

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

@ell.with_ell(mode="simple")
async def buy_lottery_ticket(number: int = Field(description="Number to play in the lottery (0-16).")) -> str:
    """Buy a lottery ticket with the given number."""
    if VERBOSE:
        sys.stderr.write(f"Calling tool: buy_lottery_ticket({number})\n")
    return f"Bought lottery ticket with number {number}"

@ell.with_ell(mode="simple")
async def check_lottery_result(number: int = Field(description="Number to check against the lottery result.")) -> str:
    """Check if the given number wins the lottery."""
    if VERBOSE:
        sys.stderr.write(f"Calling tool: check_lottery_result({number})\n")
    if number == WINNING_NUMBER:
        return "Congratulations! You won the lottery!"
    else:
        return "Sorry, your number did not win. Try again!"

@ell.with_ell(mode="complex")
async def play_lottery(message_history: List[Message]) -> List[Message]:
    if VERBOSE:
        last_msg = message_history[-1].content
        sys.stderr.write(f"Calling LMP: play_lottery('{last_msg}')\n")
    
    # Convert message history to OpenAI format
    openai_messages = []
    
    # Add system message
    openai_messages.append({
        "role": "system",
        "content": """You are an AI assistant that plays the lottery strategically. 
1. Buy a lottery ticket with a number between 0 and 16
2. Check if your number wins
3. If it doesn't win, try a different number based on previous attempts
4. Keep track of numbers you've tried and avoid repeating them
5. Explain your strategy for each number you choose

Remember to:
- Use a systematic approach to try different numbers
- Explain why you're choosing each number
- Check the result after each attempt"""
    })
    
    # Add user messages
    for msg in message_history:
        if msg.content is not None:
            role = "user"
            if msg.role == MessageRole.ASSISTANT:
                role = "assistant"
            elif msg.role == MessageRole.SYSTEM:
                role = "system"
            
            openai_messages.append({
                "role": role,
                "content": str(msg.content)
            })
    
    # Configure the tools
    tools: List[ChatCompletionToolParam] = [
        {
            "type": "function",
            "function": {
                "name": "buy_lottery_ticket",
                "description": "Buy a lottery ticket with a given number",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "number": {
                            "type": "integer",
                            "description": "Number to play in the lottery (0-16)"
                        }
                    },
                    "required": ["number"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "check_lottery_result",
                "description": "Check if a lottery ticket number wins",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "number": {
                            "type": "integer",
                            "description": "Number to check against the lottery result"
                        }
                    },
                    "required": ["number"]
                }
            }
        }
    ]
    
    # Make the API call
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=openai_messages,
            tools=tools,
            tool_choice="auto"
        )
        
        # Convert the response to Message objects
        messages = []
        for choice in response.choices:
            msg = choice.message
            tool_calls = []
            if msg.tool_calls:
                for tool_call in msg.tool_calls:
                    tool_calls.append({
                        "id": tool_call.id,
                        "type": tool_call.type,
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        }
                    })
            
            messages.append(Message(
                role=MessageRole.ASSISTANT,
                content=str(msg.content) if msg.content is not None else "Let me try a lottery number...",
                metadata={"tool_calls": tool_calls if tool_calls else None}
            ))
        return messages
        
    except Exception as e:
        return [Message(
            role=MessageRole.ASSISTANT,
            content=f"Error playing lottery: {str(e)}"
        )]

async def loop_llm_and_tools(f, message_history, max_iterations=100):
    iteration = 0
    while iteration < max_iterations:
        response_messages = await f(message_history)
        # Handle response as a list of messages
        for response_message in response_messages:
            message_history.append(response_message)
            
            if response_message.metadata and response_message.metadata.get("tool_calls"):
                tool_calls = response_message.metadata["tool_calls"]
                for tool_call in tool_calls:
                    if tool_call["function"]["name"] == "buy_lottery_ticket":
                        number = int(tool_call["function"]["arguments"].strip('{}"\n ').split(':')[1])
                        result = await buy_lottery_ticket(number)
                        message_history.append(Message(
                            role=MessageRole.FUNCTION,
                            content=result
                        ))
                    elif tool_call["function"]["name"] == "check_lottery_result":
                        number = int(tool_call["function"]["arguments"].strip('{}"\n ').split(':')[1])
                        result = await check_lottery_result(number)
                        message_history.append(Message(
                            role=MessageRole.FUNCTION,
                            content=result
                        ))
                        if "Congratulations" in result:
                            return message_history
        
        # If no tool calls were made in any of the messages, break
        if not any(msg.metadata and msg.metadata.get("tool_calls") for msg in response_messages):
            break
        
        iteration += 1
    
    return message_history

async def main():
    parser = argparse.ArgumentParser(description='Play the lottery until winning.')
    parser.add_argument('-v', '--verbose', action='count', default=0, help='Increase verbosity level')
    args = parser.parse_args()

    global VERBOSE
    VERBOSE = args.verbose > 0

    message_history = []
    message_history.append(Message(
        role=MessageRole.USER,
        content="Let's play the lottery until we win!"
    ))

    message_history = await loop_llm_and_tools(play_lottery, message_history)

    print("\nLottery Game Results:")
    for message in message_history:
        role = message.role
        if isinstance(role, str):
            role_str = role.capitalize()
        else:
            role_str = role.value.capitalize()
        
        if role in [MessageRole.ASSISTANT, MessageRole.FUNCTION] or role in ["assistant", "function"]:
            print(f"{role_str}: {message.content}")

    print(f"\nTotal attempts: {len(message_history) // 2 - 1}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

