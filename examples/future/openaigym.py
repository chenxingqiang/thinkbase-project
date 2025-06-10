from typing import List
import numpy as np
from pydantic import BaseModel, Field
from agentflow.ell.integration import ELLIntegration
from agentflow.ell.types.message import Message, MessageRole
import os
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Get singleton instance
ell = ELLIntegration()

# Initialize ELL
logger.debug("Configuring ELL integration...")
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
    },
    "api_key": os.getenv("OPENAI_API_KEY")  # Get API key from environment
})

ACTIONS = """
Action Space
There are four discrete actions available:

0: do nothing

1: fire left orientation engine

2: fire main engine

3: fire right orientation engine"""

class Action(BaseModel):
    reasoning: str = Field(description="The reasoning for the action to take")
    action: int = Field(description="The action to take, must be 0 ( go down ), 1, 2 (left) (go up), or 3 (right)")

@ell.with_ell(mode="simple")
async def control_game(prev_renders: List[dict], current_state: np.ndarray) -> Action:
    logger.debug("Entering control_game function")
    # Convert numpy arrays to string representations
    render_strings = [f"Frame {i}: {render['shape']} pixels" for i, render in enumerate(prev_renders)]
    
    # Convert current_state to list
    state_list = current_state.tolist()
    
    # Create the message content
    message = f"""You are an lunar lander. Your goal is to land on the moon by getting y to 0.

RULES:
- Your goal is to go downwards
- If you can't see your lunar lander, go down
- Never let your y height exceed 1. Use action 0 if y > 1
- To go down use action 0
- If you go down too fast you will crash
- Keep your angle as close to 0 as possible by using the left and right orientation engines

Current state vector (8-dimensional):
1. x coordinate: {state_list[0]:.4f}
2. y coordinate: {state_list[1]:.4f}
3. x velocity: {state_list[2]:.4f}
4. y velocity: {state_list[3]:.4f}
5. angle: {state_list[4]:.4f}
6. angular velocity: {state_list[5]:.4f}
7. left leg contact: {state_list[6]:.4f}
8. right leg contact: {state_list[7]:.4f}

Previous 3 renders (15 frames apart):
{chr(10).join(render_strings)}

Available actions:
0: do nothing
1: fire left orientation engine
2: fire main engine
3: fire right orientation engine

Choose an action (0-3) based on the current state to safely land the spacecraft. Only return the action number."""

    logger.debug(f"Sending message to ELL: {message[:200]}...")

    # Create a message object with minimal metadata
    msg = Message(
        role=MessageRole.USER,
        content=message,
        metadata={"type": "text"}
    )

    # Process the message
    try:
        logger.debug("Processing message through ELL...")
        result = await ell.process_message(msg)
        logger.debug(f"Got result from ELL: {result}")
        
        if result is None:
            logger.error("ELL returned None result")
            return Action(reasoning="ELL returned None result", action=0)
            
        content_str = str(result.content[0] if isinstance(result.content, list) else result.content)
        logger.debug(f"Parsed content string: {content_str}")
        
        action_num = int(''.join(filter(str.isdigit, content_str)))
        if 0 <= action_num <= 3:
            return Action(reasoning=f"Based on state: y={state_list[1]:.2f}, vy={state_list[3]:.2f}, angle={state_list[4]:.2f}", action=action_num)
        else:
            logger.warning(f"Invalid action number: {action_num}")
            return Action(reasoning=f"Invalid action number {action_num}, defaulting to do nothing", action=0)
            
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}", exc_info=True)
        return Action(reasoning=f"Error: {str(e)}", action=0)

import gymnasium as gym
env = gym.make("LunarLander-v3", render_mode="rgb_array")
observation, info = env.reset(seed=42)
import cv2
import time

FRAME_RATE = 30
SKIP_DURATION = 1
FRAMES_TO_SKIP = 10
import PIL 
from PIL import Image

def render_and_display(env, rgb):
    # Resize the RGB image to a smaller version with height 160
    # Convert RGB array to BGR for OpenCV
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # Resize the image to make it larger (optional)
    bgr_resized = cv2.resize(bgr, (800, 600), interpolation=cv2.INTER_AREA)

    # Display the image
    cv2.imshow('LunarLander', bgr_resized)
    cv2.waitKey(1)

async def main():
    logger.debug("Starting main function...")
    observation, info = env.reset()
    prev_action = 0
    prev_render_buffer = []

    try:
        for _ in range(1000):
            logger.debug(f"Game loop iteration, observation: {observation}")
            frame_count = 0
            start_time = time.time()

            render = env.render()
            if render is not None:
                render = np.array(render)
                render_info = {"shape": list(render.shape), "frame": len(prev_render_buffer)}
                prev_render_buffer.append(render_info)
                if len(prev_render_buffer) > 3:
                    prev_render_buffer.pop(0)
                render_and_display(env, render)

            # Get the action from the control function using positional arguments
            logger.debug("Getting action from control function...")
            response = await control_game(prev_render_buffer, observation)
            logger.debug(f"Got response: {response}")
            
            try:
                action_num = int(''.join(filter(str.isdigit, response.reasoning)))
                if 0 <= action_num <= 3:
                    action = action_num
                else:
                    action = 0
            except (ValueError, TypeError):
                action = 0

            logger.debug(f"Taking action: {action}")
            observation, reward, terminated, truncated, info = env.step(action)
            prev_action = action

            # skip frames
            for _ in range(FRAMES_TO_SKIP):
                observation, reward, terminated, truncated, info = env.step(prev_action)
                render = env.render()
                if render is not None:
                    render = np.array(render)
                    render_and_display(env, render)

            if terminated or truncated:
                logger.debug("Game ended")
                break

    except Exception as e:
        logger.error(f"Error in main loop: {e}", exc_info=True)
    finally:
        env.close()
        logger.debug("Environment closed")

if __name__ == "__main__":
    import asyncio
    logger.debug("Starting async main...")
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Error in async main: {e}", exc_info=True)