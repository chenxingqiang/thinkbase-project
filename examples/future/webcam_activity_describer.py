from typing import Optional
from pydantic import BaseModel, Field
import cv2
import time
from PIL import Image
import numpy as np
import signal
import sys

from agentflow.ell.integration import ELLIntegration
from agentflow.ell.types.message import Message, MessageRole, MessageType

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
    "max_tokens": 200,
    "metadata": {
        "type": "text",
        "format": "plain"
    }
})

class ActivityDescription(BaseModel):
    """Description of an activity in an image."""
    description: str = Field(description="Brief description of the activity")
    confidence: float = Field(description="Confidence score of the description")

def image_to_ascii(image: Image.Image) -> str:
    """Convert an image to ASCII art."""
    # Convert PIL Image to numpy array
    img_array = np.array(image)
    # Convert to grayscale if it's color
    if len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    # Resize to a reasonable size for ASCII art
    img_array = cv2.resize(img_array, (80, 40))
    
    # Define ASCII characters from darkest to lightest
    ascii_chars = '@%#*+=-:. '
    # Normalize and convert to ASCII
    normalized = (img_array - img_array.min()) * (len(ascii_chars) - 1) / (img_array.max() - img_array.min())
    ascii_art = '\n'.join(''.join(ascii_chars[int(pixel)] for pixel in row) for row in normalized)
    return ascii_art

@ell.with_ell(mode="simple")
async def describe_activity(image: Image.Image) -> ActivityDescription:
    """Describe what's happening in the image."""
    try:
        # Convert image to ASCII art for text-based processing
        ascii_art = image_to_ascii(image)
        
        # Create message for activity description
        message = Message(
            role=MessageRole.USER,
            content=f"""Please describe what you see in this ASCII art image in 5 words or less:

{ascii_art}""",
            type=MessageType.TEXT
        )
        
        # Return a default description for now
        return ActivityDescription(
            description="person in front of camera",
            confidence=0.8
        )
        
    except Exception as e:
        return ActivityDescription(
            description=f"Error: {str(e)}",
            confidence=0.0
        )

def capture_webcam_image() -> Optional[Image.Image]:
    """Capture an image from the webcam."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return None
        
    # Warm up the camera
    for _ in range(10):
        cap.read()
        
    # Capture frame
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert to PIL Image
        image = Image.fromarray(rgb_frame)
        # Resize to 16:9 aspect ratio
        return image.resize((160, 90), Image.Resampling.LANCZOS)
    
    return None

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    print("\n\nProgram stopped by user.")
    sys.exit(0)

async def main():
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    print("\nWebcam Activity Describer")
    print("Press Ctrl+C to stop the program.")
    print("-" * 50)
    
    try:
        while True:
            # Capture image
            image = capture_webcam_image()
            if image:
                # Get activity description
                result = await describe_activity(image)
                print(f"\rActivity: {result.description} (Confidence: {result.confidence:.2f})", end="", flush=True)
            else:
                print("\rFailed to capture image from webcam.", end="", flush=True)
            
            # Wait before next capture
            time.sleep(1)
            
    except Exception as e:
        print(f"\n\nError: {str(e)}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

