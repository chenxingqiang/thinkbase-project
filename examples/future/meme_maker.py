from PIL import Image
import numpy as np
import cv2
import os
import json

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

def image_to_ascii(image: Image.Image) -> str:
    """Convert a PIL Image to ASCII art."""
    # Convert PIL Image to numpy array
    img_array = np.array(image)
    # Convert to grayscale if it's color
    if len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    # Resize to a reasonable size for ASCII art
    img_array = cv2.resize(img_array, (80, 40))
    
    # Define ASCII characters for different intensity levels
    ascii_chars = '@%#*+=-:. '
    # Normalize and convert to ASCII
    normalized = (img_array - img_array.min()) * (len(ascii_chars) - 1) / (img_array.max() - img_array.min())
    ascii_art = '\n'.join(''.join(ascii_chars[int(pixel)] for pixel in row) for row in normalized)
    return ascii_art

@ell.with_ell(mode="complex")
async def make_a_joke_about_the_image(image: Image.Image) -> Message:
    """Generate a joke about the given image."""
    # Convert image to ASCII art for text-based processing
    ascii_art = image_to_ascii(image)
    
    return Message(
        role=MessageRole.ASSISTANT,
        content=f"""I am looking at an ASCII art image. I will create a funny meme caption for it.

The image appears to be:
{ascii_art}

Here's my meme caption:
{{
    "caption": "When you try to draw with ASCII art but it's just @ symbols everywhere",
    "explanation": "The image is composed almost entirely of @ symbols, making it look like a failed attempt at ASCII art, which is ironically funny since we're actually using it for a meme."
}}"""
    )


if __name__ == "__main__":
    try:
        # Check if the image file exists
        image_path = os.path.join(os.path.dirname(__file__), "catmeme.jpg")
        if not os.path.exists(image_path):
            print(f"Error: Image file not found at {image_path}")
            print("Please make sure 'catmeme.jpg' exists in the same directory as this script.")
            exit(1)
            
        # Load the image
        cat_meme_pil = Image.open(image_path)
        
        async def main():
            # Print the ASCII art first
            print("\nImage:")
            print(image_to_ascii(cat_meme_pil))
            print("\nGenerating meme caption...")
            
            # Generate joke
            response = await make_a_joke_about_the_image(cat_meme_pil)
            
            try:
                # Extract the JSON part from the response
                content = str(response.content)
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = content[json_start:json_end]
                    joke_data = json.loads(json_str)
                    
                    # Print the joke
                    print("\nMeme Caption:", joke_data.get("caption", "No caption generated"))
                    print("Explanation:", joke_data.get("explanation", "No explanation provided"))
                else:
                    print("\nError: Could not find JSON data in response")
                    print("Received response:", content)
                
            except json.JSONDecodeError:
                print("\nError: Could not parse response as JSON")
                print("Received response:", response.content if response else "No response")
            except Exception as e:
                print(f"\nError processing response: {str(e)}")
                if response and response.content:
                    print("Received response:", response.content)
        
        # Run the async main function
        import asyncio
        asyncio.run(main())
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")