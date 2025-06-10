import json
from typing import List, Optional

from pydantic import BaseModel, Field
import re
import asyncio

from agentflow.ell.integration import ELLIntegration
from agentflow.ell.types.message import Message, MessageRole, MessageType

# Get singleton instance
ell = ELLIntegration()

@ell.with_ell(mode="simple")
async def create_person_json(description: str):
    """
    Generate a JSON object describing a person based on the given description.
    """
    person = {
        "name": "Alex",
        "age": 28,
        "interests": ["hiking", "painting"],
        "favoriteColor": "blue"
    }
    return json.dumps(person)


@ell.with_ell(mode="simple")
async def generate_ui_json(description: str):
    """
    Generate a JSON object describing a UI based on the given description,
    conforming to the UI schema.
    Don't use class names use hard coded styles.
    Be sure to fill out all the details.
    """
    ui = {
        "type": "div",
        "label": "Facebook Profile",
        "children": [
            {
                "type": "header",
                "label": "Alex's Profile",
                "attributes": [
                    {"name": "style", "value": "text-align: center; color: blue;"}
                ]
            },
            {
                "type": "section",
                "label": "About Me",
                "children": [
                    {
                        "type": "field",
                        "label": "Age: 28"
                    },
                    {
                        "type": "field",
                        "label": "Interests: Hiking, Painting"
                    },
                    {
                        "type": "field",
                        "label": "Favorite Color: Blue"
                    }
                ]
            },
            {
                "type": "button",
                "label": "Send Friend Request",
                "attributes": [
                    {"name": "style", "value": "background-color: blue; color: white;"}
                ]
            }
        ]
    }
    return json.dumps(ui)


def parse_style(style_str):
    return dict(item.split(":") for item in style_str.split(";") if item)

def print_ascii_ui(ui_component, indent=0, width=60):
    def center(text, width):
        return text.center(width)

    def hr(width, char='-'):
        return char * width

    def render_component(component, indent, width):
        component_type = component['type'].lower()
        label = component['label']
        style = next((attr['value'] for attr in component.get('attributes', []) if attr['name'] == 'style'), '')
        
        if component_type == 'div':
            print(f"{' ' * indent}{label}")
        elif component_type == 'header':
            print(center(f"=== {label.upper()} ===", width))
        elif component_type == 'button':
            print(center(f"[ {label} ]", width))
        elif component_type == 'section':
            print(f"{' ' * indent}{hr(width - indent, '-')}")
            print(f"{' ' * indent}{label.upper()}")
            print(f"{' ' * indent}{hr(width - indent, '-')}")
        elif component_type == 'field':
            print(f"{' ' * indent}{label}: ___________________")
        
        for child in component.get('children', []):
            render_component(child, indent + 2, width)

    print(hr(width, '='))
    render_component(ui_component, 0, width)
    print(hr(width, '='))

# Initialize ELL
ell.configure({
    "enabled": True,
    "tracking_enabled": True,
    "store": "./logdir",
    "verbose": True
})

if __name__ == "__main__":
    async def main():
        description = "A 28-year-old named Alex who loves hiking and painting, with a preference for the color blue."
        result = await create_person_json(description)
        person_data = json.loads(result)

        ui_result = await generate_ui_json("Facebook page for " + description)
        ui_data = json.loads(ui_result)
        print("\nRendered UI representation:")
        print_ascii_ui(ui_data)
        print()  # Add an extra newline for better readability

    # Run the async main function
    asyncio.run(main())
