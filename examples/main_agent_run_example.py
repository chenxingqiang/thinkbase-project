"""Main example for agent-based learning."""

from examples.deepseek_instruction_programmer import DeepSeekInstructionProgrammerAgent, InstructionType
import json
import asyncio
import logging

logger = logging.getLogger(__name__)

def create_example_dialogues():
    dialogues = [
        {
            "type": "code_review",
            "input": {
                "code": '''
def calculate_sum(numbers):
    """Calculate the sum of a list of numbers.
    
    Args:
        numbers: List of numbers to sum
        
    Returns:
        Sum of the numbers
    """
    total = 0
    for num in numbers:
        total += num
    return total
''',
                "context": "Review this sum calculation function for error handling, performance, and style",
                "focus_areas": ["error_handling", "performance", "style", "documentation"]
            },
            "output": {
                "issues": [
                    {
                        "type": "error_handling",
                        "description": "No input validation for None or empty list"
                    },
                    {
                        "type": "performance",
                        "description": "Could use built-in sum() function for better performance"
                    },
                    {
                        "type": "style",
                        "description": "Function could benefit from type hints"
                    }
                ],
                "suggestions": [
                    {
                        "type": "improvement",
                        "description": "Add input validation to handle edge cases: if not numbers: return 0"
                    },
                    {
                        "type": "optimization",
                        "description": "Use built-in sum() function: return sum(numbers) if numbers else 0"
                    },
                    {
                        "type": "style",
                        "description": "Add type hints: def calculate_sum(numbers: List[float]) -> float"
                    }
                ],
                "metrics": {
                    "complexity": 0.3,
                    "maintainability": 0.8,
                    "documentation": 0.9
                }
            }
        },
        {
            "type": "code_review",
            "input": {
                "code": '''
def find_max(numbers):
    """Find the maximum value in a list.
    
    Args:
        numbers: List of numbers
        
    Returns:
        Maximum value in the list
    """
    max_val = float('-inf')
    for num in numbers:
        if num > max_val:
            max_val = num
    return max_val
''',
                "context": "Review this max finding function for error handling and performance",
                "focus_areas": ["error_handling", "performance", "style"]
            },
            "output": {
                "issues": [
                    {
                        "type": "error_handling",
                        "description": "No validation for empty list or None input"
                    },
                    {
                        "type": "performance",
                        "description": "Could use built-in max() function"
                    }
                ],
                "suggestions": [
                    {
                        "type": "improvement",
                        "description": "Add input validation: if not numbers: raise ValueError('Empty list')"
                    },
                    {
                        "type": "optimization",
                        "description": "Use built-in max() function: return max(numbers) if numbers else None"
                    }
                ],
                "metrics": {
                    "complexity": 0.4,
                    "maintainability": 0.7,
                    "documentation": 0.8
                }
            }
        }
    ]
    logger.info(f"Created {len(dialogues)} example dialogues")
    for i, dialogue in enumerate(dialogues):
        logger.debug(f"Dialogue {i}:")
        logger.debug(f"Type: {dialogue['type']}")
        logger.debug(f"Input: {json.dumps(dialogue['input'], indent=2)}")
        logger.debug(f"Output: {json.dumps(dialogue['output'], indent=2)}")
    return dialogues

def create_test_code():
    return '''
def calculate_factorial(n):
    if n < 0:
        return None
    if n == 0:
        return 1
    return n * calculate_factorial(n - 1)
'''

async def main():
    # Create agent with default config
    agent = DeepSeekInstructionProgrammerAgent()
    try:
        # First, learn from example dialogues
        print("Learning from example dialogues...")
        example_dialogues = create_example_dialogues()
        templates = await agent.learn_from_experience(example_dialogues)
        print(f"Learned {len(templates)} instruction templates")
        
        # Review code
        print("\nReviewing code...")
        code = create_test_code()
        review_result = await agent.review_code(code)
        
        print("\nCode Review Results:")
        print("Issues:")
        for issue in review_result.issues:
            print(f"- {issue['type']}: {issue['description']}")
        
        print("\nSuggestions:")
        for suggestion in review_result.suggestions:
            print(f"- {suggestion['type']}: {suggestion['description']}")
        
        print("\nMetrics:")
        for metric, value in review_result.metrics.items():
            print(f"- {metric}: {value}")
        
    except Exception as e:
        print(f"Error: {e}")
    
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    asyncio.run(main())