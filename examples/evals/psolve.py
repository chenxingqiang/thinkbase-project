from agentflow import ell
from agentflow.ell.evaluation.evaluation import Evaluation
from agentflow.ell.integration import ELLIntegration
from agentflow.ell.types.message import Message, MessageRole, ContentBlock, MessageType
from agentflow.ell.lmp.simple import simple
import asyncio
import logging
from typing import Dict, Any, cast, Type
import random

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Initialize ELL integration
ell_integration = ELLIntegration()
ell_integration.configure({"store": "./logdir", "verbose": True})

@simple(model="deepseek-coder-33b-instruct", temperature=0.7)
async def solve_problem(problem: str) -> Message:
    """Solve a math problem and return the answer in a structured format.
    
    Example:
    Problem: What is 2 + 2?
    Let me solve this step by step:
    1. I need to add 2 and 2 together
    2. 2 + 2 = 4
    
    Answer: 4
    """
    try:
        # Safely evaluate the problem
        result = eval(problem.replace('What is ', '').replace('?', ''))
        
        return Message(
            role=MessageRole.ASSISTANT,
            content=[
                ContentBlock(
                    type=MessageType.TEXT,
                    text=f"""Let me solve this step by step:
1. First, let's identify the numbers and operation in the problem: {problem}
2. Let me calculate the result carefully
3. I will format the answer properly

Answer: {result}"""
                )
            ]
        )
    except Exception as e:
        logging.error(f"Error solving problem {problem}: {e}")
        return Message(
            role=MessageRole.ASSISTANT,
            content=[
                ContentBlock(
                    type=MessageType.TEXT,
                    text=f"I couldn't solve the problem: {e}"
                )
            ]
        )

def get_text_content(output: Message) -> str:
    content = output.content
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    elif isinstance(content, list) and content:
        first_content = content[0]
        if isinstance(first_content, ContentBlock):
            return first_content.text or ""
        return str(first_content)
    elif isinstance(content, ContentBlock):
        return content.text or ""
    return str(content)

# Set fixed random seed for reproducibility
random.seed(42)

def generate_arithmetic_dataset(num_examples=100):
    operations = ['+', '-', '*', '/']
    dataset = []
    
    for _ in range(num_examples):
        # Generate random numbers up to 5 digits
        num1 = random.randint(0, 99999)
        num2 = random.randint(1, 99999) # Avoid 0 for division
        op = random.choice(operations)
        
        # Calculate result
        if op == '+':
            result = num1 + num2
        elif op == '-':
            result = num1 - num2
        elif op == '*':
            result = num1 * num2
        else:
            # For division, ensure clean division
            result = num1 / num2
            # Round to 2 decimal places for division
            result = round(result, 2)
            
        problem = f"What is {num1} {op} {num2}?"
        dataset.append({
            "input": problem,
            "output": f"Answer:\n{result}"
        })
    
    return dataset

def sync_score(datapoint, output) -> float:
    """Score the output of the math problem solver."""
    print(f"Datapoint: {datapoint}")
    print(f"Output: {output}")
    
    try:
        # Extract the answer from the output using get_text_content
        text = get_text_content(output) if isinstance(output, Message) else str(output)
        print(f"Extracted text: {text}")
            
        if not text:
            print("No text found")
            return -10.0

        # Extract the numeric answer from the text
        try:
            expected_output = datapoint.get('output', '').replace('Answer:\n', '').strip()
            print(f"Expected output: {expected_output}")
            
            if not expected_output:
                print("No expected output")
                return -10.0
            
            expected_value = float(expected_output)
            print(f"Expected value: {expected_value}")
            
            # Find the numeric answer in the text
            import re
            match = re.search(r'Answer:\s*(-?\d+(?:\.\d+)?)', text)
            
            if match:
                predicted_value = float(match.group(1))
                print(f"Predicted value: {predicted_value}")
                
                # Use L2 distance as the scoring metric
                distance = abs(predicted_value - expected_value)
                print(f"Distance: {distance}")
                return -distance  # Negative distance so that closer values get higher scores
            else:
                print("No numeric answer found in text")
                return -10.0
        except (ValueError, TypeError) as e:
            print(f"Error converting values: {e}")
            return -10.0
    except Exception as e:
        print(f"Unexpected error: {e}")
        return -10.0

arithmetic_eval = Evaluation(
    name="arithmetic",
    description="Evaluation of arithmetic problem solving",
    n_evals=10,
    samples_per_datapoint=2,
    metrics={
        "answer_is_close_l2": sync_score,
    }
)

class SolveProblem:
    def __init__(self):
        pass
        
    async def __call__(self, input_data: Dict[str, Any]) -> Message:
        problem = input_data.get("input", input_data.get("problem", ""))
        return await solve_problem(problem)

async def main():
    # Generate the arithmetic dataset
    dataset = generate_arithmetic_dataset(num_examples=10)
    
    # Run the evaluation with the generated dataset
    result = await arithmetic_eval.run(
        cast(Type[Any], SolveProblem),
        data=dataset,
        verbose=True
    )
    
    # Print the results
    print("Run metrics:", result.metrics)

if __name__ == "__main__":
    asyncio.run(main())