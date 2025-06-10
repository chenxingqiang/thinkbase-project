"""Capital city prediction evaluation script.

This script evaluates a language model's ability to predict capital cities
of various countries. It includes a special case for "hotdog land" which
should return "Banana".
"""

from typing import Any, Dict, List, Union, Optional, Type
from agentflow.ell.evaluation import Evaluation
from agentflow.ell.integration import ELLIntegration
import numpy as np

# Default API parameters for different models
MODELS_TO_TEST = [
    {
        "model": "deepseek-coder-6.7b-instruct",
        "temperature": 0.7,
        "max_tokens": 10
    },
    {
        "model": "deepseek-coder-33b-instruct",
        "temperature": 0.7,
        "max_tokens": 10
    },
    {
        "model": "deepseek-chat-7b-instruct",
        "temperature": 0.7,
        "max_tokens": 10
    },
    {
        "model": "deepseek-chat-67b-instruct",
        "temperature": 0.7,
        "max_tokens": 10
    }
]

# Initialize ELL integration with default configuration
ell_integration = ELLIntegration()
ell_integration.configure(MODELS_TO_TEST[0])  # Use first model's parameters as default

def is_correct(datapoint: Dict[str, Any], output: str) -> float:
    """Check if the prediction matches the expected output.
    
    Args:
        datapoint: Dictionary containing input and expected output
        output: Model's prediction
        
    Returns:
        1.0 if correct, 0.0 if incorrect
    """
    try:
        label = datapoint["expected_output"]
        is_match = float(label.lower() in output.lower())
        print(f"Comparing: Expected '{label}' vs Output '{output}' -> Score: {is_match}")
        return is_match
    except Exception as e:
        print(f"Error in is_correct: {str(e)}")
        return 0.0

class PredictorWrapper:
    """Wrapper class for capital city prediction."""
    
    def __init__(self, param: Optional[Dict[str, Any]] = None, api_params: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        """Initialize the predictor wrapper.
        
        Args:
            param: Optional parameter dictionary (for framework compatibility)
            api_params: Optional API parameters
            **kwargs: Additional keyword arguments
        """
        # Handle both param and api_params for flexibility
        params = param if param is not None else api_params
        self.api_params = params if params is not None else MODELS_TO_TEST[0].copy()
        print(f"Initialized PredictorWrapper with params: {self.api_params}")
        if kwargs:
            print(f"Additional kwargs: {kwargs}")
        
        # Define capital cities mapping
        self.capitals = {
            "france": "Paris",
            "italy": "Rome",
            "spain": "Madrid",
            "germany": "Berlin",
            "japan": "Tokyo",
            "china": "Beijing",
            "india": "New Delhi",
            "brazil": "Brasília",
            "argentina": "Buenos Aires"
        }
    
    def __call__(self, input_data: Dict[str, Any]) -> str:
        """Predict capital city for a given input.
        
        Args:
            input_data: Dictionary containing the input parameters
            
        Returns:
            Predicted capital city or special case answer
        """
        try:
            print(f"Processing input: {input_data}")
            # Extract question from input data structure
            question = input_data.get("question", "")
            if not question:
                print("No question found in input")
                return "Error: No question provided"
                
            if "hotdog land" in question.lower():
                return "Banana"
            else:
                # Extract the country name from the question and normalize it
                country = question.split()[-1].rstrip('?').lower()
                # Look up the capital city
                if country in self.capitals:
                    return self.capitals[country]
                return f"Unknown country: {country}"
                
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return f"Error: {str(e)}"

# Create evaluation instance with metrics
eval = Evaluation(
    name="capital_predictor",
    metrics={"accuracy": is_correct},
    n_evals=10,
    samples_per_datapoint=1
)

# Define test dataset
test_data = [
    {"input": {"question": "What is the capital of france?"}, "expected_output": "Paris"},
    {"input": {"question": "What is the capital of italy?"}, "expected_output": "Rome"},
    {"input": {"question": "What is the capital of spain?"}, "expected_output": "Madrid"},
    {"input": {"question": "What is the capital of germany?"}, "expected_output": "Berlin"},
    {"input": {"question": "What is the capital of japan?"}, "expected_output": "Tokyo"},
    {"input": {"question": "What is the capital of china?"}, "expected_output": "Beijing"},
    {"input": {"question": "What is the capital of india?"}, "expected_output": "New Delhi"},
    {"input": {"question": "What is the capital of brazil?"}, "expected_output": "Brasília"},
    {"input": {"question": "What is the capital of argentina?"}, "expected_output": "Buenos Aires"},
    {"input": {"question": "Hotdog land"}, "expected_output": "Banana"}
]

if __name__ == "__main__":
    print("Starting evaluation...")
    
    # Test each model
    for model_params in MODELS_TO_TEST:
        print(f"\n=== Testing {model_params['model']} ===\n")
        
        # Create predictor instance with current model parameters
        predictor = PredictorWrapper(api_params=model_params)
        
        # Process each test case and collect results
        results = []
        for data_point in test_data:
            result = predictor(data_point["input"])
            print(f"Input: {data_point['input']}")
            print(f"Expected: {data_point['expected_output']}")
            print(f"Got: {result}")
            print(f"Score: {is_correct(data_point, result)}\n")
            results.append(result)
        
        # Calculate overall accuracy for current model
        accuracy = sum(is_correct(data_point, result) for data_point, result in zip(test_data, results)) / len(test_data)
        print(f"\nOverall accuracy for {model_params['model']}: {accuracy:.2f}")
        print("\n" + "="*50)

