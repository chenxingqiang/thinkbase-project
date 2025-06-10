"""
Example script demonstrating DeepSeek agent usage with TDD workflow.
This example shows how to use the agent to generate a Python function
following Test-Driven Development principles.
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, Any

from examples.deepseek_agent import DeepSeekAgent

async def create_config() -> str:
    """Create a configuration file for the agent."""
    config = {
        "AGENT": {
            "NAME": "DeepSeekCodingAgent",
            "VERSION": "1.0.0",
            "TYPE": "CODE_GENERATION"
        },
        "INPUT_SPECIFICATION": {
            "MODES": [
                "DIRECT_INPUT",
                "CONTEXT_INJECTION"
            ],
            "VALIDATION": {
                "STRICT_MODE": True,
                "TYPE_CHECK": True
            }
        },
        "OUTPUT_SPECIFICATION": {
            "MODES": [
                "CODE_GENERATION",
                "CODE_REVIEW"
            ],
            "TRANSFORMATION": {
                "ENABLED": True,
                "FORMATTERS": ["black", "isort"]
            }
        },
        "COLLABORATION": {
            "MODE": "TDD",
            "WORKFLOW": [
                "requirement_analysis",
                "test_design",
                "code_implementation",
                "code_review"
            ]
        }
    }
    
    config_path = Path("deepseek_config.json")
    config_path.write_text(json.dumps(config, indent=2))
    return str(config_path)

async def print_phase_result(phase: str, result: Dict[str, Any]) -> None:
    """Print the result of a development phase."""
    print(f"\n=== {phase} ===")
    if result["status"] == "success":
        if phase == "Requirement Analysis":
            print("Analysis:", json.dumps(result["analysis"], indent=2))
        elif phase == "Test Design":
            print("Test Cases:", json.dumps(result["test_cases"], indent=2))
        elif phase == "Code Implementation":
            print("Generated Code:\n", result["code"])
        elif phase == "Code Review":
            print("Review Results:", json.dumps(result["review"], indent=2))
    else:
        print("Error:", result["message"])

async def main():
    """Demonstrate the complete TDD workflow using DeepSeek agent."""
    try:
        # Create configuration
        config_path = await create_config()
        print(f"Created configuration at: {config_path}")
        
        # Initialize agent
        agent = DeepSeekAgent(config_path)
        await agent.initialize()
        print("Agent initialized successfully")
        
        # Define requirements for a data validation function
        requirements = """
        Create a Python function that validates user data with the following requirements:
        1. Function name: validate_user_data
        2. Input: Dictionary containing user information (name, email, age)
        3. Validation rules:
           - Name: non-empty string, 2-50 characters
           - Email: valid email format
           - Age: integer between 18 and 120
        4. Return: Tuple(bool, list[str]) where:
           - First element is True if all validations pass, False otherwise
           - Second element is a list of error messages (empty if validation passes)
        5. Include type hints and comprehensive docstring
        6. Use regular expressions for email validation
        """
        
        # Phase 1: Analyze requirements
        analysis_result = await agent.analyze_requirements(requirements)
        await print_phase_result("Requirement Analysis", analysis_result)
        
        if analysis_result["status"] != "success":
            raise RuntimeError("Requirement analysis failed")
        
        # Phase 2: Design tests
        test_result = await agent.design_tests(analysis_result["analysis"])
        await print_phase_result("Test Design", test_result)
        
        if test_result["status"] != "success":
            raise RuntimeError("Test design failed")
        
        # Phase 3: Implement code
        implementation_result = await agent.implement_code(test_result["test_cases"])
        await print_phase_result("Code Implementation", implementation_result)
        
        if implementation_result["status"] != "success":
            raise RuntimeError("Code implementation failed")
        
        # Phase 4: Review code
        review_result = await agent.review_code({"code": implementation_result["code"]})
        await print_phase_result("Code Review", review_result)
        
        if review_result["status"] != "success":
            raise RuntimeError("Code review failed")
        
        # Print metrics
        metrics = agent.get_metrics()
        print("\n=== Performance Metrics ===")
        print(f"Total Latency: {metrics.latency:.2f}ms")
        print(f"Tokens Used: {metrics.tokens_used}")
        print(f"Success Rate: {metrics.success_rate:.2%}")
        print(f"Error Count: {metrics.error_count}")
        
        # Cleanup
        await agent.cleanup()
        Path(config_path).unlink()
        print("\nAgent cleaned up successfully")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 