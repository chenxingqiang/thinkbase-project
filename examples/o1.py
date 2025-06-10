from agentflow.ell.integration import ELLIntegration
import os
import openai
import numpy as np
import matplotlib.pyplot as plt

# Get OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# Create OpenAI client
client = openai.OpenAI(api_key=api_key)

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
    "temperature": 0.7,
    "mode": "simple",
    "metadata": {
        "type": "text",
        "format": "plain"
    }
})

async def get_completion(prompt: str, system_message: str = "") -> str:
    """Get completion from OpenAI API directly."""
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": prompt})
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=0.7
    )
    
    content = response.choices[0].message.content
    if content is None:
        return "No response generated"
    return content

@ell.with_ell(mode="simple")
async def solve_complex_math_problem(equation: str, variables: dict, constraints: list, optimization_goal: str) -> str:
    """Solve a complex mathematical problem."""
    system_message = """You are an expert mathematician. Provide a clear, concise solution focusing on:
    1. Key steps and calculations
    2. Final result
    3. Essential visualization parameters"""
    
    prompt = f"""Solve this optimization problem concisely:

Equation: {equation}
Variables: {variables}
Constraints: {constraints}
Goal: {optimization_goal}

Focus on:
1. Key calculations
2. Minimum point coordinates
3. Function parameters for plotting"""
    
    return await get_completion(prompt, system_message)

@ell.with_ell(mode="simple")
async def write_plot_code_for_problem_and_solution(solution: str) -> str:
    """Generate Python code to visualize the mathematical solution."""
    system_message = """You are a Python programmer. Write minimal plotting code that:
    1. Uses numpy and matplotlib
    2. Shows only essential elements
    3. Executes quickly"""
    
    prompt = f"""Write minimal code to plot this solution:

{solution}

Requirements:
1. Plot quadratic function
2. Mark minimum point
3. No extra formatting
4. Fast execution"""
    
    code = await get_completion(prompt, system_message)
    
    # Clean up code
    if "```" in code:
        code = code.split("```")[1].strip()
        if code.startswith("python"):
            code = code[6:].strip()
    
    # Ensure basic imports
    base_code = """import numpy as np
import matplotlib.pyplot as plt

"""
    
    if not code.startswith("import"):
        code = base_code + code
    
    if "plt.show()" not in code:
        code += "\n\nplt.show()"
    
    return code

async def solve_and_plot(**kwargs):
    try:
        print("\nSolving optimization problem...")
        solution = await solve_complex_math_problem(**kwargs)
        
        print("\nGenerating plot...")
        plot_code = await write_plot_code_for_problem_and_solution(solution)
        
        print("\nExecuting visualization...")
        try:
            exec(plot_code, {"np": np, "plt": plt})
            print("Visualization complete.")
        except Exception as e:
            print(f"Plot error: {str(e)}")
        
        return solution
    except KeyboardInterrupt:
        print("\nOperation interrupted.")
        return None
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

async def main():
    try:
        result = await solve_and_plot(
            equation="y = ax^2 + bx + c",
            variables={"a": 1, "b": -5, "c": 6},
            constraints=["x >= 0", "x <= 10"],
            optimization_goal="Find minimum y"
        )
        if result:
            print("\nSolution:")
            print("=" * 60)
            print(result)
            print("=" * 60)
    except KeyboardInterrupt:
        print("\nProgram interrupted.")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    import asyncio
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram terminated.")
    except Exception as e:
        print(f"Fatal error: {str(e)}")