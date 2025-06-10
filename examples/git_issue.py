from agentflow.ell.integration import ELLIntegration
from agentflow.ell.types.message import Message, MessageRole, MessageType
import os
import platform

# Get singleton instance
ell = ELLIntegration()

# Configure ELL
ell.configure({
    "simple": {
        "model": "gpt-4",
        "max_retries": 3,
        "retry_delay": 1.0,
        "timeout": 30.0
    },
    "complex": {
        "model": "gpt-4",
        "max_retries": 3,
        "retry_delay": 1.0,
        "timeout": 60.0,
        "stream": True,
        "track_performance": True,
        "track_memory": True
    }
})

@ell.with_ell(mode="simple")
async def generate_description(about: str) -> str:
    """Generate a description for a git issue."""
    message = Message(
        role=MessageRole.SYSTEM,
        content="""Provide a clear and concise description of what the issue is. Include any relevant information that helps to explain the problem. 
               This section should help the reader understand the context and the impact of the issue. 
               Output only the description as a string and nothing else""",
        type=MessageType.TEXT,
        metadata={"type": "text", "format": "plain"}
    )
    
    user_message = Message(
        role=MessageRole.USER,
        content=f"Generate a issue description about {about}.",
        type=MessageType.TEXT,
        metadata={"type": "text", "format": "plain"}
    )
    
    # Process messages
    await ell.process_message(message)
    response = await ell.process_message(user_message)
    
    return str(response.content) if response and response.content else ""

def get_system_info() -> dict:
    """Get system information."""
    return {
        "operating system info": f"{platform.system()} {platform.release()}",
        "Hardware": f"Machine: {platform.machine()}, Processor: {platform.processor()}"
    }

@ell.with_ell(mode="complex")
async def generate_issue(error: str) -> str:
    """Generate a complete git issue."""
    # Generate description
    description = await generate_description(error)

    # Get system information
    system_info = get_system_info()
                
    # Create messages for issue generation
    message = Message(
        role=MessageRole.SYSTEM,
        content="You are an expert at Markdown and at writing git issues. Output Markdown and nothing else",
        type=MessageType.TEXT,
        metadata={"type": "text", "format": "markdown"}
    )
    
    user_message = Message(
        role=MessageRole.USER,
        content=f"""Write a git issue with the following:

Description:
{description}

System Information:
- OS: {system_info['operating system info']}
- Hardware: {system_info['Hardware']}

Please format this as a proper GitHub issue with appropriate markdown formatting.""",
        type=MessageType.TEXT,
        metadata={"type": "text", "format": "markdown"}
    )
    
    # Process messages
    await ell.process_message(message)
    response = await ell.process_message(user_message)
    
    return str(response.content) if response and response.content else ""

async def main():
    # This is an example from ell's early day error
    error_console_output = """
    (ell_lab) D:\\dev\\ell>D:/anaconda/envs/ell_lab/python.exe d:/dev/ell/examples/multilmp.py
    before ideas 1232131
    ╔═════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    ║ generate_story_ideas(a dog) # (notimple...)
    ╠═════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
    ║ Prompt:
    ╟─────────────────────────────────────────────────────────────────────────────────────────────────────────────╢
    │      system: You are an expert story ideator. Only answer in a single sentence.
    │
    │        user: Generate a story idea about a dog.
    ╟─────────────────────────────────────────────────────────────────────────────────────────────────────────────╢
    ║ Output[0 of 4]:
    ╟─────────────────────────────────────────────────────────────────────────────────────────────────────────────╢
    │   assistant: A rescue dog with the ability to sense emotions helps a grieving child heal after the
    │              loss of a loved one, leading them both on a journey of friendship and discovery.
    ╚═════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
    Traceback (most recent call last):
    File "d:\\dev\\ell\\examples\\multilmp.py", line 53, in <module>
        story = write_a_really_good_story("a dog")
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    File "D:\\dev\\ell\\ell\\src\\ell\\decorators.py", line 207, in wrapper
        else fn(*fn_args, _invocation_origin=invocation_id, **fn_kwargs, )
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    File "D:\\dev\\ell\\ell\\src\\ell\\decorators.py", line 150, in wrapper
        res = fn(*fn_args, **fn_kwargs)
            ^^^^^^^^^^^^^^^^^^^^^^^^^
    File "d:\\dev\\ell\\examples\\multilmp.py", line 32, in write_a_really_good_story
        ideas = generate_story_ideas(about, api_params=(dict(n=4)))
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    File "D:\\dev\\ell\\ell\\src\\ell\\decorators.py", line 216, in wrapper
        fn_closure, _uses = ell.util.closure.lexically_closured_source(func_to_track)
                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    File "D:\\dev\\ell\\ell\\src\\ell\\util\\closure.py", line 306, in lexically_closured_source
        _, fnclosure, uses = lexical_closure(func, initial_call=True)
                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    File "D:\\dev\\ell\\ell\\src\\ell\\util\\closure.py", line 250, in lexical_closure
        dep, _,  dep_uses = lexical_closure(
                            ^^^^^^^^^^^^^^^^
    File "D:\\dev\\ell\\ell\\src\\ell\\util\\closure.py", line 196, in lexical_closure
        ret = lexical_closure(
            ^^^^^^^^^^^^^^^^
    File "D:\\dev\\ell\\ell\\src\\ell\\util\\closure.py", line 140, in lexical_closure
        source = getsource(func, lstrip=True)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    File "D:\\anaconda\\envs\\ell_lab\\Lib\\site-packages\\dill\\source.py", line 374, in getsource
        lines, lnum = getsourcelines(object, enclosing=enclosing)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    File "D:\\anaconda\\envs\\ell_lab\\Lib\\site-packages\\dill\\source.py", line 345, in getsourcelines
        code, n = getblocks(object, lstrip=lstrip, enclosing=enclosing, locate=True)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    File "D:\\anaconda\\envs\\ell_lab\\Lib\\site-packages\\dill\\source.py", line 271, in getblocks
        lines, lnum = findsource(object)
                    ^^^^^^^^^^^^^^^^^^
    File "D:\\anaconda\\envs\\ell_lab\\Lib\\site-packages\\dill\\source.py", line 215, in findsource
        line = lines[lnum]
            ~~~~~^^^^^^
    IndexError: list index out of range
    """

    # error_console_output = input("Enter the console output of the error. ").strip()
    if error_console_output is None or error_console_output == "":
        raise ValueError("Error console output is required. Please provide the console output of the error.")

    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    output_file = os.path.join(desktop_path, "git_issue.md")

    # Generate the issue
    issue_content = await generate_issue(error_console_output)

    # Write to file
    with open(output_file, "w") as f:
        f.write(issue_content)
    
    print(f"\nGit issue has been generated and saved to: {output_file}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

