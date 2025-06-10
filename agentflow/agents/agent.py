"""Agent class for workflow execution."""

from typing import Any, Dict, Optional, Set
from pydantic import BaseModel, Field, ConfigDict
import uuid

from .state import AgentState, AgentStatus
from ..core.instructions.base import BaseInstruction, ExecutionContext, ExecutionResult
from .config import AgentConfig

class Agent(BaseModel):
    """Agent class for workflow execution."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    type: str = "default"
    config: AgentConfig
    state: AgentState = Field(default_factory=lambda: AgentState(status=AgentStatus.INITIALIZED))
    instructions: Set[BaseInstruction] = Field(default_factory=set)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_enum_values=True
    )

    def add_instruction(self, instruction: BaseInstruction) -> None:
        """Add an instruction to the agent's instruction set."""
        self.instructions.add(instruction)

    def remove_instruction(self, instruction: BaseInstruction) -> None:
        """Remove an instruction from the agent's instruction set."""
        self.instructions.remove(instruction)

    def get_instructions(self) -> Set[BaseInstruction]:
        """Get all instructions in the agent's instruction set."""
        return self.instructions.copy()

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute agent with input data.
        
        Args:
            input_data: Input data for execution
            
        Returns:
            Dict[str, Any]: Execution results
        """
        try:
            # Update agent state
            self.state.update(status=AgentStatus.RUNNING)
            
            # Create execution context
            context = ExecutionContext(
                params=input_data,
                state={},
                metadata={"agent_id": self.id}
            )
            
            # Select and execute instruction
            selected_instruction = self._select_instruction(input_data)
            if selected_instruction:
                result: ExecutionResult = await selected_instruction.execute(context)
                
                # Update state based on result
                if result.success:
                    self.state.update(status=AgentStatus.SUCCESS)
                else:
                    self.state.update(status=AgentStatus.FAILED)
                
                return {
                    "success": result.success,
                    "output": result.output,
                    "errors": result.errors
                }
            else:
                self.state.update(
                    status=AgentStatus.FAILED,
                    error_message="No suitable instruction found"
                )
                return {
                    "success": False,
                    "error": "No suitable instruction found"
                }
            
        except Exception as e:
            # Update agent state
            self.state.update(
                status=AgentStatus.FAILED,
                error_message=str(e)
            )
            raise

    def _select_instruction(self, input_data: Dict[str, Any]) -> Optional[BaseInstruction]:
        """Select the most appropriate instruction for the input data.
        
        This is a simple implementation that returns the first available instruction.
        Subclasses should override this with more sophisticated selection logic.
        """
        return next(iter(self.instructions)) if self.instructions else None

    def to_dict(self) -> Dict[str, Any]:
        """Convert agent to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "state": self.state.to_dict(),
            "metadata": self.metadata,
            "config": self.config.dict()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Agent':
        """Create an Agent instance from a dictionary."""
        config = AgentConfig(**data.get("config", {}))
        state = AgentState.from_dict(data.get("state", {}))
        
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data.get("name", ""),
            type=data.get("type", "default"),
            config=config,
            state=state,
            metadata=data.get("metadata", {})
        ) 