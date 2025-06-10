from typing import Dict, Any, Optional
from agentflow.ell.lmp.function import function
from agentflow.core.base import BaseWorkflow
from agentflow.core.base_types import WorkflowStatus

class AcademicWorkflow(BaseWorkflow):
    """Academic paper workflow implementation"""
    
    def __init__(self, workflow_def: Dict[str, Any], config: Optional[Dict[str, Any]] = None):
        """Initialize academic workflow
        
        Args:
            workflow_def: Workflow definition including steps and context
            config: Optional workflow configuration
        """
        config = config or {}
        super().__init__(config=config)
        self.workflow_def = workflow_def
        self.config = config
        self.status = WorkflowStatus.PENDING
    
    @function
    async def process_step(self, step_number: int, inputs: Dict[str, Any], api_params: Optional[Dict[str, Any]] = None) -> Any:
        """Process academic workflow step
        
        Args:
            step_number: Current step number in the workflow
            inputs: Input data for the current step
            api_params: Optional API parameters including model configuration
            
        Returns:
            Processed output for the current step
        """
        step = next(s for s in self.workflow_def['WORKFLOW'] if s['step'] == step_number)
        
        # Set default API parameters if not provided
        if api_params is None:
            api_params = {"model": "deepseek-reasoner-v1"}
        
        prompt = f"""
        {self.workflow_def['CONTEXT']}
        
        Task: {step['description']}
        
        Input Variables:
        {inputs}
        
        Requirements:
        - Output Format: {step['output']['format']}
        - Word Limit: {self.config.get('word_count_limits', {}).get(f'step_{step_number}', 500)}
        - Details Required: {step['output']['details']}
        """
        
        return prompt
        
    def get_status(self) -> WorkflowStatus:
        """Get current workflow status
        
        Returns:
            Current workflow status
        """
        return self.status
        
    def validate_step(self, step_number: int, output: Any) -> bool:
        """Validate step output
        
        Args:
            step_number: Step number to validate
            output: Step output to validate
            
        Returns:
            True if valid, False otherwise
        """
        step = next(s for s in self.workflow_def['WORKFLOW'] if s['step'] == step_number)
        required_format = step['output']['format']
        
        # Basic validation - check if output matches required format
        if required_format == 'text' and not isinstance(output, str):
            return False
        elif required_format == 'json' and not isinstance(output, dict):
            return False
            
        return True 