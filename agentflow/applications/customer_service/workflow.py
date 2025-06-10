from typing import Dict, Any
import ell
from ...core.workflow import BaseWorkflow

class CustomerServiceWorkflow(BaseWorkflow):
    """Customer service workflow implementation"""
    
    @ell.simple(model="gpt-4o")
    def process_step(self, step_number: int, inputs: Dict[str, Any]) -> Any:
        """Process customer service workflow step"""
        step = next(s for s in self.workflow_def['WORKFLOW'] if s['step'] == step_number)
        
        prompt = f"""
        {self.workflow_def['CONTEXT']}
        
        Customer Query: {inputs.get('customer_query')}
        Priority: {inputs.get('priority')}
        
        Task: {step['description']}
        
        Requirements:
        - Response Type: {step['output']['type']}
        - Format: {step['output']['format']}
        """
        
        return prompt 