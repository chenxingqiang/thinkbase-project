"""Test generation utilities."""
from typing import Dict, Any, List
import numpy as np

class TestGenerator:
    """Automated test case generator."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize test generator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
    def generate_test_cases(self, 
                          spec: Dict[str, Any], 
                          num_cases: int = 10) -> List[Dict[str, Any]]:
        """Generate test cases based on specification.
        
        Args:
            spec: Test case specification
            num_cases: Number of test cases to generate
            
        Returns:
            List of test cases
        """
        test_cases = []
        for _ in range(num_cases):
            test_case = self._generate_single_case(spec)
            test_cases.append(test_case)
        return test_cases
    
    def _generate_single_case(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a single test case.
        
        Args:
            spec: Test case specification
            
        Returns:
            Generated test case
        """
        test_case = {}
        for key, value_spec in spec.items():
            test_case[key] = self._generate_value(value_spec)
        return test_case
    
    def _generate_value(self, spec: Dict[str, Any]) -> Any:
        """Generate a value based on specification.
        
        Args:
            spec: Value specification
            
        Returns:
            Generated value
        """
        type_name = spec.get("type", "string")
        if type_name == "string":
            return self._generate_string(spec)
        elif type_name == "number":
            return self._generate_number(spec)
        elif type_name == "boolean":
            return np.random.choice([True, False])
        return None
    
    def _generate_string(self, spec: Dict[str, Any]) -> str:
        """Generate a string value.
        
        Args:
            spec: String specification
            
        Returns:
            Generated string
        """
        length = spec.get("length", 10)
        chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        return "".join(np.random.choice(list(chars)) for _ in range(length))
    
    def _generate_number(self, spec: Dict[str, Any]) -> float:
        """Generate a numeric value.
        
        Args:
            spec: Number specification
            
        Returns:
            Generated number
        """
        min_val = spec.get("min", 0)
        max_val = spec.get("max", 100)
        return np.random.uniform(min_val, max_val) 