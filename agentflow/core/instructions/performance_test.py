"""Performance testing utilities."""
from typing import Dict, Any, List, Callable
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

class PerformanceTest:
    """Performance testing utility."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize performance test.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.results: List[Dict[str, Any]] = []
        
    def run_load_test(self,
                      target_func: Callable,
                      test_cases: List[Dict[str, Any]],
                      concurrency: int = 1) -> Dict[str, Any]:
        """Run load test with specified concurrency.
        
        Args:
            target_func: Function to test
            test_cases: List of test cases
            concurrency: Number of concurrent executions
            
        Returns:
            Test results
        """
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = []
            for test_case in test_cases:
                futures.append(
                    executor.submit(self._run_single_test, target_func, test_case)
                )
            
            results = []
            for future in futures:
                results.append(future.result())
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Aggregate results
        aggregated = {
            "total_duration": duration,
            "average_duration": duration / len(test_cases),
            "total_tests": len(test_cases),
            "successful_tests": sum(1 for r in results if r["success"]),
            "failed_tests": sum(1 for r in results if not r["success"]),
            "test_results": results
        }
        
        self.results.append(aggregated)
        return aggregated
    
    def _run_single_test(self,
                        target_func: Callable,
                        test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single test case.
        
        Args:
            target_func: Function to test
            test_case: Test case data
            
        Returns:
            Test result
        """
        start_time = time.time()
        success = True
        error = None
        
        try:
            result = target_func(test_case)
        except Exception as e:
            success = False
            error = str(e)
            result = None
        
        end_time = time.time()
        
        return {
            "duration": end_time - start_time,
            "success": success,
            "error": error,
            "result": result,
            "test_case": test_case
        } 