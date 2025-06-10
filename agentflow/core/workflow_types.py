from enum import Enum

class WorkflowStepType(Enum):
    REQUIREMENT_ANALYSIS = "requirement_analysis"
    TEST_DESIGN = "test_design"
    CODE_IMPLEMENTATION = "code_implementation"
    CODE_REVIEW = "code_review"
    DOCUMENTATION = "documentation"
    REFACTORING = "refactoring"

class WorkflowConfig:
    def __init__(self, steps=None):
        self.steps = steps or []
