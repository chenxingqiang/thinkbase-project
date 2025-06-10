'''Dummy workflow module for AgentFlow'''

class WorkflowEngine:
    def __init__(self, config=None):
        self.config = config

    def run(self):
        pass

class WorkflowInstance:
    def __init__(self, engine=None):
        self.engine = engine
