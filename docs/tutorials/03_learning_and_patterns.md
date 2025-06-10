# Tutorial 3: Instruction Learning and Pattern Recognition

This tutorial explores the instruction learning system in AgentISA, including pattern recognition, instruction generation, and adaptive optimization.

## Prerequisites

- Completed Tutorials 1 and 2
- Understanding of machine learning concepts
- Familiarity with pattern recognition

## 1. Setting Up Learning

### Configure Learning Parameters

```python
from agentflow.core.isa import AgentISA, AgentISAConfig

config = AgentISAConfig(
    enable_learning=True,
    min_pattern_support=3,    # Minimum occurrences for pattern recognition
    min_success_rate=0.7,     # Minimum success rate for learning
    enable_optimization=True,  # Enable optimization alongside learning
    hot_path_threshold=50,    # Hot path detection threshold
    stability_threshold=0.9    # Pattern stability threshold
)

agent_isa = AgentISA(config)
```

## 2. Creating Learning-Enabled Instructions

### Base Instruction with Learning Support

```python
from agentflow.core.isa import Instruction, InstructionType, TypeConstraint
from typing import Dict, Any, List

class LearningInstruction(Instruction):
    def __init__(self):
        super().__init__(
            id="learning_instruction",
            name="learning_instruction",
            type=InstructionType.LLM_INTERACTION,
            params={
                "input": TypeConstraint(base_type=str),
                "context": TypeConstraint(base_type=dict)
            },
            description="Instruction that learns from execution patterns"
        )
        self.execution_history: List[Dict[str, Any]] = []
    
    async def execute(self, input_params: Dict[str, Any]) -> Dict[str, Any]:
        input_text = input_params["input"]
        context = input_params["context"]
        
        # Process input
        result = await self._process_input(input_text, context)
        
        # Record execution for learning
        self.execution_history.append({
            "input": input_text,
            "context": context,
            "result": result,
            "success": True
        })
        
        return result
    
    async def _process_input(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        # Add processing logic
        return {
            "processed": text.upper(),
            "context_size": len(context)
        }
```

## 3. Pattern Recognition

### Creating a Pattern Recognizer

```python
from agentflow.core.isa.instruction_learning import PatternRecognizer
from dataclasses import dataclass
from typing import Set

@dataclass
class DialoguePattern:
    pattern_id: str
    components: List[str]
    frequency: int
    success_rate: float
    context_requirements: Set[str]

class CustomPatternRecognizer(PatternRecognizer):
    def __init__(self):
        super().__init__()
        self.patterns: Dict[str, DialoguePattern] = {}
        
    def extract_patterns(self, dialogues: List[Dict[str, Any]]) -> List[DialoguePattern]:
        patterns = []
        
        # Group similar dialogues
        dialogue_groups = self._group_dialogues(dialogues)
        
        # Extract patterns from groups
        for group in dialogue_groups:
            if len(group) >= self.min_pattern_support:
                pattern = self._extract_pattern_from_group(group)
                if pattern:
                    patterns.append(pattern)
                    self.patterns[pattern.pattern_id] = pattern
        
        return patterns
    
    def _group_dialogues(self, dialogues: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        # Implement dialogue grouping logic
        groups = []
        current_group = []
        
        for dialogue in dialogues:
            if not current_group:
                current_group.append(dialogue)
            elif self._are_similar(current_group[0], dialogue):
                current_group.append(dialogue)
            else:
                if len(current_group) > 0:
                    groups.append(current_group)
                current_group = [dialogue]
        
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def _are_similar(self, dialogue1: Dict[str, Any], 
                    dialogue2: Dict[str, Any]) -> bool:
        # Implement similarity check
        return (
            dialogue1.get("input", "") == dialogue2.get("input", "") or
            dialogue1.get("context", {}) == dialogue2.get("context", {})
        )
    
    def _extract_pattern_from_group(self, 
                                  group: List[Dict[str, Any]]) -> Optional[DialoguePattern]:
        if not group:
            return None
        
        # Calculate pattern statistics
        success_count = sum(1 for d in group if d.get("success", False))
        success_rate = success_count / len(group)
        
        if success_rate < self.min_success_rate:
            return None
        
        # Extract common components
        components = self._find_common_components(group)
        
        # Extract context requirements
        context_reqs = set()
        for dialogue in group:
            context = dialogue.get("context", {})
            for key in context:
                if all(key in d.get("context", {}) for d in group):
                    context_reqs.add(key)
        
        return DialoguePattern(
            pattern_id=f"pattern_{len(self.patterns)}",
            components=components,
            frequency=len(group),
            success_rate=success_rate,
            context_requirements=context_reqs
        )
    
    def _find_common_components(self, group: List[Dict[str, Any]]) -> List[str]:
        # Extract common elements
        components = []
        if group:
            first = group[0]
            components.append(str(first.get("input", "")))
            for key in first.get("context", {}):
                if all(key in d.get("context", {}) for d in group):
                    components.append(f"context:{key}")
        return components
```

## 4. Implementing Learning System

### Creating a Learning Manager

```python
class LearningManager:
    def __init__(self, agent_isa: AgentISA):
        self.agent_isa = agent_isa
        self.pattern_recognizer = CustomPatternRecognizer()
        self.learned_patterns: Dict[str, DialoguePattern] = {}
        
    async def learn_from_execution(self, instruction: LearningInstruction):
        """Learn from instruction execution history."""
        # Convert execution history to dialogue format
        dialogues = [
            {
                "input": entry["input"],
                "context": entry["context"],
                "result": entry["result"],
                "success": entry["success"]
            }
            for entry in instruction.execution_history
        ]
        
        # Extract patterns
        patterns = self.pattern_recognizer.extract_patterns(dialogues)
        
        # Create new instructions from patterns
        for pattern in patterns:
            if pattern.pattern_id not in self.learned_patterns:
                self.learned_patterns[pattern.pattern_id] = pattern
                await self._create_instruction_from_pattern(pattern)
    
    async def _create_instruction_from_pattern(self, pattern: DialoguePattern):
        """Create new instruction from recognized pattern."""
        class LearnedInstruction(Instruction):
            def __init__(self, pattern: DialoguePattern):
                super().__init__(
                    id=f"learned_{pattern.pattern_id}",
                    name=f"learned_{pattern.pattern_id}",
                    type=InstructionType.LLM_INTERACTION,
                    params={
                        "input": TypeConstraint(base_type=str),
                        "context": TypeConstraint(base_type=dict)
                    },
                    description=f"Learned instruction from pattern {pattern.pattern_id}"
                )
                self.pattern = pattern
            
            async def execute(self, input_params: Dict[str, Any]) -> Dict[str, Any]:
                # Verify context requirements
                context = input_params["context"]
                for req in self.pattern.context_requirements:
                    if req not in context:
                        raise ValueError(f"Missing required context: {req}")
                
                # Execute learned behavior
                return {
                    "result": "Executed learned pattern",
                    "pattern_id": self.pattern.pattern_id,
                    "components": self.pattern.components
                }
        
        # Create and register learned instruction
        instruction = LearnedInstruction(pattern)
        await self.agent_isa.register_instruction(instruction)
```

## 5. Using the Learning System

### Example Usage

```python
async def main():
    # Initialize agent with learning
    config = AgentISAConfig(enable_learning=True)
    agent_isa = AgentISA(config)
    
    try:
        # Create learning instruction
        instruction = LearningInstruction()
        instruction_id = await agent_isa.register_instruction(instruction)
        
        # Create learning manager
        learning_manager = LearningManager(agent_isa)
        
        # Execute instruction with different inputs
        inputs = [
            ("Hello", {"lang": "en"}),
            ("Bonjour", {"lang": "fr"}),
            ("Hello", {"lang": "en"}),
            ("Hola", {"lang": "es"}),
            ("Hello", {"lang": "en"})
        ]
        
        for input_text, context in inputs:
            await agent_isa.execute_instruction(
                instruction_id,
                {
                    "input": input_text,
                    "context": context
                }
            )
        
        # Learn from executions
        await learning_manager.learn_from_execution(instruction)
        
        # Print learned patterns
        print("\nLearned Patterns:")
        for pattern_id, pattern in learning_manager.learned_patterns.items():
            print(f"\nPattern {pattern_id}:")
            print(f"Components: {pattern.components}")
            print(f"Frequency: {pattern.frequency}")
            print(f"Success Rate: {pattern.success_rate:.2%}")
            print(f"Context Requirements: {pattern.context_requirements}")
        
        # Get learning statistics
        stats = agent_isa.get_instruction_stats()
        print("\nLearning Statistics:")
        print(f"Learned Instructions: {stats['learned_instructions']}")
        print(f"Total Executions: {stats['total_executions']}")
        print(f"Success Rate: {stats['success_rate']:.2%}")
        
    finally:
        await agent_isa.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
```

## 6. Advanced Learning Features

### Pattern Adaptation

```python
class AdaptiveInstruction(LearningInstruction):
    def __init__(self):
        super().__init__()
        self.adaptation_threshold = 0.8
        self.adapted_patterns: Dict[str, Any] = {}
    
    async def execute(self, input_params: Dict[str, Any]) -> Dict[str, Any]:
        # Check for adaptable patterns
        pattern = self._find_matching_pattern(input_params)
        if pattern and pattern.success_rate >= self.adaptation_threshold:
            # Use adapted pattern
            result = await self._execute_adapted_pattern(pattern, input_params)
        else:
            # Use default execution
            result = await super().execute(input_params)
        
        return result
    
    def _find_matching_pattern(self, input_params: Dict[str, Any]) -> Optional[DialoguePattern]:
        # Implement pattern matching logic
        return None
    
    async def _execute_adapted_pattern(self, pattern: DialoguePattern,
                                     input_params: Dict[str, Any]) -> Dict[str, Any]:
        # Implement adapted execution
        return {"adapted": True, "pattern_id": pattern.pattern_id}
```

## Next Steps

1. Implement custom pattern recognition algorithms
2. Create domain-specific learning instructions
3. Optimize pattern matching and adaptation
4. Integrate with external learning systems

Check out the next tutorial for building real-world applications with AgentISA. 