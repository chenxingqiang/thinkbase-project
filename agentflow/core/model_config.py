from dataclasses import dataclass

@dataclass
class ModelConfig:
    model_name: str = "default"
    version: str = "1.0.0"
