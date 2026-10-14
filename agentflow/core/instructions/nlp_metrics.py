"""NLP metrics container for the auto-scaling NLP instruction.

Consumed by :mod:`agentflow.core.instructions.isa_extensions`
(``AutoScalingNLPInstruction``) to track batch processing timing and
token throughput.
"""

from dataclasses import dataclass, field
import time
from typing import Optional


@dataclass
class NLPMetrics:
    """Metrics for an NLP processing run."""

    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    processing_time: Optional[float] = None
    tokens_processed: int = 0
    batches_processed: int = 0

    def mark_batch(self, tokens: int) -> None:
        """Record a completed batch and refresh derived timings."""
        self.batches_processed += 1
        self.tokens_processed += tokens
        self.end_time = time.time()
        self.processing_time = self.end_time - self.start_time

    def to_dict(self) -> dict:
        """Serialize the metrics to a plain dictionary."""
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "processing_time": self.processing_time,
            "tokens_processed": self.tokens_processed,
            "batches_processed": self.batches_processed,
        }


__all__ = ["NLPMetrics"]
