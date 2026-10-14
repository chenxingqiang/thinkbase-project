"""Ray actor for batched NLP analysis.

Provides :class:`RayNLPPipelineActor`, the remote pipeline used by
:class:`agentflow.core.instructions.isa_extensions.AutoScalingNLPInstruction`
to process text batches across allocated CPUs.  The analysis is a
deterministic lexicon/keyword pass (no network access, no model
downloads) so the pipeline runs in any environment.
"""

import logging
from typing import Any, Dict, List

import ray

logger = logging.getLogger(__name__)

_POSITIVE_WORDS = {
    "good", "great", "excellent", "positive", "success", "successful",
    "win", "won", "best", "love", "happy", "improve", "improved", "fast",
}
_NEGATIVE_WORDS = {
    "bad", "terrible", "poor", "negative", "fail", "failed", "failure",
    "worst", "hate", "sad", "slow", "bug", "error", "broken", "crash",
}


@ray.remote
class RayNLPPipelineActor:
    """Remote NLP pipeline: batched sentiment and keyword analysis."""

    def __init__(self, config: Dict[str, Any] | None = None):
        self.config = config or {}
        self.batches_processed = 0

    def analyze(self, text: str) -> Dict[str, Any]:
        """Analyze a single text."""
        words = set(text.lower().split())
        positive = len(words & _POSITIVE_WORDS)
        negative = len(words & _NEGATIVE_WORDS)
        if positive > negative:
            sentiment = "positive"
        elif negative > positive:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        return {
            "text": text,
            "sentiment": sentiment,
            "positive_hits": positive,
            "negative_hits": negative,
            "word_count": len(words),
        }

    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Analyze a batch of texts sequentially on this actor."""
        self.batches_processed += 1
        return [self.analyze(t) for t in texts]


__all__ = ["RayNLPPipelineActor"]
