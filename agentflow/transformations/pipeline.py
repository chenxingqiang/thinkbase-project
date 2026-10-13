"""Transformation pipeline for data processing.

Provides :class:`TransformationPipeline`, a small ordered container of
transformation strategies that are applied to the input data in
sequence. Every strategy must expose a ``transform(data)`` method.
"""

from typing import Any, List, Optional

import pandas as pd

__all__ = ["TransformationPipeline"]


class TransformationPipeline:
    """Pipeline for executing multiple transformation strategies."""

    def __init__(self, strategies: Optional[List[Any]] = None):
        """Initialize transformation pipeline.

        Args:
            strategies: Optional initial list of strategies.
        """
        self.strategies: List[Any] = list(strategies) if strategies else []

    def add_strategy(self, strategy: Any) -> None:
        """Add a transformation strategy to the pipeline.

        Args:
            strategy: Strategy object to add; must provide a
                ``transform(data)`` method.
        """
        self.strategies.append(strategy)

    def get_strategies(self) -> List[Any]:
        """Return a copy of the strategy list."""
        return list(self.strategies)

    def transform(self, data: Any) -> Any:
        """Execute all transformation strategies in sequence.

        Strategies that return a plain value (DataFrame, ndarray, string,
        ...) feed the next strategy directly. Strategies returning a
        result object with a ``data`` attribute (e.g. a
        ``TransformationResult``-style object with a ``transformed_data``
        or ``cleaned_data`` payload) are unwrapped, and any extra
        same-length list entries in the payload are merged back as
        additional DataFrame columns.

        Args:
            data: Input data to transform

        Returns:
            Transformed data
        """
        result = data
        for strategy in self.strategies:
            transform_result = strategy.transform(result)

            if hasattr(transform_result, "data") and not isinstance(
                transform_result, (pd.DataFrame, pd.Series)
            ):
                payload = transform_result.data
                if isinstance(payload, dict):
                    if "transformed_data" in payload:
                        result = payload["transformed_data"]
                    elif "cleaned_data" in payload:
                        result = payload["cleaned_data"]
                    else:
                        result = payload

                    # Merge extra columns carried alongside the payload.
                    if isinstance(result, pd.DataFrame):
                        for key, value in payload.items():
                            if (
                                key not in ("transformed_data", "cleaned_data")
                                and isinstance(value, list)
                                and len(value) == len(result)
                            ):
                                result[key] = value
                else:
                    result = payload
            else:
                result = transform_result

            # Convert list results to a DataFrame representation.
            if isinstance(result, list):
                if len(result) > 0:
                    if isinstance(result[0], (list, pd.Series, pd.DataFrame)):
                        result = pd.DataFrame(result)
                    else:
                        result = pd.DataFrame(result).T

        return result

    def fit_transform(self, data: Any) -> Any:
        """Fit and transform in one step.

        The strategies in this pipeline are stateless with respect to
        the pipeline contract, so this is an alias for :meth:`transform`.

        Args:
            data: Input data to transform

        Returns:
            Transformed data
        """
        return self.transform(data)
