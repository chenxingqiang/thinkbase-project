"""Processors package."""

__all__ = ["TransformProcessor"]


def __getattr__(name):
    if name == "TransformProcessor":
        from .transformers import TransformProcessor

        return TransformProcessor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
