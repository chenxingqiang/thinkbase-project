"""Transformations package for AgentFlow.

Exports the transformation strategies and the
:class:`~agentflow.transformations.pipeline.TransformationPipeline`.

Backward-compatibility aliases
------------------------------
Two modules that used to exist as standalone files are served from this
package until real modules are (re-)created:

* ``agentflow.transformations.text`` -- the config-dict driven,
  string-level text strategy, exposed as
  :class:`~agentflow.transformations.advanced_strategies.StringTextTransformationStrategy`.
* ``agentflow.transformations.specialized_strategies`` -- the time
  series and anomaly detection strategies.

The aliases are registered in ``sys.modules`` when this package is
imported, but only as long as no real module with the same name exists
on disk, so dropping real files back in always takes precedence.
"""

import os as _os
import sys as _sys
import types as _types

from .advanced_strategies import (
    AdvancedTransformationStrategy,
    AnomalyDetectionStrategy,
    FeatureEngineeringStrategy,
    OutlierRemovalStrategy,
    StringTextTransformationStrategy,
    TextTransformationStrategy,
    TimeSeriesTransformationStrategy,
)
from .pipeline import TransformationPipeline

__all__ = [
    "AdvancedTransformationStrategy",
    "AnomalyDetectionStrategy",
    "FeatureEngineeringStrategy",
    "OutlierRemovalStrategy",
    "StringTextTransformationStrategy",
    "TextTransformationStrategy",
    "TimeSeriesTransformationStrategy",
    "TransformationPipeline",
]

_PACKAGE_DIR = _os.path.dirname(_os.path.abspath(__file__))


def _register_alias_module(name: str, attrs: dict):
    """Register *name* in ``sys.modules`` unless a real module exists.

    Returns the synthetic module, or ``None`` when a real module file is
    present (in which case the normal import machinery handles it).
    """
    filename = name.rpartition(".")[2] + ".py"
    if _os.path.isfile(_os.path.join(_PACKAGE_DIR, filename)):
        return None
    module = _types.ModuleType(name)
    module.__dict__.update(attrs)
    module.__all__ = list(attrs)
    _sys.modules.setdefault(name, module)
    return module


# ``agentflow.transformations.text`` historically exposed the config-dict
# driven, string-level text strategy under the name TextTransformationStrategy.
_text_alias = _register_alias_module(
    "agentflow.transformations.text",
    {"TextTransformationStrategy": StringTextTransformationStrategy},
)
if _text_alias is not None:
    text = _text_alias

# ``agentflow.transformations.specialized_strategies`` historically exposed
# the time-series and anomaly-detection strategies.
_specialized_alias = _register_alias_module(
    "agentflow.transformations.specialized_strategies",
    {
        "TimeSeriesTransformationStrategy": TimeSeriesTransformationStrategy,
        "AnomalyDetectionStrategy": AnomalyDetectionStrategy,
    },
)
if _specialized_alias is not None:
    specialized_strategies = _specialized_alias
