"""Document generation package for AgentFlow.

Exports:
    DocumentGenerator -- the simple file-based generator (from ``.document``)
    ContentParser     -- content validation/parsing helpers
    DocumentSchema    -- pydantic schema for document validation
    DocumentNode      -- agent-node wrapper for document generation

Note: ``agentflow.core.document.document_generator`` imports ``pdfkit`` at
module level, but pdfkit is *not* a declared dependency (it is absent from
``requirements.txt`` and from the virtual environment).  When the real
package is unavailable, a minimal stand-in is registered in ``sys.modules``
so that importing the module does not fail; actually generating PDFs raises
a clear error instead.
"""

import sys

try:
    import pdfkit  # noqa: F401
except ImportError:  # pdfkit is an undeclared optional dependency
    import types

    _pdfkit = types.ModuleType("pdfkit")
    _pdfkit.__doc__ = (
        "Stand-in for the undeclared optional 'pdfkit' dependency. "
        "Registered by agentflow.core.document so that importing "
        "document_generator does not fail when pdfkit is not installed."
    )

    def _pdfkit_unavailable(*args, **kwargs):
        raise RuntimeError(
            "pdfkit is not installed; PDF document generation is "
            "unavailable. Install pdfkit (and the wkhtmltopdf binary) "
            "to enable PDF output."
        )

    _pdfkit.from_string = _pdfkit_unavailable
    _pdfkit.from_url = _pdfkit_unavailable
    _pdfkit.configuration = lambda **kwargs: None
    sys.modules.setdefault("pdfkit", _pdfkit)
    del types

from .document import DocumentGenerator
from .document_generator import ContentParser, DocumentSchema
from .document_node import DocumentNode

__all__ = [
    "ContentParser",
    "DocumentGenerator",
    "DocumentNode",
    "DocumentSchema",
]
