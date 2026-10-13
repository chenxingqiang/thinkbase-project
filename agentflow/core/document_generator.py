"""Thin compatibility shim for the legacy ``agentflow.core.document_generator`` path.

The real implementation lives in
:mod:`agentflow.core.document.document_generator`; every public symbol is
re-exported here so historical imports keep working.

Note: importing this shim also initializes the ``agentflow.core.document``
package, which installs a lightweight ``pdfkit`` stand-in when the real
``pdfkit`` package is unavailable (it is not a declared dependency in
``requirements.txt``).
"""

from agentflow.core.document.document_generator import (  # noqa: F401
    ContentParser,
    DocumentGenerator,
    DocumentSchema,
)

__all__ = [
    "ContentParser",
    "DocumentGenerator",
    "DocumentSchema",
]
