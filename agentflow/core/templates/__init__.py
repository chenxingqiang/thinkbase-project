"""Workflow templates.

Loads the Jinja2 ``.j2`` templates shipped in this directory and
renders them with validated parameters.

Example:
    >>> from agentflow.core.templates import WorkflowTemplate
    >>> tpl = WorkflowTemplate("research_report")
    >>> tpl.render({"title": "AgentISA Evaluation", "author": "AgentFlow"})
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_TEMPLATE_DIR = Path(__file__).parent


@dataclass
class TemplateParameter:
    """A named parameter accepted by a workflow template."""

    name: str
    description: str = ""
    required: bool = False
    default: Any = None
    type: str = "str"

    def coerce(self, value: Any) -> Any:
        """Coerce ``value`` to this parameter's declared type."""
        if value is None:
            return self.default
        if self.type == "str":
            return str(value)
        if self.type in ("int", "float"):
            return float(value) if self.type == "float" else int(value)
        if self.type == "list" and isinstance(value, str):
            return [v.strip() for v in value.split(",") if v.strip()]
        return value


class WorkflowTemplate:
    """A named Jinja2 template with parameter validation.

    Templates are resolved from the ``.j2`` files next to this module
    (``default``, ``research_report``, ...).  Unknown names fall back
    to ``default``.
    """

    def __init__(
        self,
        name: str = "default",
        parameters: Optional[List[TemplateParameter]] = None,
        template_dir: Optional[Path] = None,
    ):
        self.name = name
        self.template_dir = Path(template_dir) if template_dir else _TEMPLATE_DIR
        self.template_path = self._resolve(name)
        self.parameters: List[TemplateParameter] = parameters or []
        self._text = self.template_path.read_text(encoding="utf-8")

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------
    def _resolve(self, name: str) -> Path:
        candidate = self.template_dir / f"{name}.j2"
        if not candidate.is_file():
            fallback = self.template_dir / "default.j2"
            logger.warning(
                "Template %r not found in %s; falling back to default",
                name,
                self.template_dir,
            )
            candidate = fallback
        return candidate

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------
    def add_parameter(self, parameter: TemplateParameter) -> None:
        """Declare an additional accepted parameter."""
        self.parameters.append(parameter)

    def validate_parameters(self, params: Dict[str, Any]) -> List[str]:
        """Return the list of missing required parameter names."""
        declared = {p.name for p in self.parameters}
        return [
            p.name
            for p in self.parameters
            if p.required and params.get(p.name) is None
        ] + [k for k in params if k not in declared and self.parameters]

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render(self, params: Optional[Dict[str, Any]] = None) -> str:
        """Render the template with the given parameters."""
        params = dict(params or {})
        missing = self.validate_parameters(params)
        if missing:
            raise ValueError(f"Missing required template parameters: {missing}")
        for parameter in self.parameters:
            params.setdefault(parameter.name, parameter.default)
            params[parameter.name] = parameter.coerce(params.get(parameter.name))

        try:
            from jinja2 import Environment, StrictUndefined, Undefined
            from jinja2 import FileSystemLoader
        except ImportError:  # pragma: no cover - jinja2 is a hard dep
            raise RuntimeError(
                "jinja2 is required to render workflow templates"
            )

        environment = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            undefined=Undefined,
            keep_trailing_newline=True,
        )
        return environment.from_string(self._text).render(**params)

    def render_to_file(self, params: Dict[str, Any], output_path: Path) -> Path:
        """Render and write the result to ``output_path``."""
        output_path = Path(output_path)
        output_path.write_text(self.render(params), encoding="utf-8")
        return output_path

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @classmethod
    def list_templates(cls, template_dir: Optional[Path] = None) -> List[str]:
        """List the available template names."""
        directory = Path(template_dir) if template_dir else _TEMPLATE_DIR
        return sorted(p.stem for p in directory.glob("*.j2"))

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"WorkflowTemplate({self.name!r}, params={[p.name for p in self.parameters]})"


__all__ = ["WorkflowTemplate", "TemplateParameter"]
