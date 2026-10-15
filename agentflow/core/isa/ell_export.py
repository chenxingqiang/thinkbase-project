"""Export AgentISA instruction streams into the ELL integration layer.

An AgentISA instruction stream is a program; ELL's message store is a
conversation trace.  This module bridges the two: an executed
instruction stream is converted into an ELL ``Message`` trace (one
``user`` message describing the instruction invocation, one
``assistant`` message recording its execution metrics), registered as
an ``ELLWorkflow`` with one step per instruction, and can be
serialized to a JSONL trace file that is inspectable without the
runtime.

The mapping is intentionally conservative -- only facts the runtime
actually observed (instruction descriptors and post-execution
metrics) are recorded; no model output is invented.

Example:
    from agentflow.core.isa.ell_export import export_stream_to_ell
    workflow, messages = export_stream_to_ell(stream, name="triage")
    save_ell_trace("trace.jsonl", workflow, messages)
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agentflow.ell import (
    ELLWorkflow,
    Message,
    MessageRole,
    MessageType,
    assistant,
    system,
    user,
)
from agentflow.ell.integration import ell_integration

__all__ = [
    "describe_instruction",
    "stream_to_workflow",
    "stream_to_messages",
    "export_stream_to_ell",
    "save_ell_trace",
    "load_ell_trace",
    "render_trace",
]


def describe_instruction(instr: Any) -> Dict[str, Any]:
    """The observable descriptor of one instruction in the stream."""
    return {
        "id": getattr(instr, "id", None),
        "name": getattr(instr, "name", None),
        "type": getattr(getattr(instr, "type", None), "value", None)
        or str(getattr(instr, "type", "")),
        "description": getattr(instr, "description", ""),
        "dependencies": list(getattr(instr, "dependencies", []) or []),
        "outputs": list(getattr(instr, "outputs", []) or []),
        "side_effects": bool(getattr(instr, "side_effects", False)),
    }


def stream_to_workflow(
    instructions: List[Any],
    name: str = "agentisa-stream",
    description: str = "An AgentISA instruction stream",
) -> ELLWorkflow:
    """Build an ``ELLWorkflow`` with one step per instruction."""
    workflow = ELLWorkflow(
        name=name,
        description=description,
        metadata={"instruction_count": len(instructions)},
    )
    for instr in instructions:
        descriptor = describe_instruction(instr)
        descriptor["step"] = len(workflow.steps)
        workflow.add_step(descriptor)
    return workflow


def _execution_record(instr: Any) -> Dict[str, Any]:
    """Post-execution metrics of one instruction (ints stay ints)."""
    metrics = getattr(instr, "metrics", None)
    if metrics is None or not getattr(metrics, "total_executions", 0):
        return {"executed": False}
    record: Dict[str, Any] = {"executed": True}
    for field in (
        "total_executions", "cache_hits", "cache_misses",
        "cache_hit_rate", "average_execution_time", "success_rate",
        "error_count",
    ):
        value = getattr(metrics, field, None)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            record[field] = value
        elif isinstance(value, float):
            record[field] = round(value, 6)
    return record


def stream_to_messages(
    instructions: List[Any],
    stream_name: str = "agentisa-stream",
    execution_summary: Optional[Dict[str, Any]] = None,
    eliminated: Optional[List[str]] = None,
) -> List[Message]:
    """Convert an (executed) instruction stream into ELL messages.

    Emits a ``system`` header, then one ``user`` invocation descriptor
    and one ``assistant`` execution record per instruction.  The
    structured facts ride in ``metadata`` so the rendered ``content``
    stays human-readable.
    """
    eliminated = set(eliminated or [])
    messages: List[Message] = [
        system(
            f"AgentISA instruction stream '{stream_name}': "
            f"{len(instructions)} instructions "
            + (
                f"({len(eliminated)} eliminated by optimization: "
                f"{', '.join(sorted(eliminated))})"
                if eliminated else ""
            ),
            stream_name=stream_name,
            instruction_count=len(instructions),
            execution_summary=execution_summary or {},
        )
    ]
    for instr in instructions:
        descriptor = describe_instruction(instr)
        deps = ", ".join(descriptor["dependencies"]) or "none"
        outputs = ", ".join(descriptor["outputs"]) or "none"
        messages.append(user(
            f"-> {descriptor['name']} [{descriptor['type']}] "
            f"(deps: {deps}; produces: {outputs}"
            + ("; side effects" if descriptor["side_effects"] else "")
            + ")",
            instruction=descriptor,
            eliminated=descriptor["id"] in eliminated,
        ))
        record = _execution_record(instr)
        cache_note = ""
        if record.get("cache_hits"):
            cache_note = (
                f", {record['cache_hits']} cache hits "
                f"({record.get('cache_hit_rate', 0):.1%} hit rate)"
            )
        messages.append(assistant(
            f"<- {descriptor['name']}: "
            + (
                f"{record['total_executions']} executions"
                f"{cache_note}, avg {record.get('average_execution_time', 0):.4f}s"
                if record.get("executed")
                else "not executed"
            ),
            instruction={"id": descriptor["id"], "name": descriptor["name"]},
            execution=record,
        ))
    return messages


async def export_stream_to_ell(
    instructions: List[Any],
    name: str = "agentisa-stream",
    description: str = "An AgentISA instruction stream",
    execution_summary: Optional[Dict[str, Any]] = None,
    eliminated: Optional[List[str]] = None,
) -> Tuple[ELLWorkflow, List[Message]]:
    """Register the stream as a workflow and record its trace in ELL.

    The workflow is registered with the ``ELLIntegration`` singleton
    and every message is processed through ``process_message``, so
    ``ell_integration.get_messages()`` afterwards contains exactly the
    stream's trace.
    """
    workflow = stream_to_workflow(instructions, name, description)
    messages = stream_to_messages(
        instructions, stream_name=name,
        execution_summary=execution_summary, eliminated=eliminated,
    )
    ell_integration.register_workflow(workflow.id, workflow)
    for message in messages:
        await ell_integration.process_message(message)
    return workflow, messages


def _message_to_record(message: Message) -> Dict[str, Any]:
    """Serialize one message to a JSON-safe record."""
    return {
        "role": getattr(message.role, "value", message.role),
        "type": getattr(message.type, "value", message.type),
        "content": str(message.content) if message.content is not None else "",
        "metadata": dict(message.metadata or {}),
    }


def save_ell_trace(
    path: Any,
    workflow: ELLWorkflow,
    messages: List[Message],
) -> Path:
    """Serialize the workflow and its message trace to a JSONL file.

    The first line is the workflow record; every following line is one
    message.  The format is self-describing and round-trips through
    :func:`load_ell_trace`.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"kind": "workflow", **workflow.to_dict()},
                            ensure_ascii=False) + "\n")
        for message in messages:
            fh.write(json.dumps(
                {"kind": "message", **_message_to_record(message)},
                ensure_ascii=False) + "\n")
    return path


def load_ell_trace(path: Any) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Load a trace written by :func:`save_ell_trace`.

    Returns ``(workflow_record, message_records)``.
    """
    workflow: Optional[Dict[str, Any]] = None
    messages: List[Dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            record = json.loads(line)
            if record.get("kind") == "workflow":
                workflow = record
            elif record.get("kind") == "message":
                messages.append(record)
    if workflow is None:
        raise ValueError(f"no workflow record in trace: {path}")
    return workflow, messages


def render_trace(messages: List[Message], width: int = 72) -> str:
    """Render a message trace as a compact, human-readable listing."""
    lines: List[str] = []
    for message in messages:
        role = getattr(message.role, "value", message.role)
        prefix = f"[{role.upper():9s}]"
        content = str(message.content) if message.content is not None else ""
        if len(content) > width:
            content = content[: width - 3] + "..."
        lines.append(f"{prefix} {content}")
    return "\n".join(lines)
