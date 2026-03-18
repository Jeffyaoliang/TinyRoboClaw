"""Procedure registry."""

from __future__ import annotations

from roboclaw.embodied.execution.orchestration.procedures.model import ProcedureDefinition


class ProcedureRegistry:
    """Register built-in and custom procedures."""

    def __init__(self) -> None:
        self._entries: dict[str, ProcedureDefinition] = {}

    def register(self, procedure: ProcedureDefinition) -> None:
        if procedure.id in self._entries:
            raise ValueError(f"Procedure '{procedure.id}' is already registered.")
        self._entries[procedure.id] = procedure

    def get(self, procedure_id: str) -> ProcedureDefinition:
        try:
            return self._entries[procedure_id]
        except KeyError as exc:
            raise KeyError(f"Unknown procedure '{procedure_id}'.") from exc

    def list(self) -> tuple[ProcedureDefinition, ...]:
        return tuple(self._entries.values())
