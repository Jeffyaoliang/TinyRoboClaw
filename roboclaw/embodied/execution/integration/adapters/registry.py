"""Adapter registry."""

from __future__ import annotations

from roboclaw.embodied.execution.integration.adapters.model import AdapterBinding


class AdapterRegistry:
    """Register available adapter bindings."""

    def __init__(self) -> None:
        self._entries: dict[str, AdapterBinding] = {}

    def register(self, binding: AdapterBinding) -> None:
        if binding.id in self._entries:
            raise ValueError(f"Adapter '{binding.id}' is already registered.")
        self._entries[binding.id] = binding

    def get(self, adapter_id: str) -> AdapterBinding:
        try:
            return self._entries[adapter_id]
        except KeyError as exc:
            raise KeyError(f"Unknown adapter '{adapter_id}'.") from exc

    def for_assembly(self, assembly_id: str) -> tuple[AdapterBinding, ...]:
        return tuple(
            entry for entry in self._entries.values() if entry.assembly_id == assembly_id
        )
