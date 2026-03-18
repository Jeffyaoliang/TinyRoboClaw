"""Assembly registry."""

from __future__ import annotations

from roboclaw.embodied.definition.systems.assemblies.model import AssemblyManifest


class AssemblyRegistry:
    """Simple in-memory assembly registry."""

    def __init__(self) -> None:
        self._entries: dict[str, AssemblyManifest] = {}

    def register(self, manifest: AssemblyManifest) -> None:
        if manifest.id in self._entries:
            raise ValueError(f"Assembly '{manifest.id}' is already registered.")
        self._entries[manifest.id] = manifest

    def get(self, assembly_id: str) -> AssemblyManifest:
        try:
            return self._entries[assembly_id]
        except KeyError as exc:
            raise KeyError(f"Unknown assembly '{assembly_id}'.") from exc

    def list(self) -> tuple[AssemblyManifest, ...]:
        return tuple(sorted(self._entries.values(), key=lambda item: item.id))
