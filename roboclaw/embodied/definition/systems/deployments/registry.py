"""Deployment registry."""

from __future__ import annotations

from roboclaw.embodied.definition.systems.deployments.model import DeploymentProfile


class DeploymentRegistry:
    """Register deployment profiles."""

    def __init__(self) -> None:
        self._entries: dict[str, DeploymentProfile] = {}

    def register(self, profile: DeploymentProfile) -> None:
        if profile.id in self._entries:
            raise ValueError(f"Deployment profile '{profile.id}' is already registered.")
        self._entries[profile.id] = profile

    def get(self, profile_id: str) -> DeploymentProfile:
        try:
            return self._entries[profile_id]
        except KeyError as exc:
            raise KeyError(f"Unknown deployment profile '{profile_id}'.") from exc

    def for_assembly(self, assembly_id: str) -> tuple[DeploymentProfile, ...]:
        return tuple(
            profile for profile in self._entries.values() if profile.assembly_id == assembly_id
        )
