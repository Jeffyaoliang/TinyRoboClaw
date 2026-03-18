"""Runtime session manager."""

from __future__ import annotations

from roboclaw.embodied.execution.orchestration.runtime.model import RuntimeSession, RuntimeStatus


class RuntimeManager:
    """Manage active embodied sessions by id."""

    def __init__(self) -> None:
        self._sessions: dict[str, RuntimeSession] = {}

    def create(
        self,
        *,
        session_id: str,
        assembly_id: str,
        target_id: str,
        deployment_id: str | None = None,
        adapter_id: str | None = None,
    ) -> RuntimeSession:
        if session_id in self._sessions:
            raise ValueError(f"Runtime session '{session_id}' already exists.")
        session = RuntimeSession(
            id=session_id,
            assembly_id=assembly_id,
            target_id=target_id,
            deployment_id=deployment_id,
            adapter_id=adapter_id,
        )
        self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> RuntimeSession:
        try:
            return self._sessions[session_id]
        except KeyError as exc:
            raise KeyError(f"Unknown runtime session '{session_id}'.") from exc

    def list(self) -> tuple[RuntimeSession, ...]:
        return tuple(self._sessions.values())

    def mark_status(self, session_id: str, status: RuntimeStatus, error: str | None = None) -> None:
        session = self.get(session_id)
        session.status = status
        session.last_error = error
