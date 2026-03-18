"""Simulator registry."""

from __future__ import annotations

from roboclaw.embodied.definition.systems.simulators.model import SimulatorScenario, SimulatorWorld


class SimulatorRegistry:
    """Register simulator worlds and scenarios."""

    def __init__(self) -> None:
        self._worlds: dict[str, SimulatorWorld] = {}
        self._scenarios: dict[str, SimulatorScenario] = {}

    def register_world(self, world: SimulatorWorld) -> None:
        if world.id in self._worlds:
            raise ValueError(f"Simulator world '{world.id}' is already registered.")
        self._worlds[world.id] = world

    def register_scenario(self, scenario: SimulatorScenario) -> None:
        if scenario.id in self._scenarios:
            raise ValueError(f"Simulator scenario '{scenario.id}' is already registered.")
        self._scenarios[scenario.id] = scenario

    def get_world(self, world_id: str) -> SimulatorWorld:
        try:
            return self._worlds[world_id]
        except KeyError as exc:
            raise KeyError(f"Unknown simulator world '{world_id}'.") from exc

    def get_scenario(self, scenario_id: str) -> SimulatorScenario:
        try:
            return self._scenarios[scenario_id]
        except KeyError as exc:
            raise KeyError(f"Unknown simulator scenario '{scenario_id}'.") from exc
