"""ROS2 transport contracts for embodied assemblies."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Ros2TopicSpec:
    """ROS2 topic contract."""

    name: str
    message_type: str
    path: str
    description: str = ""


@dataclass(frozen=True)
class Ros2ServiceSpec:
    """ROS2 service contract."""

    name: str
    service_type: str
    path: str
    description: str = ""


@dataclass(frozen=True)
class Ros2ActionSpec:
    """ROS2 action contract."""

    name: str
    action_type: str
    path: str
    description: str = ""


@dataclass(frozen=True)
class Ros2InterfaceBundle:
    """Canonical ROS2 interface set for one execution target."""

    namespace: str
    topics: tuple[Ros2TopicSpec, ...] = ()
    services: tuple[Ros2ServiceSpec, ...] = ()
    actions: tuple[Ros2ActionSpec, ...] = ()
    frames: tuple[str, ...] = ()

    def topic(self, name: str) -> Ros2TopicSpec | None:
        return next((item for item in self.topics if item.name == name), None)

    def service(self, name: str) -> Ros2ServiceSpec | None:
        return next((item for item in self.services if item.name == name), None)

    def action(self, name: str) -> Ros2ActionSpec | None:
        return next((item for item in self.actions if item.name == name), None)


def canonical_ros2_namespace(system_id: str, target_id: str) -> str:
    """Return the canonical ROS2 namespace for one composed assembly target."""

    cleaned_system = system_id.strip().strip("/")
    cleaned_target = target_id.strip().strip("/")
    return f"/roboclaw/{cleaned_system}/{cleaned_target}"


def build_standard_ros2_contract(
    system_id: str,
    target_id: str,
    *,
    frames: tuple[str, ...] = (),
    extra_topics: tuple[Ros2TopicSpec, ...] = (),
    extra_services: tuple[Ros2ServiceSpec, ...] = (),
    extra_actions: tuple[Ros2ActionSpec, ...] = (),
) -> Ros2InterfaceBundle:
    """Build the standard ROS2 contract for one composed assembly target."""

    namespace = canonical_ros2_namespace(system_id, target_id)
    topics = (
        Ros2TopicSpec(
            name="state",
            message_type="roboclaw_msgs/msg/AssemblyState",
            path=f"{namespace}/state",
            description="Normalized state stream for the assembly.",
        ),
        Ros2TopicSpec(
            name="health",
            message_type="diagnostic_msgs/msg/DiagnosticArray",
            path=f"{namespace}/health",
            description="Health, warnings, and diagnostic updates.",
        ),
        Ros2TopicSpec(
            name="events",
            message_type="roboclaw_msgs/msg/AssemblyEvent",
            path=f"{namespace}/events",
            description="Action lifecycle, trace, and failure events.",
        ),
        Ros2TopicSpec(
            name="joint_states",
            message_type="sensor_msgs/msg/JointState",
            path=f"{namespace}/joint_states",
            description="Joint state projection for all attached robots.",
        ),
        *extra_topics,
    )
    services = (
        Ros2ServiceSpec(
            name="connect",
            service_type="roboclaw_msgs/srv/ConnectAssembly",
            path=f"{namespace}/connect",
            description="Connect to the selected real or simulated carrier.",
        ),
        Ros2ServiceSpec(
            name="disconnect",
            service_type="roboclaw_msgs/srv/DisconnectAssembly",
            path=f"{namespace}/disconnect",
            description="Disconnect while preserving adapter state where possible.",
        ),
        Ros2ServiceSpec(
            name="stop",
            service_type="roboclaw_msgs/srv/StopAssembly",
            path=f"{namespace}/stop",
            description="Immediately stop active motion or running tasks.",
        ),
        Ros2ServiceSpec(
            name="reset",
            service_type="roboclaw_msgs/srv/ResetAssembly",
            path=f"{namespace}/reset",
            description="Reset into a known named mode such as home or ready.",
        ),
        Ros2ServiceSpec(
            name="recover",
            service_type="roboclaw_msgs/srv/RecoverAssembly",
            path=f"{namespace}/recover",
            description="Run a recovery procedure after failure or interruption.",
        ),
        Ros2ServiceSpec(
            name="list_calibration_targets",
            service_type="roboclaw_msgs/srv/ListCalibrationTargets",
            path=f"{namespace}/list_calibration_targets",
            description="List calibration items exposed by attached robots or sensors.",
        ),
        Ros2ServiceSpec(
            name="sensor_snapshot",
            service_type="roboclaw_msgs/srv/CaptureSensor",
            path=f"{namespace}/sensor_snapshot",
            description="Capture or fetch one sensor payload.",
        ),
        Ros2ServiceSpec(
            name="debug_snapshot",
            service_type="roboclaw_msgs/srv/DebugSnapshot",
            path=f"{namespace}/debug_snapshot",
            description="Collect normalized debugging artifacts.",
        ),
        *extra_services,
    )
    actions = (
        Ros2ActionSpec(
            name="execute_primitive",
            action_type="roboclaw_msgs/action/ExecutePrimitive",
            path=f"{namespace}/execute_primitive",
            description="Execute one normalized primitive request.",
        ),
        Ros2ActionSpec(
            name="start_calibration",
            action_type="roboclaw_msgs/action/RunCalibration",
            path=f"{namespace}/start_calibration",
            description="Run calibration as an asynchronous task.",
        ),
        *extra_actions,
    )
    return Ros2InterfaceBundle(
        namespace=namespace,
        topics=topics,
        services=services,
        actions=actions,
        frames=frames,
    )
