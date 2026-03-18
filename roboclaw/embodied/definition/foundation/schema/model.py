"""Shared schema types used across embodied robots, sensors, carriers, and transports."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

try:
    from enum import StrEnum
except ImportError:  # pragma: no cover - Python < 3.11 fallback for local tooling.
    class StrEnum(str, Enum):
        """Fallback for Python versions without enum.StrEnum."""


class RobotType(StrEnum):
    """High-level robot category."""

    ARM = "arm"
    HUMANOID = "humanoid"
    MOBILE_BASE = "mobile_base"
    DRONE = "drone"
    HAND = "hand"
    DUAL_ARM = "dual_arm"
    OTHER = "other"


class SensorKind(StrEnum):
    """Sensor family."""

    CAMERA = "camera"
    DEPTH_CAMERA = "depth_camera"
    JOINT_STATE = "joint_state"
    FORCE_TORQUE = "force_torque"
    IMU = "imu"
    LIDAR = "lidar"
    OTHER = "other"


class CarrierKind(StrEnum):
    """Execution carrier family."""

    REAL = "real"
    SIM = "sim"


class SimulatorKind(StrEnum):
    """Simulation backend kind."""

    NONE = "none"
    PYBULLET = "pybullet"
    GAZEBO = "gazebo"
    ISAAC_SIM = "isaac_sim"
    MUJOCO = "mujoco"
    CUSTOM = "custom"


class TransportKind(StrEnum):
    """Transport boundary between RoboClaw and the execution target."""

    ROS2 = "ros2"


class CapabilityFamily(StrEnum):
    """Reusable capability families consumed by semantic skills."""

    LIFECYCLE = "lifecycle"
    JOINT_MOTION = "joint_motion"
    CARTESIAN_MOTION = "cartesian_motion"
    BASE_MOTION = "base_motion"
    HEAD_MOTION = "head_motion"
    END_EFFECTOR = "end_effector"
    CAMERA = "camera"
    CALIBRATION = "calibration"
    DIAGNOSTICS = "diagnostics"
    RECOVERY = "recovery"
    NAMED_POSE = "named_pose"
    TORQUE_CONTROL = "torque_control"


class PrimitiveKind(StrEnum):
    """Primitive intent category."""

    MOTION = "motion"
    END_EFFECTOR = "end_effector"
    POSE = "pose"
    PERCEPTION = "perception"
    MAINTENANCE = "maintenance"


class ValueUnit(StrEnum):
    """Unit marker for action parameters and observations."""

    UNITLESS = "unitless"
    METER = "m"
    RADIAN = "rad"
    DEGREE = "deg"
    SECOND = "s"
    METER_PER_SECOND = "m/s"
    RADIAN_PER_SECOND = "rad/s"
    DEGREE_PER_SECOND = "deg/s"
    NEWTON = "N"
    NEWTON_METER = "N*m"
    PERCENT = "%"


class CompletionSemantics(StrEnum):
    """How an action reports completion."""

    COMMAND_ACCEPTED = "command_accepted"
    GOAL_REACHED = "goal_reached"
    STABLE_HOLD = "stable_hold"
    EVENT_CONFIRMED = "event_confirmed"


class CommandMode(StrEnum):
    """Normalized command mode for one action primitive."""

    POSITION = "position"
    VELOCITY = "velocity"
    TORQUE = "torque"
    CARTESIAN_DELTA = "cartesian_delta"
    CARTESIAN_POSE = "cartesian_pose"
    WAYPOINT = "waypoint"
    MISSION = "mission"
    OFFBOARD = "offboard"
    DISCRETE_TRIGGER = "discrete_trigger"
    OTHER = "other"


class FeedbackMode(StrEnum):
    """Expected feedback mode for one action primitive."""

    COMMAND_ACCEPT = "command_accept"
    STATE_FEEDBACK = "state_feedback"
    TRAJECTORY_STATUS = "trajectory_status"
    EVENT_STATUS = "event_status"
    NONE = "none"


class HealthLevel(StrEnum):
    """Normalized health level names."""

    OK = "ok"
    WARN = "warn"
    ERROR = "error"
    STALE = "stale"


@dataclass(frozen=True)
class ToleranceSpec:
    """Tolerance rules used by action parameters or action completion."""

    absolute: float | None = None
    relative: float | None = None
    settle_time_s: float | None = None


@dataclass(frozen=True)
class CompletionSpec:
    """Completion behavior for one action."""

    semantics: CompletionSemantics
    timeout_s: float | None = None
    success_states: tuple[str, ...] = field(default_factory=lambda: ("succeeded",))
    notes: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ActionSchema:
    """Machine-checkable action schema for one primitive."""

    id: str
    command_mode: CommandMode
    feedback_mode: FeedbackMode = FeedbackMode.STATE_FEEDBACK
    parameter_order: tuple[str, ...] = field(default_factory=tuple)
    command_frame: str | None = None
    command_rate_hz: float | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("Action schema id cannot be empty.")
        if self.command_frame is not None and not self.command_frame.strip():
            raise ValueError("Action schema command_frame cannot be empty when specified.")
        if self.command_rate_hz is not None and self.command_rate_hz <= 0:
            raise ValueError("Action schema command_rate_hz must be > 0 when specified.")
        if any(not name.strip() for name in self.parameter_order):
            raise ValueError("Action schema parameter_order cannot contain empty values.")


@dataclass(frozen=True)
class ParameterSpec:
    """One primitive parameter."""

    name: str
    value_type: str
    description: str
    required: bool = False
    unit: ValueUnit = ValueUnit.UNITLESS
    frame: str | None = None
    tolerance: ToleranceSpec | None = None


@dataclass(frozen=True)
class ObservationFieldSpec:
    """One field in a robot observation schema."""

    name: str
    value_type: str
    description: str
    unit: ValueUnit = ValueUnit.UNITLESS
    frame: str | None = None


@dataclass(frozen=True)
class ObservationSchema:
    """Machine-checkable observation schema for one robot profile."""

    id: str
    fields: tuple[ObservationFieldSpec, ...]
    frequency_hz: float | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class HealthFieldSpec:
    """One field in a health schema."""

    name: str
    value_type: str
    description: str


@dataclass(frozen=True)
class HealthSchema:
    """Machine-checkable health schema for one robot profile."""

    id: str
    fields: tuple[HealthFieldSpec, ...]
    severity_levels: tuple[HealthLevel, ...] = field(
        default_factory=lambda: (
            HealthLevel.OK,
            HealthLevel.WARN,
            HealthLevel.ERROR,
            HealthLevel.STALE,
        )
    )
    notes: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class SafetyProfile:
    """Default safety policy for a robot or assembly."""

    emergency_stop_required: bool = True
    supports_soft_stop: bool = True
    default_reset_mode: str = "home"
    notes: tuple[str, ...] = field(default_factory=tuple)
