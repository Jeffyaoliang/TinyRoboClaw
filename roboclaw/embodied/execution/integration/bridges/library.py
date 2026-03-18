"""Built-in domain bridge contracts."""

from __future__ import annotations

from roboclaw.embodied.definition.foundation.schema import CapabilityFamily, RobotType
from roboclaw.embodied.execution.integration.bridges.model import (
    BridgeDomain,
    BridgeKind,
    ControlSurfaceSpec,
    DomainBridgeContract,
    ObservationSurfaceSpec,
)

ARM_HAND_BRIDGE = DomainBridgeContract(
    id="bridge_arm_hand_v1",
    domain=BridgeDomain.ARM_HAND,
    kind=BridgeKind.ROS2_CONTROL,
    description="Bridge profile for arm/hand manipulation systems.",
    supported_robot_types=(RobotType.ARM, RobotType.HAND, RobotType.DUAL_ARM),
    supported_capabilities=(
        CapabilityFamily.LIFECYCLE,
        CapabilityFamily.JOINT_MOTION,
        CapabilityFamily.CARTESIAN_MOTION,
        CapabilityFamily.END_EFFECTOR,
        CapabilityFamily.DIAGNOSTICS,
        CapabilityFamily.RECOVERY,
    ),
    control_surfaces=(
        ControlSurfaceSpec(
            id="joint_trajectory",
            mode="position",
            interface="ros2_action:FollowJointTrajectory",
            description="Joint-space trajectory control for arms and hands.",
        ),
        ControlSurfaceSpec(
            id="servo_delta",
            mode="velocity",
            interface="ros2_topic:/servo_twist_command",
            description="Incremental servo control for short-horizon manipulation.",
        ),
    ),
    observation_surfaces=(
        ObservationSurfaceSpec(
            id="joint_state",
            stream="joint_state",
            interface="ros2_topic:/joint_states",
            description="Joint states and effort feedback.",
            expected_rate_hz=50.0,
        ),
    ),
)

HUMANOID_WHOLE_BODY_BRIDGE = DomainBridgeContract(
    id="bridge_humanoid_whole_body_v1",
    domain=BridgeDomain.HUMANOID_WHOLE_BODY,
    kind=BridgeKind.WHOLE_BODY_CONTROLLER,
    description="Bridge profile for humanoid whole-body controllers.",
    supported_robot_types=(RobotType.HUMANOID,),
    supported_capabilities=(
        CapabilityFamily.LIFECYCLE,
        CapabilityFamily.JOINT_MOTION,
        CapabilityFamily.BASE_MOTION,
        CapabilityFamily.HEAD_MOTION,
        CapabilityFamily.DIAGNOSTICS,
        CapabilityFamily.RECOVERY,
    ),
    control_surfaces=(
        ControlSurfaceSpec(
            id="whole_body_command",
            mode="whole_body",
            interface="ros2_topic:/whole_body/command",
            description="Whole-body command surface for coordinated posture and locomotion.",
        ),
    ),
    observation_surfaces=(
        ObservationSurfaceSpec(
            id="body_state",
            stream="body_state",
            interface="ros2_topic:/whole_body/state",
            description="Normalized humanoid body state stream.",
            expected_rate_hz=30.0,
        ),
    ),
)

MOBILE_BASE_FLEET_BRIDGE = DomainBridgeContract(
    id="bridge_mobile_base_fleet_v1",
    domain=BridgeDomain.MOBILE_BASE_FLEET,
    kind=BridgeKind.NAV2_RMF,
    description="Bridge profile for mobile base and fleet navigation stacks.",
    supported_robot_types=(RobotType.MOBILE_BASE,),
    supported_capabilities=(
        CapabilityFamily.LIFECYCLE,
        CapabilityFamily.BASE_MOTION,
        CapabilityFamily.NAMED_POSE,
        CapabilityFamily.DIAGNOSTICS,
        CapabilityFamily.RECOVERY,
    ),
    control_surfaces=(
        ControlSurfaceSpec(
            id="nav_goal",
            mode="waypoint",
            interface="ros2_action:NavigateToPose",
            description="Goal-based navigation control for one mobile base.",
        ),
    ),
    observation_surfaces=(
        ObservationSurfaceSpec(
            id="odometry",
            stream="base_state",
            interface="ros2_topic:/odom",
            description="Base odometry and motion estimates.",
            expected_rate_hz=30.0,
        ),
    ),
)

DRONE_BRIDGE = DomainBridgeContract(
    id="bridge_drone_v1",
    domain=BridgeDomain.DRONE,
    kind=BridgeKind.MAVSDK_MAVLINK,
    description="Bridge profile for drone command and telemetry over MAVLink/MAVSDK.",
    supported_robot_types=(RobotType.DRONE,),
    supported_capabilities=(
        CapabilityFamily.LIFECYCLE,
        CapabilityFamily.BASE_MOTION,
        CapabilityFamily.CAMERA,
        CapabilityFamily.DIAGNOSTICS,
        CapabilityFamily.RECOVERY,
    ),
    control_surfaces=(
        ControlSurfaceSpec(
            id="offboard_setpoint",
            mode="offboard",
            interface="mavsdk:offboard",
            description="Offboard setpoint control for local velocity/pose targets.",
        ),
    ),
    observation_surfaces=(
        ObservationSurfaceSpec(
            id="flight_telemetry",
            stream="flight_state",
            interface="mavsdk:telemetry",
            description="Flight telemetry including position, velocity, and health state.",
            expected_rate_hz=20.0,
        ),
    ),
)

SIMULATOR_BRIDGE = DomainBridgeContract(
    id="bridge_simulator_v1",
    domain=BridgeDomain.SIMULATOR,
    kind=BridgeKind.SIM_RUNTIME,
    description="Bridge profile for simulator runtimes and world orchestration.",
    supported_robot_types=(
        RobotType.ARM,
        RobotType.HUMANOID,
        RobotType.MOBILE_BASE,
        RobotType.DRONE,
        RobotType.HAND,
        RobotType.DUAL_ARM,
        RobotType.OTHER,
    ),
    supported_capabilities=(
        CapabilityFamily.LIFECYCLE,
        CapabilityFamily.DIAGNOSTICS,
        CapabilityFamily.RECOVERY,
        CapabilityFamily.CALIBRATION,
    ),
    control_surfaces=(
        ControlSurfaceSpec(
            id="world_control",
            mode="scenario",
            interface="ros2_service:/sim/world_control",
            description="Pause/reset/step simulation world and scenario state.",
        ),
    ),
    observation_surfaces=(
        ObservationSurfaceSpec(
            id="sim_clock",
            stream="clock",
            interface="ros2_topic:/clock",
            description="Simulation clock and synchronization state.",
            expected_rate_hz=100.0,
        ),
    ),
)

DEFAULT_DOMAIN_BRIDGES = (
    ARM_HAND_BRIDGE,
    HUMANOID_WHOLE_BODY_BRIDGE,
    MOBILE_BASE_FLEET_BRIDGE,
    DRONE_BRIDGE,
    SIMULATOR_BRIDGE,
)

