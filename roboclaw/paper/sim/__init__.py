from roboclaw.paper.sim.base_env import BaseEnvironment
from roboclaw.paper.sim.tabletop_env import TabletopEnv

__all__ = ["BaseEnvironment", "TabletopEnv"]


# Hardware environments — lazy import (require external SDKs)
def get_k1_env():
    from roboclaw.paper.sim.k1_env import K1Config, K1Env
    return K1Config, K1Env


def get_piper_env():
    from roboclaw.paper.sim.piper_env import PiperConfig, PiperEnv
    return PiperConfig, PiperEnv


# Simulation environments — lazy import (require specific packages)
def get_piper_mujoco_env():
    from roboclaw.paper.sim.piper_mujoco_env import PiperMujocoConfig, PiperMujocoEnv
    return PiperMujocoConfig, PiperMujocoEnv


def get_k1_mujoco_env():
    from roboclaw.paper.sim.k1_mujoco_env import K1MujocoConfig, K1MujocoEnv
    return K1MujocoConfig, K1MujocoEnv


def get_isaac_lab_env():
    from roboclaw.paper.sim.isaac_lab_env import IsaacLabConfig, IsaacLabEnv
    return IsaacLabConfig, IsaacLabEnv


def get_gazebo_env():
    from roboclaw.paper.sim.gazebo_env import GazeboConfig, GazeboEnv
    return GazeboConfig, GazeboEnv
