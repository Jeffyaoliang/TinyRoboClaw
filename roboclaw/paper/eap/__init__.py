from roboclaw.paper.eap.trajectory import Trajectory, TrajectoryDataset
from roboclaw.paper.eap.judge import SuccessJudge
from roboclaw.paper.eap.engine import EAPEngine

__all__ = ["Trajectory", "TrajectoryDataset", "SuccessJudge", "EAPEngine"]

# GPU-parallel engine — lazy import (requires torch + Isaac Lab)
def get_parallel_engine():
    from roboclaw.paper.eap.parallel_engine import ParallelEAPEngine, ParallelEAPConfig
    return ParallelEAPEngine, ParallelEAPConfig
