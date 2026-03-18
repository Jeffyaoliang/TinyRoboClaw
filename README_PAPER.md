# 🦀 TinyRoboClaw

**First open-source reproduction of [RoboClaw](https://arxiv.org/abs/2603.11558): An Agentic Framework for Scalable Long-Horizon Robotic Tasks**

> Minimal, clean-room implementation of the RoboClaw data flywheel — analogous to what [TinyZero](https://github.com/Jiayi-Pan/TinyZero) is to DeepSeek R1-Zero.

Built on top of the [official RoboClaw framework](https://github.com/MINT-SJTU/RoboClaw), extending it with the paper's core algorithms: **EAP data engine, data flywheel, deployment supervision, and OpenPI integration**.

[![Tests](https://img.shields.io/badge/tests-152%20passed-brightgreen)]()
[![Paper](https://img.shields.io/badge/arXiv-2603.11558-b31b1b)](https://arxiv.org/abs/2603.11558)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)]()

---

## What is RoboClaw?

RoboClaw is an agentic robotics framework that unifies **data collection**, **policy learning**, and **task execution** under a single VLM-driven controller. The key innovation is **Entangled Action Pairs (EAP)** — coupling forward manipulation with inverse recovery to create self-resetting loops, enabling autonomous data collection without human resets.

**Paper results**: 25% higher success rate on long-horizon tasks, 53.7% reduction in human effort.

<p align="center">
  <img src="docs/architecture.png" alt="RoboClaw Architecture" width="700">
</p>

## Key Features

| Feature | Description |
|---------|-------------|
| **EAP Data Engine** | Forward-inverse self-resetting loops for autonomous data collection |
| **Data Flywheel** | Iterative: collect → accumulate → retrain → repeat |
| **VLM Agent** | Structured memory (r_t, g_t, w_t) + Chain-of-Thought reasoning |
| **6 MCP Tools** | StartPolicy, TerminatePolicy, SwitchPolicy, EnvSummary, FetchRobotStats, CallHuman |
| **Skills Layer** | Reusable `data-collection` and `long-horizon-execution` skill orchestration |
| **Process Supervision** | Failure detection (non-degrading vs degrading) + automatic recovery |
| **Policy Pool** | Multi-subtask policy management for long-horizon tasks |
| **Deployment Recycling** | Deployment trajectories flow back to training dataset |
| **OpenPI Integration** | π0/π0.5 VLA inference (HTTP + WebSocket) + LoRA fine-tuning |
| **152 Tests** | Unit + end-to-end + integration, all passing |

## Supported Platforms

|  | MuJoCo | Isaac Lab (GPU) | Gazebo (ROS2) | Real Robot |
|---|:---:|:---:|:---:|:---:|
| **Agilex Piper** (6-DOF) | ✅ | ✅ | ✅ | ✅ |
| **Unitree K1** (dual-arm 7+7-DOF) | ✅ | ✅ | ✅ | ✅ |

## Quick Start

### Installation

```bash
git clone https://github.com/Jeffyaoliang/TinyRoboClaw.git
cd TinyRoboClaw
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install numpy Pillow
```

### Run EAP Data Collection (Mock Mode)

```bash
python -m roboclaw.paper.cli collect --task pick_place --iterations 3 --episodes 20 --mock
```

### Run Long-Horizon Deployment (Mock Mode)

```bash
python -m roboclaw.paper.cli deploy --task "organize the table" --mock
```

### Run Tests

```bash
pip install pytest pytest-asyncio
pytest tests/test_paper_*.py -v
```

### Full Pipeline (Python API)

```python
from roboclaw.paper.integration import create_full_pipeline
from roboclaw.paper.config import PaperConfig
from roboclaw.paper.sim import TabletopEnv
from roboclaw.paper.policy import MockPolicy

config = PaperConfig(task_name="pick_place", task_instruction="pick up the block")
env = TabletopEnv(config.sim)
provider = ...  # Your LLM provider

components = create_full_pipeline(
    config, env, provider,
    forward_policy=MockPolicy(name="fwd"),
    inverse_policy=MockPolicy(name="inv"),
)

# Run data flywheel
results = await components["flywheel"].run(num_iterations=5, mock_training=True)
```

## With Real Robots

### OpenPI Setup (on GPU server)

```bash
bash scripts/setup_openpi.sh install    # Install OpenPI
bash scripts/setup_openpi.sh serve-bg   # Start π0-FAST inference server
bash scripts/setup_openpi.sh test       # Verify connectivity
```

### Piper Arm

```python
from roboclaw.paper.sim.piper_env import PiperEnv, PiperConfig

env = PiperEnv(PiperConfig(can_port="can0", camera_device=0))
obs = env.reset()
# ... run EAP with real robot
```

### K1 Humanoid

```python
from roboclaw.paper.sim.k1_env import K1Env, K1Config

env = K1Env(K1Config(arm_id=0))  # Left arm
obs = env.reset()
```

### GPU-Parallel Collection (Isaac Lab)

```python
from roboclaw.paper.sim.isaac_lab_env import IsaacLabEnv, IsaacLabConfig

env = IsaacLabEnv(IsaacLabConfig(num_envs=512, device="cuda:0"))  # 512 parallel envs
```

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   VLM Agent (GPT-4o / Qwen-VL)      │
│                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │ Memory   │  │   CoT    │  │   Skills Layer   │  │
│  │ r_t,g_t, │  │ Reasoning│  │ data-collection  │  │
│  │ w_t      │  │          │  │ long-horizon     │  │
│  └──────────┘  └──────────┘  └──────────────────┘  │
└────────────────────────┬────────────────────────────┘
                         │ MCP Interface
┌────────────────────────┴────────────────────────────┐
│                    6 MCP Tools                       │
│  StartPolicy │ TerminatePolicy │ SwitchPolicy       │
│  EnvSummary  │ FetchRobotStats │ CallHuman          │
└────────────────────────┬────────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────┐
│               Policy Layer (OpenPI π0/π0.5)          │
│                                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │
│  │ PolicyPool  │  │ PolicyMgr   │  │  Trainer   │  │
│  │ per-subtask │  │ start/stop  │  │ LoRA fine- │  │
│  │ fwd + inv   │  │ switch      │  │ tune       │  │
│  └─────────────┘  └─────────────┘  └────────────┘  │
└────────────────────────┬────────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────┐
│              Environment Adapters                    │
│  K1 Real │ K1 MuJoCo │ Piper Real │ Piper MuJoCo   │
│  Isaac Lab (GPU×512)  │ Gazebo (ROS2)               │
└─────────────────────────────────────────────────────┘
```

## Data Flywheel

```
  ┌──────────────────────────────────────────┐
  │            Iteration i                    │
  │                                          │
  │  1. EAP Collect  ──→  τ_fwd + τ_inv     │
  │  2. Accumulate   ──→  Dataset D          │
  │  3. Export       ──→  LeRobot format     │
  │  4. Train        ──→  LoRA checkpoint    │
  │  5. Reload       ──→  New policy         │
  │                                          │
  │  Deploy trajectories also recycle → D    │
  └──────────────────────────────────────────┘
                    ↓ repeat
```

## Project Structure

```
roboclaw/paper/
├── config.py                 # PaperConfig (Pydantic)
├── integration.py            # Full pipeline factory
├── cli.py                    # CLI entry point
│
├── agent/                    # VLM Agent (Sec 3.1)
│   ├── memory.py             # Structured memory: r_t, g_t, w_t
│   ├── prompts.py            # CoT prompt templates
│   └── loop.py               # Agent main loop
│
├── tools/                    # 6 MCP Tools (Table 1)
│   ├── start_policy.py
│   ├── terminate_policy.py
│   ├── switch_policy.py
│   ├── env_summary.py
│   ├── fetch_robot_stats.py
│   └── call_human.py
│
├── skills/                   # Skills Layer (Sec 3.1)
│   ├── base.py               # RoboticSkill ABC
│   ├── data_collection.py    # EAP orchestration
│   └── long_horizon.py       # Multi-subtask execution
│
├── policy/                   # Policy Layer (Sec 3.2)
│   ├── base.py               # PolicyInterface ABC
│   ├── openpi_client.py      # OpenPI HTTP + WebSocket client
│   ├── manager.py            # Policy lifecycle
│   ├── pool.py               # Multi-subtask policy pool
│   ├── trainer.py            # LoRA fine-tuning pipeline
│   └── mock_policy.py        # Testing
│
├── eap/                      # EAP Engine (Sec 3.2)
│   ├── trajectory.py         # Trajectory + Dataset
│   ├── judge.py              # VLM success judge
│   ├── engine.py             # Forward-inverse loop
│   ├── parallel_engine.py    # GPU-parallel (Isaac Lab)
│   └── demo_import.py        # Human demonstration import
│
├── flywheel/                 # Data Flywheel (Sec 4.2)
│   ├── flywheel.py           # Iterative collect-train loop
│   └── recycler.py           # Deployment data recycling
│
├── deployment/               # Deployment (Sec 3.3)
│   ├── supervisor.py         # Process supervision + failure handling
│   └── executor.py           # Long-horizon task executor
│
├── mcp/                      # MCP Server
│   └── server.py             # Standard MCP protocol exposure
│
└── sim/                      # Environments
    ├── base_env.py            # BaseEnvironment ABC
    ├── tabletop_env.py        # Simple test env
    ├── k1_env.py              # Unitree K1 (ROS2)
    ├── k1_mujoco_env.py       # K1 MuJoCo simulation
    ├── piper_env.py           # Agilex Piper (CAN SDK)
    ├── piper_mujoco_env.py    # Piper MuJoCo simulation
    ├── isaac_lab_env.py       # Isaac Lab GPU-parallel
    └── gazebo_env.py          # Gazebo (ROS2)
```

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{li2026roboclaw,
  title={RoboClaw: An Agentic Framework for Scalable Long-Horizon Robotic Tasks},
  author={Li, Ruiying and Zhou, Yunlang and Zhu, YuYao and Chen, Kylin and Wang, Jingyuan and Wang, Sukai and Hu, Kongtao and Yu, Minhui and Jiang, Bowen and Su, Zhan and Ma, Jiayao and He, Xin and Shen, Yongjian and Yang, Yang and Ren, Guanghui and Yao, Maoqing and Wang, Wenhao and Mu, Yao},
  journal={arXiv preprint arXiv:2603.11558},
  year={2026}
}
```

## Acknowledgments

- **[RoboClaw (MINT-SJTU)](https://github.com/MINT-SJTU/RoboClaw)** — Official framework that provides the foundational agent infrastructure (agent loop, tools, providers, config). TinyRoboClaw extends it with the paper's core algorithms.
- **[OpenPI (Physical Intelligence)](https://github.com/Physical-Intelligence/openpi)** — π0/π0.5 VLA models used as the underlying policy backbone.
- **[TinyZero](https://github.com/Jiayi-Pan/TinyZero)** — Inspiration for the "Tiny" reproduction approach.

## Disclaimer

This is an **independent reproduction** built on top of the [official RoboClaw open-source framework](https://github.com/MINT-SJTU/RoboClaw). The paper-specific algorithms (EAP, flywheel, deployment supervision) were implemented from scratch based on the publicly available paper. We welcome feedback and collaboration from the original authors.

## License

Apache License 2.0
