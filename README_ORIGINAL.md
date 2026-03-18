<h1>
  <img src="assets/roboclaw_icon.png" alt="RoboClaw icon" width="84" />
  RoboClaw
</h1>

<p>
  <img src="https://img.shields.io/badge/status-early_stage-orange" alt="status">
  <img src="https://img.shields.io/badge/open-source-blue" alt="open source">
  <img src="https://img.shields.io/badge/community-co--create-green" alt="community">
  <img src="https://img.shields.io/badge/focus-embodied_ai-black" alt="focus">
</p>

**RoboClaw** is an open-source embodied intelligence assistant framework.

## ✨ What We Want To Build

Imagine a setup where you can take a robot arm, add a camera, spend a few minutes calibrating it, and then instruct it in natural language to complete a concrete task.

In the short term, we want to establish a practical paradigm: different embodiments, environments, and tasks should all be able to plug into the system through a unified semantic interface, with deployable implementations added step by step.

In the long term, we want RoboClaw to do more than execute tasks. It should also participate in training and analysis: judging whether a task is complete, evaluating execution quality, identifying when and why failures happen, helping recover the scene after failure, and analyzing what can be improved in both reasoning and execution. The goal is to support the continuous development of embodied intelligence systems, not just complete a single task once.

The immediate objective is narrower and more concrete:

- let a first-time user talk to RoboClaw
- let RoboClaw discover the setup and write setup-specific assets into workspace
- let RoboClaw complete `connect / calibrate / move / debug / reset`
- keep the framework extensible enough to onboard many open-source embodiments

The current embodied stack is organized around:

- `Agent`: dialogue understanding, setup guidance, workspace authoring, procedure selection
- `Embodied Definition Plane`: schema, robots, sensors, assemblies, deployments, simulators
- `Embodied Execution Plane`: carriers, transports, adapters, bridges, runtime, procedures
- `Workspace Assets`: user-specific setup files under `~/.roboclaw/workspace/embodied/`

## 🌱 Current Status

RoboClaw is still at a very early stage.

Right now, we are mainly working on:

- building the first end-to-end embodied execution pipeline
- making the workspace-first setup flow actually usable
- validating the critical path from natural language to ROS2-backed execution

Key documents:

- [Current embodied framework](./embodiment-ros2-architecture.md)
- [Installation guide](./roboclaw-installation-guide.md)
- [ROS2 install playbook for RoboClaw](./roboclaw/templates/embodied/guides/ROS2_INSTALL.md)

The direction is clear, and we will continue making the process public as we move forward.

## 📦 Installation

If you like using AI, you can simply ask your coding assistant:

```text
Help me install RoboClaw from https://github.com/MINT-SJTU/RoboClaw
```

If you prefer a manual step-by-step setup, follow the [installation guide](./roboclaw-installation-guide.md).

### Conda

```bash
conda create -n roboclaw python=3.11 -y
conda activate roboclaw
pip install -e ".[dev]"
```

## 📢 Community Co-Creation

RoboClaw aims to move forward in a genuinely open and collaborative way.

That means some important decisions that should not be made by maintainers alone will be discussed openly, with the community invited to participate where possible, such as:

- which real robots to support first
- which simulation platforms to support first
- the priority of the first batch of core features
- what should be prioritized on the roadmap

If you care about embodied AI architecture, capability abstraction, execution pipelines, or robot integration, you can contribute through:

- `Issues`: bug reports, feature requests, and implementation suggestions
- `Pull Requests`: direct code or documentation improvements
- `GitHub Discussions`: conversations around direction, design, and usage

Areas where contributions are especially useful right now include:

- embodied AI architecture design
- capability abstraction and semantic skill interfaces
- ROS2 and execution-layer integration
- simulator support
- real robot adaptation
- evaluation and validation
- documentation and developer experience

## ❓ FAQ

### What is RoboClaw?

RoboClaw is an open-source project for embodied intelligence assistants. It focuses not only on model capability, but also on embodiment abstraction, skill interfaces, execution supervision, simulation integration, and real robot deployment.

### What is the relationship between RoboClaw and OpenClaw?

OpenClaw and several lightweight agent projects have been important inspirations for us. But RoboClaw is not focused on being a general assistant. Its emphasis is on embodiment, skills, execution supervision, simulation, and real-world robot deployment.

### How is this different from the usual "goal understanding -> planning -> sub-skills -> execution" approach?

The pipeline itself is not new. Many projects are already exploring it. What is different is that RoboClaw does not want to hard-code "sub-skills" into a fixed set of capabilities, nor bind the execution side to a single robot. We care more about building an extensible connection between semantics and action, so different embodiments can access, implement, and extend their own skills under one shared paradigm.

### Will RoboClaw support multi-robot scenarios?

RoboClaw will support multi-robot scenarios in the future, but that is not the top priority in the first stage. Right now, we are more focused on getting single-robot capability abstraction, semantic skill interfaces, execution supervision, and the simulation-to-real pipeline into solid shape before moving into more complex problems such as multi-robot coordination, task allocation, and state synchronization.

## 🗺️ To-do List

- [x] Set up the open-source repository, publish the initial README, and add GitHub-native proposal entry points
- [ ] Launch GitHub Discussions and start the first logo / icon community vote
- [x] Add the first embodied framework architecture document
- [ ] Define unified embodied capabilities and semantic interfaces
- [ ] Run the first real setup end to end with workspace-generated assets
- [ ] Support the first real robot platform
- [ ] Design safe-stop and recovery mechanisms
- [ ] ...

Coming soon.

## 🤝 Collaboration Notes

RoboClaw is still in an early stage, and the immediate focus is on making the core functionality and execution pipeline solid.

If you care about embodied intelligence, robot execution systems, simulation platforms, capability abstraction, or evaluation, you are welcome to join through issues, discussions, or PRs.

If you want to be an active contributor, please contact us by emailing bozhaonanjing [[@]] gmail [[DOT]] com

## 🙏 Acknowledgments

RoboClaw references and inherits part of its initial thinking from [nanobot](https://github.com/HKUDS/nanobot). We appreciate its lightweight practice along the OpenClaw line, which helped us build the first prototype faster and continue evolving toward embodied intelligence.
