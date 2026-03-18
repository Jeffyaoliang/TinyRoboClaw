---
name: embodied-setup
description: Guide RoboClaw through workspace-first robot setup generation and refinement.
always: true
---

# Embodied Setup

Use this skill when RoboClaw is helping a user create or refine a robot setup.

## What This Skill Does

- reads the workspace embodied policy
- starts intake early
- reuses built-in framework definitions when possible
- writes setup-specific assets into `~/.roboclaw/workspace/embodied/`

## Rules

1. Read `EMBODIED.md` in the active workspace before creating files.
2. Treat `roboclaw/embodied/` as framework code and the workspace as the place for setup-specific assets.
3. As soon as the user identifies the robot class or model, create or update `embodied/intake/<slug>.md`.
4. If the robot already exists in framework code, inspect its manifest and setup hints before asking follow-up questions.
5. Assume the current integration path is ROS2.
6. Infer obvious facts from framework code, the local environment, and existing workspace files before asking the user.
7. Ask only one small next-step question at a time.
8. Keep ids stable so later turns refine the same setup.
9. If ROS2 is missing or clearly not initialized, read `embodied/guides/ROS2_INSTALL.md` in the active workspace and follow that playbook instead of inventing ad-hoc install steps.

## What To Generate

Generate or refine only the files needed for the current setup:

- `embodied/robots/`
- `embodied/sensors/`
- `embodied/assemblies/`
- `embodied/deployments/`
- `embodied/adapters/`
- `embodied/simulators/`

Use `_templates/` as a scaffold, not as final content.

## What Not To Do

- Do not put user-specific setup files under `roboclaw/embodied/`.
- Do not ask the user to choose between ROS2 and SDK paths.
- Do not ask for namespaces, topics, package paths, or device identifiers before they are needed.
- Do not front-load a large questionnaire.
- Do not put robot-specific defaults into this skill.
- Do not freehand a ROS2 installation recipe from memory when the workspace guide already covers the user's platform.
