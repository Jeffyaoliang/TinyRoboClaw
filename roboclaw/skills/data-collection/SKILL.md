---
name: data-collection
description: Autonomous EAP data collection skill — orchestrates forward/inverse policy loops for self-resetting data acquisition.
always: false
metadata: '{"roboclaw": {"requires": {}}}'
---

# Data Collection Skill

This skill orchestrates autonomous data collection using Entangled Action Pairs (EAP).
It manages the full forward-inverse loop cycle, monitors execution, and handles failures.

## Workflow

When activated, follow this procedure:

1. **Observe** — Call `env_summary` to assess the current scene.
2. **Start Forward Policy** — Call `start_policy` with direction="forward" and the task instruction.
3. **Monitor Forward** — Periodically call `fetch_robot_stats` and `env_summary` to monitor progress.
4. **Judge Forward** — After the forward policy completes, assess if the task was accomplished.
5. **Start Inverse Policy** — Call `switch_policy` to direction="inverse", then `start_policy` to reset the scene.
6. **Judge Inverse** — Assess if the scene was successfully reset.
7. **Loop** — Repeat steps 1-6 for the configured number of episodes.

## Tool Sequence

```
env_summary → start_policy(forward) → [monitor] → terminate_policy
           → switch_policy(inverse) → start_policy(inverse) → [monitor] → terminate_policy
           → switch_policy(forward) → [repeat]
```

## Error Handling

- If forward policy fails (no progress after monitoring), call `terminate_policy` and retry once.
- If inverse policy fails (scene not reset), call `call_human` for manual reset.
- If 3 consecutive failures occur, stop and report.

## Key Rules

1. Always observe the scene before starting a policy.
2. Never start a new policy while one is running — always terminate first.
3. After switching direction, wait for confirmation before starting.
4. Monitor at regular intervals (every 5-10 seconds during execution).
5. Record all observations in the reasoning trace for the VLM judge.
