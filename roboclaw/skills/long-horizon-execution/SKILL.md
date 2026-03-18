---
name: long-horizon-execution
description: Execute complex multi-step manipulation tasks by decomposing into subtasks and orchestrating policy primitives with process supervision.
always: false
metadata: '{"roboclaw": {"requires": {}}}'
---

# Long-Horizon Execution Skill

This skill decomposes complex tasks into subtasks and executes them with supervision and recovery.

## Workflow

1. **Decompose** — Given a high-level task instruction, break it into ordered atomic subtasks.
2. **For each subtask:**
   a. **Observe** — Call `env_summary` to get the current scene state.
   b. **Select policy** — Choose the appropriate policy for this subtask from the policy pool.
   c. **Execute** — Call `start_policy` with the subtask instruction.
   d. **Monitor** — Periodically call `env_summary` and `fetch_robot_stats` during execution.
   e. **Evaluate** — After policy finishes, judge if subtask succeeded.
   f. **Handle failure** — See failure handling below.
3. **Verify completion** — After all subtasks, do a final `env_summary` to confirm the overall task.

## Failure Handling

When a subtask fails, classify the failure:

### Non-degrading failure (scene unchanged)
- Retry the same policy (up to 3 times).
- Example: robot didn't reach the object but nothing fell.

### Degrading failure (scene worsened)
- Switch to inverse policy to recover the scene.
- Call `switch_policy(inverse)` → `start_policy` with recovery instruction.
- After recovery, retry the original subtask.

### Unrecoverable
- Call `call_human` with a description of the situation.
- Wait for human response before continuing.

## Key Rules

1. Always decompose before executing — never try to run a complex task as one policy.
2. Monitor between every subtask — call `env_summary` to confirm preconditions.
3. Never skip a failed subtask — either recover or call for help.
4. Track subtask progress — report which subtasks succeeded/failed.
5. Deployment data is valuable — all trajectories (success or failure) are collected.
6. Use `fetch_robot_stats` before starting any policy to verify robot state.

## Example Task Decomposition

Task: "Organize the vanity table"
Subtasks:
1. "Pick up the lipstick from the left side"
2. "Place the lipstick in the lipstick holder"
3. "Pick up the brush from the center"
4. "Place the brush in the brush cup"
