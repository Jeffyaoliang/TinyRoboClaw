# Embodied Framework

`roboclaw/embodied/` contains the framework side of RoboClaw's embodied stack.

It exists to keep robot-facing code out of the top-level `roboclaw/` package.
Reusable robot manifests, shared contracts, runtime logic, and ROS2-facing
execution abstractions live here. User-specific setups do not.

## Package Split

```text
embodied/
  definition/
    foundation/schema/
    components/
      robots/
      sensors/
    systems/
      assemblies/
      deployments/
      simulators/
  execution/
    integration/
      carriers/
      transports/
      adapters/
      bridges/
    orchestration/
      runtime/
      procedures/
    observability/
      telemetry/
  catalog.py
  workspace.py
```

- `definition/` describes embodied things statically.
- `execution/` describes how those things are selected, driven, and observed.
- `catalog.py` merges built-in framework definitions with workspace assets.
- `workspace.py` validates and loads user-generated embodied assets.

## Runtime Shape

```text
user request
  -> agent guidance
  -> workspace assets
  -> build_catalog(workspace)
  -> runtime session
  -> procedure
  -> adapter / bridge
  -> ROS2
  -> real or simulated embodiment
```

This package owns only the reusable side of that chain. Concrete lab setups,
device paths, namespaces, and scenario files belong under
`~/.roboclaw/workspace/embodied/`.

## Boundary

- Put reusable robot and sensor definitions in `definition/components/`.
- Put static composition contracts in `definition/systems/`.
- Put ROS2-facing execution contracts in `execution/integration/`.
- Put session control and reusable flows in `execution/orchestration/`.
- Put traces, diagnostics, and state snapshots in `execution/observability/`.
- Put user-specific assemblies, deployments, adapters, and simulator assets in the workspace, not here.

## Current Product Goal

The current framework is optimized for one near-term goal first:

- help a first-time user complete `connect / calibrate / move / debug / reset`

Broader goals such as cross-embodiment skills and research workflows remain
extension directions. This package should keep those paths open without
hardcoding them into the first-run stack.
