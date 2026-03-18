# Workspace Embodied Assets

This directory is for setup-specific embodied assets generated for one user,
one machine, or one lab environment.

## Directory Map

- `intake/`: facts gathered from the user or local inspection
- `robots/`: local-only robot manifests not ready for framework reuse
- `sensors/`: local-only sensor manifests not ready for framework reuse
- `assemblies/`: setup topology and attachments
- `deployments/`: site-specific ROS2 and device values
- `adapters/`: setup-specific adapter bindings
- `simulators/`: world and scenario files for local simulation
- `guides/`: operator and agent playbooks for setup tasks such as ROS2 installation
- `_templates/`: minimal scaffolds for generated Python assets

## Normal Generation Order

1. Write or refine `intake/`
2. Reuse built-in framework definitions where possible
3. Add `robots/` or `sensors/` only when framework coverage is insufficient
4. Generate `assemblies/`
5. Generate `deployments/`
6. Generate `adapters/`
7. Generate `simulators/` only when the setup includes simulation

## Rules

- Prefer importing reusable definitions from `roboclaw.embodied.*`.
- Do not copy framework files into workspace unless the setup is truly local-only.
- Keep setup-specific changes here instead of editing `roboclaw/embodied/`.
- Export `ROBOT`, `SENSOR`, `ASSEMBLY`, `DEPLOYMENT`, `ADAPTER`, `WORLD`, `SCENARIO`, or the plural form so the workspace loader can discover the file.
- Include `WORKSPACE_ASSET = WorkspaceAssetContract(...)` in generated Python files so the loader can validate and migrate them.
- Treat `_templates/` as bare scaffolds. Replace placeholders from intake facts instead of copying them as-is.
