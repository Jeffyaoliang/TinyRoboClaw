# Workspace Robots

Place local-only robot manifests here when a robot definition is not yet generic enough to live in framework code.

Export `ROBOT` or `ROBOTS` from each Python file so the workspace loader can register them.

Use this directory sparingly. For a first run, prefer reusing a built-in robot
id first and only add a local robot manifest when the framework does not yet
cover the embodiment well enough.
