# Workspace Sensors

Place local-only sensor manifests here when a sensor definition is not yet generic enough to live in framework code.

Export `SENSOR` or `SENSORS` from each Python file so the workspace loader can register them.

Use this directory only when a setup-specific sensor really cannot be described
by an existing framework sensor type plus an assembly attachment.
