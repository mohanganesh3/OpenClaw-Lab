# OpenClaw Trajectory Export

After running this workflow inside an OpenClaw session, export the trajectory from OpenClaw.

Suggested OpenClaw action:

```text
/export-trajectory run_20260601143340_traceable-autonomous-research-agen_9d75ea
```

If using the OpenClaw UI, use the trajectory export control and copy the resulting bundle path into this run directory.

Expected export destination:

```text
/Users/mohanganesh/OpenClaw-Lab/openresearchos/runs/run_20260601143340_traceable-autonomous-research-agen_9d75ea/trajectory_export
```

This local runner records artifacts, logs, metrics, and decisions. The OpenClaw trajectory export records the tool-level conversation and agent actions.
