#!/bin/bash
# OpenResearchOS Daemon Launcher
# caffeinate flags:
#   -s  = prevent system sleep (AC power)
#   -i  = prevent idle sleep (battery + AC)
#   -m  = prevent disk sleep
#   -u  = prevent user-idle sleep timing out
#   -t 86400 = hold assertion for 24h then renew (loops forever via exec)
#
# This keeps the Mac awake even on battery with lid closed.
# Combined with pmset sleep=0, this is the maximum possible wakefulness.

while true; do
  caffeinate -simu node /Users/mohanganesh/OpenClaw-Lab/openresearchos/src/research_daemon.mjs "$@"
  echo "[Launcher] Daemon exited — restarting in 10s..."
  sleep 10
done
