#!/bin/bash
# OpenClaw Gateway Launcher
# Keeps OpenClaw gateway running under caffeinate (prevents sleep)
exec caffeinate -simu openclaw gateway run
