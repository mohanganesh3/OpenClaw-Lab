// pm2 ecosystem — Full OpenClaw Stack
// Manages 2 persistent services:
//   1. openclaw-gateway  — OpenClaw WebSocket Gateway (brain/backbone)
//   2. openresearchos    — Autonomous 24/7 research daemon
//
// Both run under caffeinate -simu to prevent Mac sleep even on battery/lid closed.
// pm2 auto-restarts both on crash. launchd auto-restarts pm2 on reboot.

module.exports = {
  apps: [
    // ─── 1. OpenClaw Gateway ─────────────────────────────────────────────────
    {
      name: "openclaw-gateway",
      script: "/Users/mohanganesh/OpenClaw-Lab/start_gateway.sh",
      interpreter: "bash",

      autorestart: true,
      watch: false,
      max_restarts: 9999,
      restart_delay: 5000,       // 5s — gateway should recover fast
      min_uptime: "10s",

      out_file: "/Users/mohanganesh/.openclaw/logs/gateway.log",
      error_file: "/Users/mohanganesh/.openclaw/logs/gateway_error.log",
      merge_logs: true,
      log_date_format: "YYYY-MM-DD HH:mm:ss",

      kill_timeout: 5000,

      env: {
        HOME: "/Users/mohanganesh",
        PATH: process.env.PATH,
        NODE_ENV: "production",
      },
    },

    // ─── 2. OpenResearchOS Research Daemon ───────────────────────────────────
    {
      name: "openresearchos",
      script: "/Users/mohanganesh/OpenClaw-Lab/openresearchos/start_daemon.sh",
      interpreter: "bash",

      autorestart: true,
      watch: false,
      max_restarts: 9999,
      restart_delay: 15000,      // 15s — give gateway time to be up first
      min_uptime: "30s",

      out_file: "/Users/mohanganesh/OpenClaw-Lab/openresearchos/logs/daemon.log",
      error_file: "/Users/mohanganesh/OpenClaw-Lab/openresearchos/logs/daemon_error.log",
      merge_logs: true,
      log_date_format: "YYYY-MM-DD HH:mm:ss",

      kill_timeout: 10000,

      env: {
        HOME: "/Users/mohanganesh",
        PATH: process.env.PATH,
        NODE_ENV: "production",
      },
    },
  ],
};
