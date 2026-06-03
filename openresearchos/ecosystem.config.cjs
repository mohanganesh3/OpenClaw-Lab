module.exports = {
  apps: [
    {
      name: "openresearchos",
      script: "/Users/mohanganesh/OpenClaw-Lab/openresearchos/start_daemon.sh",
      interpreter: "bash",

      // Auto-restart on crash
      autorestart: true,
      watch: false,
      max_restarts: 9999,
      restart_delay: 15000,      // wait 15s before restarting
      min_uptime: "30s",

      // Logging
      out_file: "/Users/mohanganesh/OpenClaw-Lab/openresearchos/logs/daemon.log",
      error_file: "/Users/mohanganesh/OpenClaw-Lab/openresearchos/logs/daemon_error.log",
      merge_logs: true,
      log_date_format: "YYYY-MM-DD HH:mm:ss",

      // Give time to finish Telegram messages on shutdown
      kill_timeout: 10000,

      env: {
        NODE_ENV: "production",
        HOME: "/Users/mohanganesh",
        PATH: process.env.PATH,
      },
    },
  ],
};
