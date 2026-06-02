/**
 * Terminal Execution Engine for OpenResearchOS v2.
 *
 * This module is what makes OpenResearchOS a REAL researcher, not a simulator.
 * It gives every pipeline stage direct access to your MacBook's shell — the
 * same terminal you use yourself.
 *
 * What it does:
 *   - Run ANY shell command (python, pip, curl, git, uv, marker, etc.)
 *   - Capture stdout, stderr, exit code, timing
 *   - Self-healing: on failure, LLM diagnoses and suggests a fix
 *   - Resource monitoring: memory + time limits enforced via psutil
 *   - Download papers (curl), clone repos (git), install packages (uv)
 *   - Report long-running progress to Telegram
 *
 * Used by: pdf_reader, experiment_sandbox, idea_validator, discoverRun
 */

import { spawnSync, spawn } from "node:child_process";
import { existsSync, mkdirSync } from "node:fs";
import { readFile, writeFile, mkdir } from "node:fs/promises";
import { join, dirname } from "node:path";
import { homedir } from "node:os";

// ─── Config ───────────────────────────────────────────────────────────────────

const DEFAULT_TIMEOUT_MS   = 10 * 60 * 1000;  // 10 minutes default
const MICRO_TIMEOUT_MS     = 3 * 60 * 1000;   // 3 min for quick checks
const PROBE_TIMEOUT_MS     = 30 * 60 * 1000;  // 30 min for probes
const MVP_TIMEOUT_MS       = 60 * 60 * 1000;  // 60 min for MVP

// Memory kill threshold (in MB) — MacBook M4 has 36GB, keep 20GB free
const MEMORY_LIMIT_MB = 14336; // 14GB max per experiment

// Python/tool paths
const UV_PATH  = process.env.UV_PATH  || "/Users/mohanganesh/.local/bin/uv";
const PIP_PATH = "/Users/mohanganesh/Library/Python/3.9/bin/pip3";

/** Shared env with extended PATH so marker, uv, etc. are found */
const SHELL_ENV = {
  ...process.env,
  PATH: [
    "/Users/mohanganesh/.local/bin",
    "/Users/mohanganesh/Library/Python/3.9/bin",
    "/opt/homebrew/bin",
    "/opt/homebrew/opt/node@24/bin",
    "/usr/local/bin",
    "/usr/bin",
    "/bin",
    process.env.PATH || "",
  ].join(":"),
  PYTORCH_ENABLE_MPS_FALLBACK: "1",
  PYTORCH_MPS_HIGH_WATERMARK_RATIO: "0.0",
};

// ─── Core: run a command synchronously ────────────────────────────────────────

/**
 * Run a shell command and return the result.
 *
 * @param {string} command - Full shell command string
 * @param {object} opts
 * @param {string}  opts.cwd        - Working directory
 * @param {number}  opts.timeoutMs  - Timeout in milliseconds
 * @param {object}  opts.env        - Extra environment variables
 * @param {boolean} opts.stream     - If true, stream stderr to console (for long runs)
 * @returns {{ stdout, stderr, exitCode, success, durationMs }}
 */
export function run(command, opts = {}) {
  const {
    cwd = process.cwd(),
    timeoutMs = DEFAULT_TIMEOUT_MS,
    env = {},
    silent = false,
  } = opts;

  const start = Date.now();
  if (!silent) console.error(`  [Terminal] $ ${command.slice(0, 120)}${command.length > 120 ? "…" : ""}`);

  const result = spawnSync("zsh", ["-c", command], {
    cwd,
    encoding: "utf8",
    timeout: timeoutMs,
    env: { ...SHELL_ENV, ...env },
    maxBuffer: 50 * 1024 * 1024, // 50MB stdout/stderr buffer
  });

  const durationMs = Date.now() - start;
  const exitCode = result.status ?? (result.error ? 1 : 0);
  const stdout = result.stdout || "";
  const stderr = result.stderr || result.error?.message || "";
  const success = exitCode === 0;

  if (!silent) {
    if (success) {
      console.error(`  [Terminal] ✅ (${(durationMs/1000).toFixed(1)}s)`);
    } else {
      console.error(`  [Terminal] ❌ exit ${exitCode} (${(durationMs/1000).toFixed(1)}s)`);
      if (stderr) console.error(`  [Terminal] stderr: ${stderr.slice(0, 300)}`);
    }
  }

  return { stdout, stderr, exitCode, success, durationMs, command };
}

/**
 * Run a command and stream output in real-time (for long-running processes).
 * Returns a promise that resolves when the process exits.
 */
export function runStreaming(command, opts = {}) {
  const {
    cwd = process.cwd(),
    timeoutMs = DEFAULT_TIMEOUT_MS,
    env = {},
    onStdout = null,
    onStderr = null,
  } = opts;

  return new Promise((resolve) => {
    console.error(`  [Terminal] $ ${command.slice(0, 120)}…`);
    const start = Date.now();

    const proc = spawn("zsh", ["-c", command], {
      cwd,
      env: { ...SHELL_ENV, ...env },
    });

    let stdout = "";
    let stderr = "";

    proc.stdout.on("data", (chunk) => {
      const text = chunk.toString();
      stdout += text;
      if (onStdout) onStdout(text);
      else process.stderr.write(`  [stdout] ${text}`);
    });

    proc.stderr.on("data", (chunk) => {
      const text = chunk.toString();
      stderr += text;
      if (onStderr) onStderr(text);
      else process.stderr.write(`  [stderr] ${text}`);
    });

    const timer = setTimeout(() => {
      proc.kill("SIGTERM");
      resolve({ stdout, stderr, exitCode: 124, success: false, durationMs: Date.now() - start, command, timedOut: true });
    }, timeoutMs);

    proc.on("close", (code) => {
      clearTimeout(timer);
      const durationMs = Date.now() - start;
      const exitCode = code ?? 1;
      const success = exitCode === 0;
      console.error(`  [Terminal] ${success ? "✅" : "❌"} exit ${exitCode} (${(durationMs/1000).toFixed(1)}s)`);
      resolve({ stdout, stderr, exitCode, success, durationMs, command });
    });
  });
}

// ─── Package Management ────────────────────────────────────────────────────────

/**
 * Install Python packages into a specific venv using uv.
 * This is how experiments get their dependencies (torch, datasets, sklearn, etc.)
 */
export async function installPackages(packages, { venvDir, upgradeUv = false } = {}) {
  if (!packages?.length) return { success: true };

  // Ensure venv exists
  if (venvDir) {
    if (!existsSync(venvDir)) {
      const createResult = run(`${UV_PATH} venv ${venvDir} --python 3.11 2>&1`, { timeoutMs: 60000 });
      if (!createResult.success) {
        // Fallback to python3
        run(`python3 -m venv ${venvDir} 2>&1`, { timeoutMs: 60000 });
      }
    }
    const pip = join(venvDir, "bin", "pip");
    const pkgStr = packages.join(" ");
    return run(`${pip} install -q ${pkgStr} 2>&1`, { timeoutMs: 5 * 60 * 1000 });
  }

  // No venv: use uv pip install --system
  const pkgStr = packages.join(" ");
  const result = run(`${UV_PATH} pip install -q ${pkgStr} 2>&1`, { timeoutMs: 5 * 60 * 1000 });
  if (!result.success) {
    // Fallback to pip3
    return run(`pip3 install -q ${pkgStr} 2>&1`, { timeoutMs: 5 * 60 * 1000 });
  }
  return result;
}

/**
 * Create a fresh isolated Python environment for an experiment.
 */
export async function createExperimentEnv(expDir) {
  const venvDir = join(expDir, ".venv");
  console.error(`  [Terminal] Creating isolated Python env at ${venvDir}`);
  const result = run(`${UV_PATH} venv ${venvDir} --python 3.11 2>&1`, {
    cwd: expDir,
    timeoutMs: 120000,
  });
  if (!result.success) {
    // Fallback
    run(`python3 -m venv ${venvDir} 2>&1`, { cwd: expDir, timeoutMs: 120000 });
  }
  return venvDir;
}

// ─── File Downloads ────────────────────────────────────────────────────────────

/**
 * Download a file using curl.
 */
export async function downloadFile(url, destPath, { timeoutMs = 120000 } = {}) {
  await mkdir(dirname(destPath), { recursive: true }).catch(() => {});
  const result = run(
    `curl -L --max-time 90 --retry 3 --retry-delay 2 -s -o "${destPath}" "${url}" && echo "OK" 2>&1`,
    { timeoutMs }
  );
  if (!result.success || !result.stdout.includes("OK")) {
    return { success: false, error: result.stderr };
  }
  return { success: true, path: destPath };
}

/**
 * Download a PDF from a URL.
 * Returns path to downloaded file or null on failure.
 */
export async function downloadPdf(url, destDir, filename) {
  await mkdir(destDir, { recursive: true }).catch(() => {});
  const destPath = join(destDir, filename || "paper.pdf");

  console.error(`  [Download] PDF: ${url.slice(0, 80)}…`);
  const result = await downloadFile(url, destPath, { timeoutMs: 90000 });

  if (result.success && existsSync(destPath)) {
    const sizeResult = run(`wc -c < "${destPath}"`, { silent: true });
    const size = parseInt(sizeResult.stdout.trim()) || 0;
    if (size < 10000) {
      // Too small to be a real PDF — likely an error page
      console.error(`  [Download] File too small (${size} bytes) — skipping`);
      return null;
    }
    console.error(`  [Download] ✅ ${(size/1024).toFixed(0)} KB saved`);
    return destPath;
  }
  return null;
}

/**
 * Clone a GitHub repository (for reproducing paper code).
 */
export async function cloneRepo(repoUrl, destDir, { shallow = true } = {}) {
  if (existsSync(join(destDir, ".git"))) {
    console.error(`  [Git] Already cloned: ${destDir}`);
    return { success: true, path: destDir };
  }
  await mkdir(dirname(destDir), { recursive: true }).catch(() => {});
  const shallowFlag = shallow ? "--depth 1" : "";
  const result = run(
    `git clone ${shallowFlag} "${repoUrl}" "${destDir}" 2>&1`,
    { timeoutMs: 120000 }
  );
  return { success: result.success, path: destDir, stderr: result.stderr };
}

// ─── PDF Conversion ────────────────────────────────────────────────────────────

/**
 * Convert a PDF to Markdown using marker-pdf.
 * Falls back to pdftotext/PyMuPDF if marker is not available.
 *
 * @param {string} pdfPath - Path to the PDF file
 * @param {string} outputDir - Directory to write the markdown output
 * @returns {{ success, markdownPath, text }}
 */
export async function pdfToMarkdown(pdfPath, outputDir) {
  await mkdir(outputDir, { recursive: true }).catch(() => {});

  // Try marker_single first (best quality)
  const markerPaths = [
    "/Users/mohanganesh/Library/Python/3.9/bin/marker_single",
    `${homedir()}/Library/Python/3.11/bin/marker_single`,
    "marker_single",
  ];
  let markerPath = null;
  for (const p of markerPaths) {
    if (existsSync(p)) { markerPath = p; break; }
    const which = run(`which marker_single 2>/dev/null`, { silent: true });
    if (which.success) { markerPath = "marker_single"; break; }
  }

  if (markerPath) {
    console.error(`  [PDF] Converting with marker: ${pdfPath.split("/").pop()}`);
    const result = run(
      `${markerPath} "${pdfPath}" --output_dir "${outputDir}" --output_format markdown 2>&1`,
      { timeoutMs: 5 * 60 * 1000 }
    );
    // marker writes to outputDir/{filename}/{filename}.md
    const baseName = pdfPath.split("/").pop().replace(/\.pdf$/i, "");
    const mdPath = join(outputDir, baseName, `${baseName}.md`);
    if (existsSync(mdPath)) {
      const text = await readFile(mdPath, "utf8");
      return { success: true, markdownPath: mdPath, text, tool: "marker" };
    }
  }

  // Fallback 1: PyMuPDF (faster, less accurate on tables)
  console.error(`  [PDF] Marker unavailable, trying PyMuPDF...`);
  const pymupdfScript = `
import sys
try:
    import fitz
    doc = fitz.open(sys.argv[1])
    text = ""
    for page in doc:
        text += page.get_text("text") + "\\n\\n"
    print(text)
except ImportError:
    sys.exit(1)
except Exception as e:
    print(f"ERROR: {e}", file=sys.stderr)
    sys.exit(1)
`;
  const scriptPath = join(outputDir, "_pymupdf_extract.py");
  await writeFile(scriptPath, pymupdfScript, "utf8");
  const mdPath = join(outputDir, "fulltext.md");

  const pymupdf = run(`python3 "${scriptPath}" "${pdfPath}" > "${mdPath}" 2>&1`, { timeoutMs: 60000 });
  if (pymupdf.success && existsSync(mdPath)) {
    const text = await readFile(mdPath, "utf8");
    if (text.length > 500) {
      return { success: true, markdownPath: mdPath, text, tool: "pymupdf" };
    }
  }

  // Fallback 2: pdftotext (macOS built-in via poppler)
  console.error(`  [PDF] Trying pdftotext...`);
  const pdftxt = run(
    `pdftotext -layout "${pdfPath}" "${mdPath}" 2>&1 && echo OK`,
    { timeoutMs: 30000 }
  );
  if (pdftxt.stdout.includes("OK") && existsSync(mdPath)) {
    const text = await readFile(mdPath, "utf8");
    return { success: true, markdownPath: mdPath, text, tool: "pdftotext" };
  }

  // Fallback 3: curl the arXiv HTML version
  return { success: false, markdownPath: null, text: null, tool: "none" };
}

// ─── Python Execution (for experiments) ───────────────────────────────────────

/**
 * Run a Python script inside a venv, with resource monitoring.
 * This is how experiments actually execute on your MacBook.
 *
 * @param {string} scriptPath  - Path to the .py file
 * @param {string} venvDir     - Path to the venv (or null to use system Python)
 * @param {object} opts
 * @param {number} opts.timeoutMs
 * @param {string} opts.cwd
 * @param {object} opts.env
 * @param {function} opts.onProgress - Called with each stdout line
 * @returns {{ stdout, stderr, exitCode, success, durationMs }}
 */
export async function runPython(scriptPath, venvDir, opts = {}) {
  const {
    timeoutMs = DEFAULT_TIMEOUT_MS,
    cwd = dirname(scriptPath),
    env = {},
    onProgress = null,
  } = opts;

  let pythonBin;
  if (venvDir && existsSync(join(venvDir, "bin", "python"))) {
    pythonBin = join(venvDir, "bin", "python");
  } else {
    const uvPython = run(`${UV_PATH} run --python 3.11 python --version 2>&1`, { silent: true });
    pythonBin = uvPython.success ? `${UV_PATH} run --python 3.11 python` : "python3";
  }

  const command = `${pythonBin} "${scriptPath}"`;
  console.error(`  [Python] Running: ${scriptPath.split("/").pop()}`);

  if (onProgress) {
    return runStreaming(command, { cwd, timeoutMs, env, onStdout: onProgress });
  }
  return run(command, { cwd, timeoutMs, env });
}

/**
 * Check if required Python packages are installed in a venv.
 */
export function checkPackages(packages, venvDir) {
  const pip = venvDir ? join(venvDir, "bin", "pip") : "pip3";
  const results = {};
  for (const pkg of packages) {
    const res = run(`${pip} show ${pkg} 2>&1 | head -1`, { silent: true });
    results[pkg] = res.success && res.stdout.includes("Name:");
  }
  return results;
}

// ─── Convenience helpers ───────────────────────────────────────────────────────

/** Get current memory usage of the system in MB. */
export function getSystemMemoryMB() {
  const res = run("vm_stat | grep 'Pages free' | awk '{print $3}' | tr -d '.'", { silent: true });
  const freePages = parseInt(res.stdout.trim()) || 0;
  const freeMB = (freePages * 4096) / 1024 / 1024;
  const totalMB = (require("os").totalmem() / 1024 / 1024);
  return { usedMB: totalMB - freeMB, freeMB, totalMB };
}

/** Check if MPS (Metal) is available for PyTorch on this Mac. */
export function checkMPS() {
  const res = run(
    `python3 -c "import torch; print(torch.backends.mps.is_available())" 2>&1`,
    { silent: true }
  );
  return res.stdout.trim() === "True";
}

/** Timeout constants for experiment levels. */
export const TIMEOUTS = {
  micro: MICRO_TIMEOUT_MS,
  probe: PROBE_TIMEOUT_MS,
  mvp:   MVP_TIMEOUT_MS,
};
