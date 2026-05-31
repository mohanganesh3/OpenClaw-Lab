#!/usr/bin/env node

import { readFile, writeFile, mkdir } from "node:fs/promises";
import { existsSync } from "node:fs";
import { basename, dirname, resolve } from "node:path";
import { homedir } from "node:os";

const API_BASE = "https://api.sarvam.ai";
const CHAT_BASE = "https://api.sarvam.ai/v1";
const DEFAULT_KEY_FILE = `${homedir()}/.openclaw/secrets/sarvam-api-key.txt`;

const MIME_BY_EXT = {
  ".wav": "audio/wav",
  ".mp3": "audio/mpeg",
  ".m4a": "audio/mp4",
  ".mp4": "audio/mp4",
  ".flac": "audio/flac",
  ".ogg": "audio/ogg",
  ".opus": "audio/opus",
  ".aac": "audio/aac",
  ".webm": "audio/webm",
};

function usage() {
  console.log(`Sarvam CLI for OpenClaw voice workflows.

Usage:
  node tools/sarvam_cli.mjs <command> [options]

Commands:
  chat              Chat completion via sarvam-105b
  tts               Text to speech via bulbul:v3
  stt               Speech to text via saaras:v3 modes
  stt-translate     Speech to English text via speech-to-text-translate
  translate         Text translation
  transliterate     Script transliteration
  detect-language   Text language/script detection

Common options:
  --plain           Print only the main text/path
  --key-file PATH   Defaults to ${DEFAULT_KEY_FILE}

Run a command with --help for examples.`);
}

function commandHelp(command) {
  const examples = {
    chat: `node tools/sarvam_cli.mjs chat --text "Say hello in Hindi" --plain`,
    tts: `node tools/sarvam_cli.mjs tts --text "Welcome to OpenClaw Lab" --output artifacts/sarvam-demo.wav --plain`,
    stt: `node tools/sarvam_cli.mjs stt --file artifacts/sarvam-demo.wav --mode transcribe --plain`,
    "stt-translate": `node tools/sarvam_cli.mjs stt-translate --file artifacts/sarvam-demo.wav --plain`,
    translate: `node tools/sarvam_cli.mjs translate --text "I am building a research agent" --source-language en-IN --target-language hi-IN --plain`,
    transliterate: `node tools/sarvam_cli.mjs transliterate --text "मैं रिसर्च एजेंट बना रहा हूँ" --source-language hi-IN --target-language en-IN --plain`,
    "detect-language": `node tools/sarvam_cli.mjs detect-language --text "मैं रिसर्च एजेंट बना रहा हूँ" --plain`,
  };
  console.log(examples[command] ?? "Unknown command.");
}

function parseArgs(argv) {
  const [command, ...tokens] = argv;
  const opts = { _: [] };
  for (let i = 0; i < tokens.length; i += 1) {
    const token = tokens[i];
    if (!token.startsWith("--")) {
      opts._.push(token);
      continue;
    }
    const key = token.slice(2);
    if (key === "plain" || key === "help" || key === "spoken-form" || key === "timestamps" || key === "enable-preprocessing") {
      opts[key] = true;
      continue;
    }
    const value = tokens[i + 1];
    if (value === undefined || value.startsWith("--")) {
      throw new Error(`Missing value for --${key}`);
    }
    opts[key] = value;
    i += 1;
  }
  return { command, opts };
}

function requireOpt(opts, key) {
  if (!opts[key]) {
    throw new Error(`Missing required option --${key}`);
  }
  return opts[key];
}

async function loadKey(opts) {
  if (opts["api-key"]) {
    return String(opts["api-key"]).trim();
  }
  const envKey = process.env.SARVAM_API_KEY || process.env.SARVAM_API_SUBSCRIPTION_KEY;
  if (envKey) {
    return envKey.trim();
  }
  const keyFile = opts["key-file"] || DEFAULT_KEY_FILE;
  if (existsSync(keyFile)) {
    return (await readFile(keyFile, "utf8")).trim();
  }
  throw new Error(`Sarvam API key not found. Set SARVAM_API_KEY or create ${DEFAULT_KEY_FILE}`);
}

async function requestJson(url, opts, { payload, body, headers = {} } = {}) {
  const apiKey = await loadKey(opts);
  const response = await fetch(url, {
    method: "POST",
    headers: {
      "api-subscription-key": apiKey,
      ...(payload ? { "Content-Type": "application/json" } : {}),
      ...headers,
    },
    body: payload ? JSON.stringify(payload) : body,
  });
  const text = await response.text();
  let json;
  try {
    json = JSON.parse(text);
  } catch {
    throw new Error(`Sarvam returned non-JSON response: ${text.slice(0, 500)}`);
  }
  if (!response.ok) {
    throw new Error(`${response.status} from Sarvam: ${JSON.stringify(json)}`);
  }
  return json;
}

function printResult(result, plain) {
  if (plain !== undefined && plain !== null) {
    console.log(plain);
    return;
  }
  console.log(JSON.stringify(result, null, 2));
}

function mimeFor(path) {
  const lower = path.toLowerCase();
  const ext = Object.keys(MIME_BY_EXT).find((suffix) => lower.endsWith(suffix));
  return ext ? MIME_BY_EXT[ext] : "application/octet-stream";
}

async function audioForm(opts, extraFields) {
  const filePath = resolve(requireOpt(opts, "file"));
  const form = new FormData();
  for (const [key, value] of Object.entries(extraFields)) {
    if (value !== undefined && value !== null && value !== "") {
      form.append(key, String(value));
    }
  }
  const audioBytes = await readFile(filePath);
  form.append("file", new Blob([audioBytes], { type: mimeFor(filePath) }), basename(filePath));
  return form;
}

async function cmdChat(opts) {
  const result = await requestJson(`${CHAT_BASE}/chat/completions`, opts, {
    payload: {
      model: opts.model || "sarvam-105b",
      messages: [{ role: "user", content: requireOpt(opts, "text") }],
      temperature: Number(opts.temperature ?? 0.2),
      max_tokens: Number(opts["max-tokens"] ?? 512),
    },
  });
  const text = result.choices?.[0]?.message?.content;
  printResult(result, opts.plain ? text : undefined);
}

async function cmdTts(opts) {
  const payload = {
    text: requireOpt(opts, "text"),
    target_language_code: opts["target-language"] || "en-IN",
    model: opts.model || "bulbul:v3",
    speaker: opts.speaker || "shubh",
    pace: Number(opts.pace ?? 1.0),
    temperature: Number(opts.temperature ?? 0.6),
    speech_sample_rate: Number(opts["sample-rate"] ?? 24000),
    output_audio_codec: opts.codec || "wav",
  };
  if (payload.model === "bulbul:v2") {
    payload.enable_preprocessing = Boolean(opts["enable-preprocessing"]);
    if (opts.pitch) payload.pitch = Number(opts.pitch);
    if (opts.loudness) payload.loudness = Number(opts.loudness);
  }
  const result = await requestJson(`${API_BASE}/text-to-speech`, opts, { payload });
  const audio = result.audios?.[0];
  if (!audio) {
    throw new Error(`TTS response did not include audio: ${JSON.stringify(result)}`);
  }
  const output = resolve(requireOpt(opts, "output"));
  await mkdir(dirname(output), { recursive: true });
  await writeFile(output, Buffer.from(audio, "base64"));
  result.output_path = output;
  printResult(result, opts.plain ? output : undefined);
}

async function cmdStt(opts) {
  const form = await audioForm(opts, {
    model: opts.model || "saaras:v3",
    mode: opts.mode || "transcribe",
    language_code: opts.language || "unknown",
    prompt: opts.prompt,
    input_audio_codec: opts["input-audio-codec"],
    with_timestamps: opts.timestamps ? "true" : undefined,
  });
  const result = await requestJson(`${API_BASE}/speech-to-text`, opts, { body: form });
  printResult(result, opts.plain ? result.transcript : undefined);
}

async function cmdSttTranslate(opts) {
  const form = await audioForm(opts, {
    model: opts.model || "saaras:v2.5",
    prompt: opts.prompt,
    input_audio_codec: opts["input-audio-codec"],
  });
  const result = await requestJson(`${API_BASE}/speech-to-text-translate`, opts, { body: form });
  printResult(result, opts.plain ? result.transcript : undefined);
}

async function cmdTranslate(opts) {
  const payload = {
    input: requireOpt(opts, "text"),
    source_language_code: opts["source-language"] || "auto",
    target_language_code: requireOpt(opts, "target-language"),
    model: opts.model || "sarvam-translate:v1",
    mode: opts.mode || "formal",
    numerals_format: opts["numerals-format"] || "international",
  };
  if (opts["speaker-gender"]) payload.speaker_gender = opts["speaker-gender"];
  if (opts["output-script"]) payload.output_script = opts["output-script"];
  const result = await requestJson(`${API_BASE}/translate`, opts, { payload });
  printResult(result, opts.plain ? result.translated_text : undefined);
}

async function cmdTransliterate(opts) {
  const result = await requestJson(`${API_BASE}/transliterate`, opts, {
    payload: {
      input: requireOpt(opts, "text"),
      source_language_code: requireOpt(opts, "source-language"),
      target_language_code: requireOpt(opts, "target-language"),
      spoken_form: Boolean(opts["spoken-form"]),
    },
  });
  printResult(result, opts.plain ? result.transliterated_text : undefined);
}

async function cmdDetectLanguage(opts) {
  const result = await requestJson(`${API_BASE}/text-lid`, opts, {
    payload: { input: requireOpt(opts, "text") },
  });
  const plain = [result.language_code, result.script_code].filter(Boolean).join(" ");
  printResult(result, opts.plain ? plain : undefined);
}

const handlers = {
  chat: cmdChat,
  tts: cmdTts,
  stt: cmdStt,
  "stt-translate": cmdSttTranslate,
  translate: cmdTranslate,
  transliterate: cmdTransliterate,
  "detect-language": cmdDetectLanguage,
};

try {
  const { command, opts } = parseArgs(process.argv.slice(2));
  if (!command || opts.help || command === "help") {
    command ? commandHelp(command) : usage();
    process.exit(0);
  }
  const handler = handlers[command];
  if (!handler) {
    usage();
    throw new Error(`Unknown command: ${command}`);
  }
  await handler(opts);
} catch (error) {
  console.error(`error: ${error.message}`);
  process.exit(1);
}
