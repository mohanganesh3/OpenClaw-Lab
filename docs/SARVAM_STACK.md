# Sarvam Stack In OpenClaw Lab

## Configured

- OpenClaw default LLM: `sarvam/sarvam-105b`
- OpenClaw audio transcription hook: Sarvam `saaras:v3`
- Local CLI: `tools/sarvam_cli.mjs`
- Secret source: `~/.openclaw/secrets/sarvam-api-key.txt`

## Voice APIs

### Text To Speech

```bash
node tools/sarvam_cli.mjs tts \
  --text "Welcome to OpenClaw Lab" \
  --target-language en-IN \
  --speaker shubh \
  --output artifacts/sarvam-demo.wav \
  --plain
```

Primary model: `bulbul:v3`

Useful languages: `hi-IN`, `bn-IN`, `ta-IN`, `te-IN`, `gu-IN`, `kn-IN`, `ml-IN`, `mr-IN`, `pa-IN`, `od-IN`, `en-IN`

### Speech To Text

```bash
node tools/sarvam_cli.mjs stt \
  --file artifacts/sarvam-demo.wav \
  --mode transcribe \
  --language unknown \
  --plain
```

Primary model: `saaras:v3`

Modes:

- `transcribe`: output in detected/source script
- `translate`: translate speech to English
- `verbatim`: preserve more speech detail
- `translit`: Romanized transcript
- `codemix`: mixed-language transcript

### Speech Translate

```bash
node tools/sarvam_cli.mjs stt-translate \
  --file artifacts/sarvam-demo.wav \
  --plain
```

This is useful for a voice-agent flow where incoming Indian-language speech should become English for research-agent reasoning.

## Text APIs

### Translation

```bash
node tools/sarvam_cli.mjs translate \
  --text "I am building a research agent" \
  --source-language en-IN \
  --target-language hi-IN \
  --plain
```

Primary model: `sarvam-translate:v1`

### Transliteration

```bash
node tools/sarvam_cli.mjs transliterate \
  --text "मैं रिसर्च एजेंट बना रहा हूँ" \
  --source-language hi-IN \
  --target-language en-IN \
  --plain
```

### Language Detection

```bash
node tools/sarvam_cli.mjs detect-language \
  --text "मैं रिसर्च एजेंट बना रहा हूँ" \
  --plain
```

## OpenClaw Audio Hook

OpenClaw now runs this command for command-backed audio transcription:

```bash
/opt/homebrew/bin/node /Users/mohanganesh/OpenClaw-Lab/tools/sarvam_cli.mjs stt --file "{{MediaPath}}" --mode transcribe --language unknown --plain
```

That makes Sarvam the first-line transcription layer for audio ingestion.

## Demo Direction

The strongest first demo is not only a voice chatbot. It should be a multilingual research-agent interface:

1. User speaks in Hindi, English, or code-mixed speech.
2. Sarvam STT converts voice to text.
3. OpenClaw/Sarvam LLM plans the research step.
4. The agent reads papers, extracts claims, and proposes experiments.
5. Sarvam TTS speaks the summary back in the chosen Indian language.

This gives the project a distinctive India-first research-lab agent angle.
