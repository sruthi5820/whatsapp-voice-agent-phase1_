# Voice App – Phase 1 (STT + TTS)

A standalone Malayalam voice-processing service (mixed Malayalam-English is supported for TTS): **Speech-to-Text** (Qwen2-Audio, Whisper, AI4Bharat IndicConformer) and **Text-to-Speech** (`ai4bharat/indic-parler-tts`, with `facebook/mms-tts-mal` as a lighter fallback), exposed through a FastAPI backend with a simple web frontend.

This is **Phase 1** of a WhatsApp AI agent. WhatsApp, n8n and LLM-agent integration are intentionally *not* included; they come in Phase 2.

## Features

- `POST /stt` – upload audio, choose a model, get text back
- `POST /tts` – send Malayalam text, get a WAV file back
- `GET /health` – health check (also lists registered/loaded models)
- Swagger UI at `/docs`
- Responsive frontend (tabs for STT and TTS, loading states, error messages, audio player and download)
- Browser **Phone Call** mode: the AI greets you first, then listens/responds automatically using silence detection until you press **End Call**; you can also type your reply in the **You said** box and press **Send typed message**
- Plugin-style model registries, so new models can be added without touching the API

## Project structure

```
voice-app/
├── stt/        # base_stt.py (interface), whisper_stt.py, indicconformer_stt.py, registry in __init__.py
├── tts_models/ # base_tts.py (interface), maya_tts.py, mms_tts.py, registry in __init__.py
├── api/        # main.py (app, CORS, errors), stt_routes.py, tts_routes.py, schemas.py
├── frontend/   # index.html, style.css, script.js
├── outputs/    # generated audio (git-ignored)
├── tests/      # test_stt.py, test_tts.py
├── requirements.txt, .env.example, .gitignore
```

## Requirements

- Python 3.10+
- **ffmpeg** (needed to decode mp3 / m4a / webm / opus uploads)
  - macOS: `brew install ffmpeg` · Ubuntu: `sudo apt install ffmpeg` · Windows: `winget install ffmpeg`
- ~2–24 GB of disk for model weights (downloaded from Hugging Face on first use and cached outside the repo)
- A GPU is optional; models fall back to CPU automatically

## Installation

```bash
git clone <your-repo-url> whatsapp-voice-agent-phase1
cd whatsapp-voice-agent-phase1

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
cp .env.example .env               # then edit if needed
```

## Model setup

Weights are **not** stored in the repo. By default, models download automatically the first time they are used. For Whisper, you can download the snapshot into this machine before starting the API, then run from that local copy (including offline):

```bash
source .venv/bin/activate
python scripts/download_whisper.py
```

This saves the configured Whisper model (`openai/whisper-small` by default) to `models/whisper-small`. For Qwen2-Audio, download the checkpoint once and configure its local directory:

```bash
hf download Qwen/Qwen2-Audio-7B-Instruct --local-dir "/media/ganesh/New Volume/models/Qwen2-Audio-7B-Instruct"
```

Set `QWEN_AUDIO_MODEL_ID` to that directory and `DEFAULT_STT_MODEL=qwen_audio` in `.env`. Qwen2-Audio needs a recent Transformers release and Accelerate; both are included in `requirements.txt`. Model weights load lazily on the first transcription, or can be loaded at startup with `PRELOAD_MODELS=qwen_audio`.

| Model | Hugging Face ID | Notes |
|---|---|---|
| Qwen2-Audio | `Qwen/Qwen2-Audio-7B-Instruct` (default) | Set `QWEN_AUDIO_MODEL_ID` to the local checkpoint path; GPU recommended. |
| Whisper | `openai/whisper-small` | Set `WHISPER_MODEL_ID=openai/whisper-large-v3` before downloading for better Malayalam accuracy (needs more RAM/GPU) |
| IndicConformer | `ai4bharat/indic-conformer-600m-multilingual` | Loaded with `trust_remote_code=True`. If you get a 401/403, accept the model's terms on its Hugging Face page and set `HF_TOKEN` in `.env`. If loading fails on a missing package, check the model card for its current requirements. |
| Maya 2 Native | `maya-research/maya2-native` (default) | Streaming Malayalam/Indic TTS via the `maya3` engine; needs Linux + NVIDIA CUDA 12.8. |
| Indic Parler-TTS | `ai4bharat/indic-parler-tts` | Natural Malayalam speech; GPU recommended. |
| MMS TTS | `facebook/mms-tts-mal` | Lightweight VITS fallback, outputs 16 kHz audio. |

> The IndicConformer integration follows the usage shown on its model card. It has not been run end-to-end in the environment this project was generated in, so test it on your machine and adjust `stt/indicconformer_stt.py` if the model's interface has changed.

### Your model configuration

The defaults in `.env` are set for your stack:

| Stage | Model | Where it runs | Env var |
|-------|-------|---------------|---------|
| STT | `Qwen/Qwen2-Audio-7B-Instruct` | Local checkpoint | `QWEN_AUDIO_MODEL_ID` (local HF path or Hub ID) |
| LLM | `google/gemma-4-E4B-it` | AWS GPU server (vLLM) | `LLM_HOST`, `LLM_MODEL`, `LLM_API_STYLE=openai`, optional `LLM_API_KEY` |
| TTS | `maya-research/maya2-native` | Local NVIDIA GPU (CUDA 12.8) | `MAYA_TTS_MODEL_ID` (local release directory), `MAYA_TTS_VOICE`, `MAYA_TTS_LANG`, `MAYA_TTS_NORMALIZE`, `MAYA_TTS_MAX_SEATS` |

If you already downloaded `whisper-large-v3` to your PC via HF, set `WHISPER_MODEL_ID` to that local directory path instead of the Hub ID so it loads offline (`HF_HUB_OFFLINE=1`). The STT/TTS code auto-detects CUDA, so both will run on your local GPU when available.

Download the large model once (the `hf` CLI comes from `huggingface_hub[cli]`):

```bash
hf download openai/whisper-large-v3 --local-dir ~/models/whisper-large-v3
```

Then point the app at it in `.env`:

```bash
WHISPER_MODEL_ID=~/models/whisper-large-v3
HF_HUB_OFFLINE=1
```

`~` is expanded automatically, so both `~/models/...` and absolute paths work.

Point `LLM_HOST` at the AWS instance serving gemma, set `LLM_MODEL` to the bare model name, and use `LLM_API_STYLE=openai` for vLLM/TGI (OpenAI-compatible) or `LLM_API_STYLE=ollama` for Ollama. `LLM_HOST` may include `/v1` (it is normalized automatically), and `LLM_API_KEY` is sent as a bearer token when set:

```bash
LLM_HOST=https://chatbotdevgpu.infospica.com
LLM_MODEL=google/gemma-4-E4B-it
LLM_API_STYLE=openai
LLM_API_KEY=
```

## Running

```bash
uvicorn api.main:app --reload --port 8000
```

- Frontend: http://localhost:8000
- Swagger UI: http://localhost:8000/docs
- Health: http://localhost:8000/health

## API usage

### `POST /stt`

Multipart form: `file` (required), `model` (`qwen_audio` | `whisper` | `indicconformer`, optional), `language` (default `ml`).

```bash
curl -X POST http://localhost:8000/stt \
  -F "file=@sample.wav" \
  -F "model=whisper"
```

```json
{ "text": "…", "model": "whisper", "language": "ml", "processing_seconds": 3.42 }
```

Supported formats: wav, mp3, ogg, opus, m4a, mp4, flac, webm, aac (max `MAX_UPLOAD_MB`, default 25). The browser frontend can also record from your microphone; microphone access requires localhost or HTTPS.

### `POST /tts`

JSON body; returns `audio/wav`.

```bash
curl -X POST http://localhost:8000/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "നമസ്കാരം, സുഖമാണോ?", "model": "mms"}' \
  --output speech.wav
```

### Errors

All errors share one shape:

```json
{ "error": "Unsupported audio format", "detail": "Got '.txt'. Allowed: .aac, .flac, ..." }
```

| Status | Meaning |
|---|---|
| 400 | Unsupported format, empty file, or unknown model |
| 413 | Upload larger than `MAX_UPLOAD_MB` |
| 422 | Invalid request (e.g. empty or too-long text, missing file) |
| 500 | Transcription / speech generation failed |
| 503 | Model could not be loaded (missing weights, no network, auth) |

## Configuration

All settings are environment variables; see `.env.example`. Key ones: `DEFAULT_STT_MODEL`, `QWEN_AUDIO_MODEL_ID`, `QWEN_AUDIO_MAX_NEW_TOKENS`, `DEFAULT_TTS_MODEL`, `DEFAULT_LLM_MODEL`, `WHISPER_MODEL_ID`, `INDICCONFORMER_DECODER` (`ctc`/`rnnt`), `MAYA_TTS_MODEL_ID`, `MAYA_TTS_VOICE`, `MAYA_TTS_LANG`, `MAYA_TTS_NORMALIZE`, `MAYA_TTS_MAX_SEATS`, `MMS_TTS_MODEL_ID`, `INDIC_PARLER_TTS_MODEL_ID`, `INDIC_PARLER_TTS_DEVICE`, `INDIC_PARLER_TTS_DTYPE`, `INDIC_PARLER_TTS_MAX_AUDIO_SECONDS`, `LLM_HOST`, `LLM_MODEL`, `LLM_API_STYLE`, `LLM_API_KEY`, `LLM_MAX_TOKENS`, `CORS_ORIGINS`, `PRELOAD_MODELS`, `OUTPUT_DIR`, `MAX_UPLOAD_MB`, `TTS_MAX_CHARS`.

TTS replies are loudness-normalised before they are returned or streamed (`TTS_NORMALIZE=1` by default) so they match the level of the Maya Research web demo. Tune with `TTS_TARGET_RMS_DBFS` (default `-16`), `TTS_PEAK_DBFS` (default `-1`) and `TTS_MAX_GAIN_DB` (default `18`).

The ARI stream also has VAD / half-duplex tuning: `ARI_VAD_ENERGY_THRESHOLD`, `ARI_VAD_SILENCE_SECONDS`, `ARI_VAD_MIN_SPEECH_SECONDS`, `ARI_VAD_PREROLL_SECONDS`, `ARI_FRAME_MS`, `ARI_ECHO_TAIL_SECONDS`.

## Tests

```bash
pytest -v
```

Tests use fake models, so they run quickly and download nothing.

## Adding a new model

1. Subclass `BaseSTT` (or `BaseTTS`) and implement `_load()` and `_transcribe()` (or `_synthesize()`).
2. Register it in the package's `_BUILTIN` dict, or call `register_stt_model("name", MyClass)` / `register_tts_model(...)`.
3. It is immediately selectable via the `model` parameter, e.g. for Indic Parler-TTS or VITS in later phases.

## Notes

- Generated audio is saved in `outputs/` and never committed. Delete old files whenever you like.
- `CORS_ORIGINS=*` is for local development; set explicit origins before deploying.

## Asterisk ARI voice bridge (Phase 2 telephony)

The full call chain is:

```
Caller (PSTN) → Jio SIP trunk → Asterisk PBX → ARI externalMedia → FastAPI /ari/stream
                                                    │                      STT → LLM → TTS
                                                    └────── PCM audio ◄────────────────────┘
```

`asterisk/ari_bridge.py` is an ARI **Stasis application** that:

1. Subscribes to the `voice-bridge` Stasis app over the ARI events WebSocket.
2. On `StasisStart` (incoming call from the Jio trunk), creates an `externalMedia` channel that Asterisk connects (as WebSocket client) to FastAPI's `/ari/stream`.
3. Puts the caller channel and the `externalMedia` channel into one mixing bridge, so PCM audio flows bidirectionally between the caller and FastAPI.
4. On `StasisEnd` (caller hangs up), hangs up `externalMedia` and destroys the bridge.

FastAPI's `/ari/stream` WebSocket (already implemented in `api/ari_routes.py`) runs the STT → remote LLM → TTS pipeline on the caller's audio and streams PCM reply audio back. It uses energy-based voice activity detection to decide when the caller has finished speaking, runs STT/LLM/TTS in worker threads, and is half-duplex: incoming audio is muted while the bot is synthesising/playing its reply so it does not transcribe its own voice.

### Files under `asterisk/`

| File | Purpose |
|------|---------|
| `pjsip.conf` | Jio SIP trunk endpoint + transport (IP auth; registration block included as an option) |
| `extensions.conf` | Routes inbound Jio calls to `Stasis(voice-bridge)`; test extension `7000` |
| `ari.conf` | ARI user `voice-bridge` used by the Python Stasis app |
| `ari_bridge.py` | Stasis application: event listener + externalMedia channel creation + bridging |

### Asterisk modules required

Enable these in `modules.conf` (or `module reload`):

```
res_pjsip.so
res_pjsip_endpoint_identifier_ip.so
res_ari.so
res_ari_externalmedia.so
```

### Running the bridge

```bash
# 1. Install bridge deps
pip install -r requirements.txt

# 2. Configure Asterisk (copy into /etc/asterisk/) then reload:
cp asterisk/pjsip.conf /etc/asterisk/pjsip.conf
cp asterisk/extensions.conf /etc/asterisk/extensions.conf
cp asterisk/ari.conf /etc/asterisk/ari.conf
asterisk -rx "pjsip reload"
asterisk -rx "dialplan reload"
asterisk -rx "module reload res_ari.so res_ari_externalmedia.so"

# 3. Start FastAPI (media endpoint for /ari/stream)
uvicorn api.main:app --reload --port 8000

# 4. Start the ARI Stasis application (from this repo root)
python asterisk/ari_bridge.py
```

### Environment variables

Added to `.env` / `.env.example` under the ARI section:

- `ARI_HOST`, `ARI_PORT` — Asterisk ARI endpoint (default `localhost:8088`)
- `ARI_APP` — Stasis app name (must match `extensions.conf`, default `voice-bridge`)
- `ARI_USER`, `ARI_PASSWORD` — ARI credentials (must match `ari.conf`)
- `FASTAPI_HOST`, `FASTAPI_PORT`, `FASTAPI_PATH` — media endpoint Asterisk connects to (default `localhost:8000/ari/stream`)
- `ARI_SAMPLE_RATE_IN`, `ARI_SAMPLE_RATE_OUT` — PCM sample rates (default `8000`)
- `ARI_VAD_ENERGY_THRESHOLD`, `ARI_VAD_SILENCE_SECONDS`, `ARI_VAD_MIN_SPEECH_SECONDS`, `ARI_VAD_PREROLL_SECONDS` — end-of-utterance detection
- `ARI_ECHO_TAIL_SECONDS` — how long to keep muting caller input after sending a reply (playback/echo drain)

### Jio SIP trunk

`pjsip.conf` uses **IP authentication** by default (`match=YOUR_JIO_TRUNK_IP` in the `identify` section) — replace with the IP Jio routes calls from. If your Jio trunk uses registration instead, uncomment the `trunk-jio-reg` / `trunk-jio-auth` blocks and set `YOUR_JIO_SIP_PASSWORD`.

## Phase 2 (not implemented here)

WhatsApp Business Cloud API, n8n workflows, an LLM agent, and sending voice replies. The `/stt` and `/tts` endpoints are designed to be called from n8n as plain HTTP requests.
