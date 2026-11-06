"""Asterisk ARI WebSocket bridge.

Connects Asterisk (ARI externalMedia / Stasis) to the voice pipeline:

    audio in (PCM)  ->  VAD  ->  STT  ->  LLM  ->  TTS  ->  audio out (PCM)

Asterisk sends audio as binary WebSocket frames (16-bit PCM mono).
Control commands are text JSON frames:

    {"action": "flush"}   - transcribe buffered audio now
    {"action": "end"}     - close the stream

The server replies with text JSON (stt / reply metadata) followed by binary
PCM audio frames the PBX can play back to the caller.

The stream is half-duplex: while a reply is being synthesised and played back,
incoming caller audio is discarded so the bot never transcribes its own voice.
Heavy STT/LLM/TTS work runs in worker threads so the WebSocket keeps reading.
"""
import asyncio
import json
import logging
import os
import tempfile
import uuid
from collections import deque

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from llm import (
    CompletionError,
    LLMModelLoadError,
    UnknownLLMModelError,
    get_llm_model,
)
from stt import (
    ModelLoadError as STTModelLoadError,
    TranscriptionError,
    UnknownSTTModelError,
    get_stt_model,
)
from stt.base_stt import TARGET_SAMPLE_RATE
from tts_models import (
    SynthesisError,
    TTSModelLoadError,
    UnknownTTSModelError,
    get_tts_model,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Asterisk ARI"])

SAMPLE_RATE_IN = int(os.getenv("ARI_SAMPLE_RATE_IN", "8000"))
SAMPLE_RATE_OUT = int(os.getenv("ARI_SAMPLE_RATE_OUT", "8000"))
MAX_BUFFER_SECONDS = float(os.getenv("ARI_MAX_BUFFER_SECONDS", "6.0"))
MIN_UTTERANCE_SECONDS = float(os.getenv("ARI_MIN_UTTERANCE_SECONDS", "0.8"))
LANGUAGE = os.getenv("ARI_LANGUAGE", "ml")

# --- Voice activity detection ---
FRAME_MS = int(os.getenv("ARI_FRAME_MS", "20"))
VAD_ENERGY_THRESHOLD = float(os.getenv("ARI_VAD_ENERGY_THRESHOLD", "0.012"))
VAD_SILENCE_SECONDS = float(os.getenv("ARI_VAD_SILENCE_SECONDS", "0.8"))
VAD_MIN_SPEECH_SECONDS = float(os.getenv("ARI_VAD_MIN_SPEECH_SECONDS", "0.3"))
VAD_PREROLL_SECONDS = float(os.getenv("ARI_VAD_PREROLL_SECONDS", "0.3"))

# How long to keep ignoring caller audio after the reply finishes sending, to
# let Asterisk drain its playback buffer (prevents the bot hearing itself).
ECHO_TAIL_SECONDS = float(os.getenv("ARI_ECHO_TAIL_SECONDS", "0.5"))

DEFAULT_SYSTEM = (
    "You are a helpful voice assistant on a phone call. Reply in Malayalam with "
    "one short sentence of about 15-20 words. English technical words are fine; "
    "natural Malayalam-English mixing is allowed. Keep it natural for "
    "text-to-speech. No markdown, no bullet points, no special characters—plain "
    "spoken text only."
)

BYTES_PER_SAMPLE = 2  # 16-bit PCM
FRAME_BYTES = max(1, int(SAMPLE_RATE_IN * FRAME_MS / 1000)) * BYTES_PER_SAMPLE
BYTES_PER_SECOND = BYTES_PER_SAMPLE * SAMPLE_RATE_IN


def _pcm16_to_float32(data: bytes, sample_rate: int) -> np.ndarray:
    """Convert raw 16-bit PCM bytes to a mono float32 numpy array at 16 kHz."""
    if len(data) % BYTES_PER_SAMPLE != 0:
        data = data[: len(data) - (len(data) % BYTES_PER_SAMPLE)]
    arr = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
    if sample_rate != TARGET_SAMPLE_RATE:
        import librosa

        arr = librosa.resample(arr, orig_sr=sample_rate, target_sr=TARGET_SAMPLE_RATE)
    return arr


def _float32_to_pcm16(audio: np.ndarray, sample_rate: int, target_rate: int) -> bytes:
    """Convert a float32 numpy array to raw 16-bit PCM bytes."""
    audio = np.clip(audio, -1.0, 1.0)
    if sample_rate != target_rate:
        import librosa

        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_rate)
    pcm = (audio * 32767).astype(np.int16)
    return pcm.tobytes()


def _write_wav(audio: np.ndarray, sample_rate: int) -> str:
    """Write float32 audio to a temp WAV file and return its path."""
    import soundfile as sf

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio, sample_rate)
    return tmp.name


def _rms(frame: bytes) -> float:
    if len(frame) < BYTES_PER_SAMPLE:
        return 0.0
    samples = np.frombuffer(frame, dtype="<i2").astype(np.float32) / 32768.0
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples * samples)))


class _UtteranceDetector:
    """Energy-based VAD that buffers one spoken utterance at a time.

    Frames are classified as speech/silence by RMS energy. A short pre-roll
    keeps the first syllable from being clipped, and an utterance ends after
    ``VAD_SILENCE_SECONDS`` of silence or ``MAX_BUFFER_SECONDS`` of speech.
    """

    def __init__(self) -> None:
        self._carry = bytearray()
        preroll_frames = max(1, int(VAD_PREROLL_SECONDS * 1000 / FRAME_MS))
        self._preroll: deque[bytes] = deque(maxlen=preroll_frames)
        self._utterance = bytearray()
        self._speech_active = False
        self._speech_frames = 0
        self._silence_frames = 0

    def push(self, data: bytes) -> bytes | None:
        """Feed audio; return a completed utterance, or ``None``."""
        self._carry.extend(data)
        finished: bytes | None = None
        while len(self._carry) >= FRAME_BYTES:
            frame = bytes(self._carry[:FRAME_BYTES])
            del self._carry[:FRAME_BYTES]
            result = self._push_frame(frame)
            if result is not None:
                finished = result
        return finished

    def _push_frame(self, frame: bytes) -> bytes | None:
        if _rms(frame) >= VAD_ENERGY_THRESHOLD:
            if not self._speech_active:
                self._speech_active = True
                self._utterance.extend(b"".join(self._preroll))
                self._preroll.clear()
            self._utterance.extend(frame)
            self._speech_frames += 1
            self._silence_frames = 0
            if len(self._utterance) >= MAX_BUFFER_SECONDS * BYTES_PER_SECOND:
                return self._finish()
            return None

        if self._speech_active:
            self._utterance.extend(frame)
            self._silence_frames += 1
            if self._silence_frames * FRAME_MS / 1000 >= VAD_SILENCE_SECONDS:
                return self._finish()
            return None

        self._preroll.append(frame)
        return None

    def flush(self) -> bytes:
        """End the current utterance immediately (used by the flush command)."""
        return self._finish()

    def _finish(self) -> bytes:
        utterance = bytes(self._utterance)
        speech_seconds = self._speech_frames * FRAME_MS / 1000
        self._carry.clear()
        self._preroll.clear()
        self._utterance = bytearray()
        self._speech_active = False
        self._speech_frames = 0
        self._silence_frames = 0
        if speech_seconds < VAD_MIN_SPEECH_SECONDS:
            return b""
        return utterance


async def _send_json(ws: WebSocket, payload: dict) -> None:
    await ws.send_text(json.dumps(payload, ensure_ascii=False))


def _transcribe_blocking(stt_model, pcm_bytes: bytes, language: str) -> str:
    audio = _pcm16_to_float32(pcm_bytes, SAMPLE_RATE_IN)
    wav_path = _write_wav(audio, TARGET_SAMPLE_RATE)
    try:
        return stt_model.transcribe(wav_path, language)
    finally:
        try:
            os.remove(wav_path)
        except OSError:
            pass


async def _process_audio(
    ws: WebSocket,
    pcm_bytes: bytes,
    session_id: str,
    stt_model,
    llm_model,
    tts_model,
    system_prompt: str,
) -> float:
    """Run one STT -> LLM -> TTS pass and stream audio back.

    Returns the reply audio duration in seconds (0 when nothing was spoken).
    """
    if len(pcm_bytes) < BYTES_PER_SAMPLE * SAMPLE_RATE_IN * MIN_UTTERANCE_SECONDS:
        await _send_json(
            ws,
            {
                "type": "skip",
                "session": session_id,
                "reason": "utterance too short",
            },
        )
        return 0.0

    # --- STT (blocking; run off the event loop) ---
    try:
        text = await asyncio.to_thread(_transcribe_blocking, stt_model, pcm_bytes, LANGUAGE)
    except STTModelLoadError as exc:
        await _send_json(ws, {"type": "error", "stage": "stt", "detail": str(exc)})
        return 0.0
    except TranscriptionError as exc:
        await _send_json(ws, {"type": "error", "stage": "stt", "detail": str(exc)})
        return 0.0

    if not text or not text.strip():
        await _send_json(ws, {"type": "stt", "session": session_id, "text": ""})
        return 0.0

    await _send_json(
        ws,
        {
            "type": "stt",
            "session": session_id,
            "text": text,
        },
    )
    logger.info("ARI[%s] STT: %s", session_id, text)

    # --- LLM ---
    try:
        reply = await asyncio.to_thread(llm_model.complete, text, system_prompt)
    except LLMModelLoadError as exc:
        await _send_json(ws, {"type": "error", "stage": "llm", "detail": str(exc)})
        return 0.0
    except CompletionError as exc:
        await _send_json(ws, {"type": "error", "stage": "llm", "detail": str(exc)})
        return 0.0

    await _send_json(
        ws,
        {
            "type": "reply",
            "session": session_id,
            "text": reply,
        },
    )
    logger.info("ARI[%s] LLM: %s", session_id, reply)

    # --- TTS ---
    try:
        result = await asyncio.to_thread(tts_model.synthesize, reply)
    except TTSModelLoadError as exc:
        await _send_json(ws, {"type": "error", "stage": "tts", "detail": str(exc)})
        return 0.0
    except SynthesisError as exc:
        await _send_json(ws, {"type": "error", "stage": "tts", "detail": str(exc)})
        return 0.0

    pcm_out = await asyncio.to_thread(
        _float32_to_pcm16, result.waveform, int(result.sampling_rate), SAMPLE_RATE_OUT
    )

    # Stream audio back in chunks so Asterisk can play progressively.
    chunk = 320  # 20 ms @ 8kHz mono PCM16
    for i in range(0, len(pcm_out), chunk):
        await ws.send_bytes(pcm_out[i : i + chunk])
    await _send_json(ws, {"type": "done", "session": session_id})
    duration = len(pcm_out) / (SAMPLE_RATE_OUT * BYTES_PER_SAMPLE)
    logger.info("ARI[%s] sent %.2fs of reply audio", session_id, duration)
    return duration


@router.websocket("/ari/stream")
async def ari_stream(ws: WebSocket) -> None:
    """WebSocket entry point for Asterisk ARI media streams."""
    await ws.accept()
    session_id = str(uuid.uuid4())[:8]
    logger.info("ARI stream connected: %s", session_id)

    # Resolve models lazily (weights load on first use).
    try:
        stt_model = get_stt_model(os.getenv("DEFAULT_STT_MODEL", "whisper"))
        llm_model = get_llm_model(os.getenv("DEFAULT_LLM_MODEL", "remote"))
        tts_model = get_tts_model(os.getenv("DEFAULT_TTS_MODEL", "indic-parler-tts"))
    except (UnknownSTTModelError, UnknownLLMModelError, UnknownTTSModelError) as exc:
        await _send_json(ws, {"type": "error", "stage": "config", "detail": str(exc)})
        await ws.close(code=4000)
        return
    except (STTModelLoadError, LLMModelLoadError, TTSModelLoadError) as exc:
        await _send_json(ws, {"type": "error", "stage": "load", "detail": str(exc)})
        await ws.close(code=5003)
        return

    detector = _UtteranceDetector()
    muted = False
    background: set[asyncio.Task] = set()

    async def handle_utterance(pcm_bytes: bytes, system_prompt: str = DEFAULT_SYSTEM) -> None:
        nonlocal muted
        try:
            duration = await _process_audio(
                ws,
                pcm_bytes,
                session_id,
                stt_model,
                llm_model,
                tts_model,
                system_prompt,
            )
            # Half-duplex: keep discarding caller audio until playback drains.
            await asyncio.sleep(duration + ECHO_TAIL_SECONDS)
        except WebSocketDisconnect:
            pass
        except Exception:
            logger.exception("ARI[%s] failed to process utterance", session_id)
        finally:
            muted = False

    def start_utterance(pcm_bytes: bytes, system_prompt: str = DEFAULT_SYSTEM) -> None:
        nonlocal muted
        if not pcm_bytes or muted:
            return
        muted = True
        task = asyncio.create_task(handle_utterance(pcm_bytes, system_prompt))
        background.add(task)
        task.add_done_callback(background.discard)

    try:
        while True:
            msg = await ws.receive()

            if msg["type"] == "websocket.disconnect":
                break

            if msg.get("bytes"):
                if muted:
                    continue  # discard echo / barge-in while the bot is speaking
                utterance = detector.push(msg["bytes"])
                if utterance is not None:
                    start_utterance(utterance)

            if msg.get("text") is not None:
                try:
                    cmd = json.loads(msg["text"])
                except json.JSONDecodeError:
                    cmd = {}

                action = cmd.get("action")
                if action == "flush":
                    if not muted:
                        start_utterance(detector.flush(), cmd.get("system", DEFAULT_SYSTEM))
                elif action == "end":
                    break
                elif action == "ping" and not muted:
                    await _send_json(ws, {"type": "pong", "session": session_id})

    except WebSocketDisconnect:
        logger.info("ARI stream disconnected: %s", session_id)
    finally:
        for task in list(background):
            task.cancel()
        if background:
            await asyncio.gather(*background, return_exceptions=True)
        if not muted:
            pending = detector.flush()
            if pending:
                try:
                    await _process_audio(
                        ws,
                        pending,
                        session_id,
                        stt_model,
                        llm_model,
                        tts_model,
                        DEFAULT_SYSTEM,
                    )
                except Exception:
                    logger.debug("ARI[%s] final flush failed", session_id, exc_info=True)


@router.get("/ari/status", tags=["Asterisk ARI"])
def ari_status() -> dict:
    """ARI bridge configuration and model availability."""
    from llm import available_llm_models
    from stt import available_stt_models
    from tts_models import available_tts_models

    return {
        "websocket": "/ari/stream",
        "sample_rate_in": SAMPLE_RATE_IN,
        "sample_rate_out": SAMPLE_RATE_OUT,
        "max_buffer_seconds": MAX_BUFFER_SECONDS,
        "language": LANGUAGE,
        "vad": {
            "frame_ms": FRAME_MS,
            "energy_threshold": VAD_ENERGY_THRESHOLD,
            "silence_seconds": VAD_SILENCE_SECONDS,
            "min_speech_seconds": VAD_MIN_SPEECH_SECONDS,
            "preroll_seconds": VAD_PREROLL_SECONDS,
            "echo_tail_seconds": ECHO_TAIL_SECONDS,
        },
        "stt_models": available_stt_models(),
        "llm_models": available_llm_models(),
        "tts_models": available_tts_models(),
    }
